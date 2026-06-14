"""LitStream cost ledger — record per-turn usage and enforce the monthly budget.

Pure stdlib (sqlite3). The headless driver calls:

    ledger = CostLedger("litstream.db")
    gate = ledger.preflight()
    run_id = ledger.start_run(project="citeseq_apoe", routine="weekly-citeseq")
    ...
    ledger.record(run_id, model="claude-fable-5", phase="mine",
                  input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                  cached_input_tokens=u.cache_read_input_tokens,
                  total_cost_usd=u.total_cost_usd)
    if not ledger.phase_gate().ok: ...
    ledger.finish_run(run_id, status="completed")

Money is stored as integer cents; tokens as integers. SDK-reported total_cost_usd
is preferred; otherwise cost is computed from model_pricing.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_month_start(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m-01T00:00:00Z")


@dataclass
class BudgetState:
    """Result of a budget check. ok=False means new work must not start/continue."""
    ok: bool
    observed_cents: int
    cap_cents: int
    warn_cents: int
    tripped: str | None  # None | 'warning' | 'hard_stop'

    @property
    def utilization(self) -> float:
        return (self.observed_cents / self.cap_cents) if self.cap_cents else 0.0


class CostLedger:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_PATH.read_text())

    # -- setup ---------------------------------------------------------------

    def set_policy(self, cap_usd: float, warn_pct: int = 80, hard_stop: bool = True) -> None:
        self._conn.execute(
            """INSERT INTO budget_policies (id, cap_cents, warn_pct, hard_stop, updated_at)
               VALUES (1, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 cap_cents=excluded.cap_cents, warn_pct=excluded.warn_pct,
                 hard_stop=excluded.hard_stop, updated_at=excluded.updated_at""",
            (round(cap_usd * 100), warn_pct, 1 if hard_stop else 0, _utcnow_iso()),
        )
        self._conn.commit()

    def seed_pricing(self, model: str, input_per_mtok: float,
                     output_per_mtok: float, cached_input_per_mtok: float) -> None:
        """Prices in USD per 1M tokens; stored as (fractional) cents per 1M tokens."""
        self._conn.execute(
            """INSERT INTO model_pricing
                 (model, input_cents_per_mtok, output_cents_per_mtok,
                  cached_input_cents_per_mtok, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(model) DO UPDATE SET
                 input_cents_per_mtok=excluded.input_cents_per_mtok,
                 output_cents_per_mtok=excluded.output_cents_per_mtok,
                 cached_input_cents_per_mtok=excluded.cached_input_cents_per_mtok,
                 updated_at=excluded.updated_at""",
            (model, input_per_mtok * 100, output_per_mtok * 100,
             cached_input_per_mtok * 100, _utcnow_iso()),
        )
        self._conn.commit()

    # -- runs ----------------------------------------------------------------

    def start_run(self, project: str, routine: str | None = None,
                  invocation: str = "manual", run_id: str | None = None) -> str:
        run_id = run_id or f"{routine or 'adhoc'}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        self._conn.execute(
            """INSERT INTO runs (id, routine, project, invocation, status, started_at)
               VALUES (?, ?, ?, ?, 'running', ?)""",
            (run_id, routine, project, invocation, _utcnow_iso()),
        )
        self._conn.commit()
        return run_id

    def finish_run(self, run_id: str, status: str = "completed",
                   error: str | None = None, papers_new: int | None = None) -> None:
        sets = ["status = ?", "finished_at = ?"]
        args: list = [status, _utcnow_iso()]
        if error is not None:
            sets.append("error = ?"); args.append(error)
        if papers_new is not None:
            sets.append("papers_new = ?"); args.append(papers_new)
        args.append(run_id)
        self._conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", args)
        self._conn.commit()

    # -- cost capture --------------------------------------------------------

    def _cost_cents(self, model: str, input_tokens: int, output_tokens: int,
                    cached_input_tokens: int, total_cost_usd: float | None,
                    cache_creation_tokens: int = 0):
        """Return (cost_in_cents_as_float, source). Fractional, no rounding.

        input_tokens is regular (non-cached) input. The three input classes are
        priced separately:
          - regular input          → input rate
          - cache creation (write) → 1.25 x input rate (Claude's 5-min cache write)
          - cache read (hit)       → cached (discounted) rate
        Providers that don't charge for cache writes report 0 creation tokens.
        """
        if total_cost_usd is not None:
            return total_cost_usd * 100.0, "sdk_reported"
        row = self._conn.execute(
            "SELECT * FROM model_pricing WHERE model = ?", (model,)
        ).fetchone()
        if row is None:
            # Unknown model and no SDK cost: record 0, flagged via cost_source.
            return 0.0, "computed"
        cents = (
            input_tokens * row["input_cents_per_mtok"]
            + cache_creation_tokens * row["input_cents_per_mtok"] * 1.25
            + cached_input_tokens * row["cached_input_cents_per_mtok"]
            + output_tokens * row["output_cents_per_mtok"]
        ) / 1_000_000
        return cents, "computed"

    def record(self, run_id: str, model: str, *, phase: str | None = None,
               role: str | None = None, input_tokens: int = 0, output_tokens: int = 0,
               cached_input_tokens: int = 0, cache_creation_tokens: int = 0,
               total_cost_usd: float | None = None) -> int:
        """Insert one cost_event and update the run's rollup. Returns cost_cents.
        input_tokens = regular (non-cached); cache read/creation priced separately."""
        cost_cents, source = self._cost_cents(
            model, input_tokens, output_tokens, cached_input_tokens, total_cost_usd,
            cache_creation_tokens=cache_creation_tokens,
        )
        self._conn.execute(
            """INSERT INTO cost_events
                 (run_id, phase, role, model, input_tokens, output_tokens,
                  cached_input_tokens, cost_cents, cost_source, occurred_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, phase, role, model, input_tokens, output_tokens,
             cached_input_tokens, cost_cents, source, _utcnow_iso()),
        )
        self._conn.execute(
            """UPDATE runs SET
                 input_tokens = input_tokens + ?,
                 output_tokens = output_tokens + ?,
                 cached_input_tokens = cached_input_tokens + ?,
                 cost_cents = cost_cents + ?
               WHERE id = ?""",
            (input_tokens, output_tokens, cached_input_tokens, cost_cents, run_id),
        )
        self._conn.commit()
        return cost_cents

    # -- budget --------------------------------------------------------------

    def month_to_date_cents(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_cents),0) AS c FROM cost_events WHERE occurred_at >= ?",
            (_utc_month_start(),),
        ).fetchone()
        return float(row["c"])

    def run_cost_cents(self, run_id: str) -> float:
        """Cost accrued so far by one run — for the per-run ceiling check."""
        row = self._conn.execute("SELECT cost_cents FROM runs WHERE id = ?", (run_id,)).fetchone()
        return float(row["cost_cents"]) if row else 0.0

    def _check(self) -> BudgetState:
        pol = self._conn.execute("SELECT * FROM budget_policies WHERE id = 1").fetchone()
        if pol is None:
            return BudgetState(True, self.month_to_date_cents(), 0, 0, None)
        observed = self.month_to_date_cents()
        cap = pol["cap_cents"]
        warn = math.ceil(cap * pol["warn_pct"] / 100)
        tripped = None
        if observed >= cap:
            tripped = "hard_stop"
        elif observed >= warn:
            tripped = "warning"
        if tripped:
            self._open_incident(tripped, cap, observed)
        ok = not (tripped == "hard_stop" and pol["hard_stop"])
        return BudgetState(ok, observed, cap, warn, tripped)

    def preflight(self) -> BudgetState:
        """Call before starting a run. ok=False → do not start (hard-stop hit)."""
        return self._check()

    def phase_gate(self) -> BudgetState:
        """Call between pipeline phases. ok=False → abort the run cleanly."""
        return self._check()

    def _open_incident(self, threshold: str, cap_cents: int, observed_cents: int) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO budget_incidents
                 (threshold, window_start, cap_cents, observed_cents, status, opened_at)
               VALUES (?, ?, ?, ?, 'open', ?)""",
            (threshold, _utc_month_start(), cap_cents, observed_cents, _utcnow_iso()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
