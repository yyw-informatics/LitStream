"""Unit tests for litstream.ledger.cost.CostLedger.

Pure-logic coverage: fractional-cent costing, cache-token pricing, SDK-cost
override, unknown-model fallback, budget gates / incidents, and run rollups.

Each test uses a fresh temp DB via the tmp_path fixture and closes the ledger.
No network, no real LLM.
"""

from __future__ import annotations

import math

import pytest

from litstream.ledger.cost import BudgetState, CostLedger


@pytest.fixture
def ledger(tmp_path):
    """A CostLedger backed by a throwaway SQLite file, closed on teardown."""
    led = CostLedger(str(tmp_path / "test.db"))
    try:
        yield led
    finally:
        led.close()


def _running_run(led: CostLedger) -> str:
    return led.start_run(project="testproj", routine="r", invocation="manual")


# ---------------------------------------------------------------------------
# Fractional-cent cost
# ---------------------------------------------------------------------------

def test_subcent_cost_is_stored_fractional_not_rounded_up(ledger):
    # $3 / 1M input tokens -> 300 cents per 1M tokens.
    ledger.seed_pricing("m", input_per_mtok=3.0, output_per_mtok=15.0,
                        cached_input_per_mtok=0.3)
    run_id = _running_run(ledger)

    # 100 regular input tokens: 100 * 300 / 1_000_000 = 0.03 cents (sub-cent).
    cost = ledger.record(run_id, model="m", phase="mine", input_tokens=100)

    expected = 100 * 300 / 1_000_000  # 0.03 cents
    assert cost == pytest.approx(expected)
    assert cost == pytest.approx(0.03)
    # The whole point: it must NOT round up to 1 cent.
    assert cost < 1.0

    row = ledger._conn.execute(
        "SELECT cost_cents, cost_source FROM cost_events WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row["cost_cents"] == pytest.approx(0.03)
    assert row["cost_source"] == "computed"


def test_record_returns_and_stores_same_value(ledger):
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)
    cost = ledger.record(run_id, model="m", input_tokens=1234, output_tokens=56)
    stored = ledger._conn.execute(
        "SELECT cost_cents FROM cost_events WHERE run_id = ?", (run_id,)
    ).fetchone()["cost_cents"]
    assert stored == pytest.approx(cost)


# ---------------------------------------------------------------------------
# Cache-token pricing
# ---------------------------------------------------------------------------

def test_cache_token_pricing_formula(ledger):
    # input $3/mtok -> 300 c/mtok; output $15 -> 1500; cached $0.30 -> 30.
    ledger.seed_pricing("m", input_per_mtok=3.0, output_per_mtok=15.0,
                        cached_input_per_mtok=0.3)
    run_id = _running_run(ledger)

    cost = ledger.record(
        run_id, model="m",
        input_tokens=1000,            # regular input @ input rate
        output_tokens=500,            # output @ output rate
        cached_input_tokens=2000,     # cache READ @ cached rate
        cache_creation_tokens=400,    # cache WRITE @ 1.25x input rate
    )

    expected = (
        1000 * 300            # regular input
        + 400 * 300 * 1.25    # cache creation @ 1.25x input rate
        + 2000 * 30           # cache read @ cached rate
        + 500 * 1500          # output
    ) / 1_000_000
    assert expected == pytest.approx(1.26)
    assert cost == pytest.approx(expected)


def test_cache_creation_priced_above_regular_input(ledger):
    # Cache creation (write) must be 1.25x the regular input rate for the same tokens.
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)

    regular = ledger.record(run_id, model="m", input_tokens=1000)
    creation = ledger.record(run_id, model="m", cache_creation_tokens=1000)
    assert creation == pytest.approx(regular * 1.25)


def test_cache_read_cheaper_than_regular_input(ledger):
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)
    regular = ledger.record(run_id, model="m", input_tokens=1000)
    cache_read = ledger.record(run_id, model="m", cached_input_tokens=1000)
    # cached rate 0.3 < input rate 3.0 -> 10x cheaper here.
    assert cache_read == pytest.approx(regular * (0.3 / 3.0))


# ---------------------------------------------------------------------------
# SDK-reported cost wins
# ---------------------------------------------------------------------------

def test_sdk_reported_cost_overrides_computed(ledger):
    # Even with known pricing, total_cost_usd wins and source is sdk_reported.
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)

    cost = ledger.record(
        run_id, model="m",
        input_tokens=999_999, output_tokens=999_999,  # would compute to a lot
        total_cost_usd=0.0123,                          # $0.0123 = 1.23 cents
    )
    assert cost == pytest.approx(1.23)
    row = ledger._conn.execute(
        "SELECT cost_cents, cost_source FROM cost_events WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row["cost_cents"] == pytest.approx(1.23)
    assert row["cost_source"] == "sdk_reported"


def test_sdk_reported_cost_for_unknown_model(ledger):
    # No pricing seeded, but SDK cost present -> still sdk_reported, value used.
    run_id = _running_run(ledger)
    cost = ledger.record(run_id, model="never-seen", input_tokens=10,
                        total_cost_usd=0.05)
    assert cost == pytest.approx(5.0)
    src = ledger._conn.execute(
        "SELECT cost_source FROM cost_events WHERE run_id = ?", (run_id,)
    ).fetchone()["cost_source"]
    assert src == "sdk_reported"


# ---------------------------------------------------------------------------
# Unknown model + no SDK cost
# ---------------------------------------------------------------------------

def test_unknown_model_no_sdk_cost_is_zero_computed(ledger):
    run_id = _running_run(ledger)
    cost = ledger.record(run_id, model="unknown-model", input_tokens=1000,
                        output_tokens=1000)
    assert cost == 0.0
    row = ledger._conn.execute(
        "SELECT cost_cents, cost_source FROM cost_events WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row["cost_cents"] == 0.0
    assert row["cost_source"] == "computed"


# ---------------------------------------------------------------------------
# Run rollup + start/finish status transitions
# ---------------------------------------------------------------------------

def test_run_rollup_sums_tokens_and_cost(ledger):
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)

    ledger.record(run_id, model="m", input_tokens=1000, output_tokens=100,
                  cached_input_tokens=50)
    ledger.record(run_id, model="m", input_tokens=2000, output_tokens=200,
                  cached_input_tokens=25)

    row = ledger._conn.execute(
        "SELECT input_tokens, output_tokens, cached_input_tokens, cost_cents "
        "FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row["input_tokens"] == 3000
    assert row["output_tokens"] == 300
    assert row["cached_input_tokens"] == 75

    expected_cost = (
        (1000 * 300 + 100 * 1500 + 50 * 30) / 1_000_000
        + (2000 * 300 + 200 * 1500 + 25 * 30) / 1_000_000
    )
    assert row["cost_cents"] == pytest.approx(expected_cost)
    # run_cost_cents mirrors the run's rollup.
    assert ledger.run_cost_cents(run_id) == pytest.approx(expected_cost)


def test_run_cost_cents_unknown_run_is_zero(ledger):
    assert ledger.run_cost_cents("does-not-exist") == 0.0


def test_start_run_is_running_then_finish_completed(ledger):
    run_id = _running_run(ledger)
    status = ledger._conn.execute(
        "SELECT status FROM runs WHERE id = ?", (run_id,)
    ).fetchone()["status"]
    assert status == "running"

    ledger.finish_run(run_id, status="completed")
    row = ledger._conn.execute(
        "SELECT status, finished_at FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row["status"] == "completed"
    assert row["finished_at"] is not None


def test_finish_run_records_error_and_papers(ledger):
    run_id = _running_run(ledger)
    ledger.finish_run(run_id, status="failed", error="boom", papers_new=7)
    row = ledger._conn.execute(
        "SELECT status, error, papers_new FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "boom"
    assert row["papers_new"] == 7


def test_start_run_explicit_run_id_is_used(ledger):
    rid = ledger.start_run(project="p", run_id="explicit-id")
    assert rid == "explicit-id"
    assert ledger._conn.execute(
        "SELECT id FROM runs WHERE id = ?", ("explicit-id",)
    ).fetchone() is not None


# ---------------------------------------------------------------------------
# month_to_date_cents
# ---------------------------------------------------------------------------

def test_month_to_date_sums_across_runs(ledger):
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    r1 = ledger.start_run(project="p", run_id="r1")
    r2 = ledger.start_run(project="p", run_id="r2")
    c1 = ledger.record(r1, model="m", input_tokens=1000)   # 0.3 cents
    c2 = ledger.record(r2, model="m", input_tokens=5000)   # 1.5 cents
    assert ledger.month_to_date_cents() == pytest.approx(c1 + c2)


def test_month_to_date_zero_when_empty(ledger):
    assert ledger.month_to_date_cents() == 0.0


# ---------------------------------------------------------------------------
# Budget gates / BudgetState
# ---------------------------------------------------------------------------

def test_no_policy_means_ok_and_zero_cap(ledger):
    state = ledger.preflight()
    assert isinstance(state, BudgetState)
    assert state.ok is True
    assert state.cap_cents == 0
    assert state.tripped is None
    assert state.utilization == 0.0  # cap of 0 -> guarded division


def test_under_warn_is_ok_no_trip(ledger):
    ledger.set_policy(cap_usd=1.00, warn_pct=80)   # cap 100c, warn 80c
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)
    ledger.record(run_id, model="m", input_tokens=100_000)  # 30 cents

    state = ledger.preflight()
    assert state.ok is True
    assert state.tripped is None
    assert state.cap_cents == 100
    assert state.warn_cents == 80
    assert state.observed_cents == pytest.approx(30.0)
    assert state.utilization == pytest.approx(0.30)


def test_warning_trips_at_warn_pct_but_stays_ok(ledger):
    ledger.set_policy(cap_usd=1.00, warn_pct=80)   # warn at 80c
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)
    # 300_000 input tokens -> 90 cents: >= warn (80) but < cap (100).
    ledger.record(run_id, model="m", input_tokens=300_000)

    state = ledger.phase_gate()
    assert state.observed_cents == pytest.approx(90.0)
    assert state.tripped == "warning"
    assert state.ok is True            # warning never blocks
    assert state.utilization == pytest.approx(0.90)


def test_hard_stop_trips_at_cap_and_blocks(ledger):
    ledger.set_policy(cap_usd=1.00, warn_pct=80, hard_stop=True)
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)
    # 400_000 input tokens -> 120 cents: >= cap (100).
    ledger.record(run_id, model="m", input_tokens=400_000)

    state = ledger.preflight()
    assert state.observed_cents == pytest.approx(120.0)
    assert state.tripped == "hard_stop"
    assert state.ok is False
    assert state.utilization == pytest.approx(1.20)


def test_hard_stop_false_stays_ok_even_over_cap(ledger):
    ledger.set_policy(cap_usd=1.00, warn_pct=80, hard_stop=False)
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)
    ledger.record(run_id, model="m", input_tokens=400_000)  # 120c, over cap

    state = ledger.preflight()
    assert state.tripped == "hard_stop"   # still flagged as crossing cap
    assert state.ok is True               # but not blocked when hard_stop disabled


def test_exactly_at_cap_is_hard_stop(ledger):
    ledger.set_policy(cap_usd=1.00, warn_pct=80)
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)
    # 333_334 * 300 / 1e6 = 100.0002 cents -> >= cap.
    ledger.record(run_id, model="m", input_tokens=333_334)
    state = ledger.preflight()
    assert state.observed_cents >= state.cap_cents
    assert state.tripped == "hard_stop"


def test_warn_cents_uses_ceil(ledger):
    # cap 7 cents, warn 80% -> ceil(7*80/100)=ceil(5.6)=6.
    ledger.set_policy(cap_usd=0.07, warn_pct=80)
    state = ledger.preflight()
    assert state.cap_cents == 7
    assert state.warn_cents == math.ceil(7 * 80 / 100)
    assert state.warn_cents == 6


def test_set_policy_upserts_singleton(ledger):
    ledger.set_policy(cap_usd=1.00, warn_pct=80)
    ledger.set_policy(cap_usd=2.50, warn_pct=50, hard_stop=False)
    rows = ledger._conn.execute(
        "SELECT cap_cents, warn_pct, hard_stop FROM budget_policies"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["cap_cents"] == 250
    assert rows[0]["warn_pct"] == 50
    assert rows[0]["hard_stop"] == 0


# ---------------------------------------------------------------------------
# Budget incidents
# ---------------------------------------------------------------------------

def test_warning_opens_incident(ledger):
    ledger.set_policy(cap_usd=1.00, warn_pct=80)
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)
    ledger.record(run_id, model="m", input_tokens=300_000)  # 90c -> warning
    ledger.preflight()

    rows = ledger._conn.execute(
        "SELECT threshold, status, observed_cents, cap_cents FROM budget_incidents"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["threshold"] == "warning"
    assert rows[0]["status"] == "open"
    assert rows[0]["observed_cents"] == pytest.approx(90.0)
    assert rows[0]["cap_cents"] == 100


def test_hard_stop_opens_incident(ledger):
    ledger.set_policy(cap_usd=1.00, warn_pct=80)
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)
    ledger.record(run_id, model="m", input_tokens=400_000)  # 120c -> hard_stop
    ledger.preflight()
    rows = ledger._conn.execute(
        "SELECT threshold FROM budget_incidents"
    ).fetchall()
    assert [r["threshold"] for r in rows] == ["hard_stop"]


def test_repeated_checks_do_not_duplicate_incident(ledger):
    # (threshold, window_start) is UNIQUE -> INSERT OR IGNORE keeps one row.
    ledger.set_policy(cap_usd=1.00, warn_pct=80)
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)
    ledger.record(run_id, model="m", input_tokens=300_000)  # warning
    for _ in range(5):
        ledger.preflight()
    count = ledger._conn.execute(
        "SELECT COUNT(*) AS n FROM budget_incidents WHERE threshold = 'warning'"
    ).fetchone()["n"]
    assert count == 1


def test_no_incident_when_under_warn(ledger):
    ledger.set_policy(cap_usd=1.00, warn_pct=80)
    ledger.seed_pricing("m", 3.0, 15.0, 0.3)
    run_id = _running_run(ledger)
    ledger.record(run_id, model="m", input_tokens=100_000)  # 30c, under warn
    ledger.preflight()
    count = ledger._conn.execute(
        "SELECT COUNT(*) AS n FROM budget_incidents"
    ).fetchone()["n"]
    assert count == 0
