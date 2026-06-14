"""`litstream costs` — read-only spend report over the ledger.

    python3 -m litstream.ledger.report litstream.db

Shows month-to-date vs budget, utilization, spend by routine / model / phase,
open budget incidents, and the last N runs. No external dependencies.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone


def _month_start() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-01T00:00:00Z")


def _d(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def report(db_path: str, last_n: int = 10) -> None:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    ms = _month_start()

    pol = con.execute("SELECT * FROM budget_policies WHERE id = 1").fetchone()
    mtd = con.execute(
        "SELECT COALESCE(SUM(cost_cents),0) c FROM cost_events WHERE occurred_at >= ?", (ms,)
    ).fetchone()["c"]

    print(f"\n  LitStream cost report  ·  {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    print("  " + "─" * 56)
    if pol:
        cap, warn = pol["cap_cents"], pol["cap_cents"] * pol["warn_pct"] // 100
        util = mtd / cap if cap else 0
        bar_n = min(int(util * 28), 28)
        bar = "█" * bar_n + "·" * (28 - bar_n)
        flag = " ⚠ WARN" if mtd >= warn else ""
        flag = " 🛑 OVER CAP" if mtd >= cap else flag
        print(f"  Month-to-date   {_d(mtd)} / {_d(cap)}   {util:5.1%}{flag}")
        print(f"  [{bar}]  warn at {_d(warn)}")
    else:
        print(f"  Month-to-date   {_d(mtd)}   (no budget policy set)")

    def table(title, rows, key):
        if not rows:
            return
        print(f"\n  {title}")
        for r in rows:
            print(f"    {str(r[key] or '—'):<22} {_d(r['c']):>12}")

    table("By routine (this month)", con.execute(
        """SELECT r.routine key, SUM(e.cost_cents) c FROM cost_events e
           JOIN runs r ON r.id = e.run_id WHERE e.occurred_at >= ?
           GROUP BY r.routine ORDER BY c DESC""", (ms,)).fetchall(), "key")
    table("By model (this month)", con.execute(
        """SELECT model key, SUM(cost_cents) c FROM cost_events
           WHERE occurred_at >= ? GROUP BY model ORDER BY c DESC""", (ms,)).fetchall(), "key")
    table("By phase (this month)", con.execute(
        """SELECT phase key, SUM(cost_cents) c FROM cost_events
           WHERE occurred_at >= ? GROUP BY phase ORDER BY c DESC""", (ms,)).fetchall(), "key")

    inc = con.execute(
        "SELECT threshold, observed_cents, opened_at FROM budget_incidents WHERE status='open' ORDER BY id"
    ).fetchall()
    if inc:
        print("\n  Open incidents")
        for r in inc:
            print(f"    {r['threshold']:<10} at {_d(r['observed_cents'])}  opened {r['opened_at']}")

    runs = con.execute(
        """SELECT id, project, status, cost_cents, papers_new, started_at
           FROM runs ORDER BY started_at DESC LIMIT ?""", (last_n,)).fetchall()
    if runs:
        print(f"\n  Last {len(runs)} runs")
        for r in runs:
            print(f"    {r['started_at']}  {r['status']:<14} {_d(r['cost_cents']):>10}  "
                  f"{r['papers_new']:>3} papers  {r['project']}")
    print()
    con.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 -m litstream.ledger.report <litstream.db> [last_n]")
        raise SystemExit(2)
    report(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
