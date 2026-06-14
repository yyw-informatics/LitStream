"""Build a run digest: what's new since the last one, plus cost and synthesis.

Pulls from the shared litstream.db (runs, project_papers, papers, source_stats,
cost_events) and the project's synthesis file. "New since last digest" compares
acquired_at against the previous digest's timestamp, so each digest is a delta
rather than a full re-dump.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _month_start() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-01T00:00:00Z")


def _d(cents: float) -> str:
    return f"${cents / 100:,.2f}"


def build_digest(*, db_path: str, project: str, routine: str, project_dir: Path,
                 digests_dir: Path, now: datetime) -> tuple[Path, str]:
    digests_dir.mkdir(parents=True, exist_ok=True)
    prev = sorted(digests_dir.glob(f"{routine}_*.md"))
    # previous digest's timestamp (from its filename) bounds the "new since" delta
    last_ts = "1970-01-01T00:00:00Z"
    if prev:
        stem = prev[-1].stem.rsplit("_", 1)[-1]      # routine_YYYYMMDDTHHMMSSZ
        try:
            last_ts = datetime.strptime(stem, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    new_kept = con.execute(
        """SELECT p.title, p.journal, p.year, p.sources, pp.triage_score, p.pdf_status
           FROM project_papers pp JOIN papers p ON p.paper_id = pp.paper_id
           WHERE pp.project = ? AND pp.status = 'kept' AND pp.acquired_at > ?
           ORDER BY pp.triage_score DESC""", (project, last_ts)).fetchall()
    run = con.execute(
        "SELECT * FROM runs WHERE project = ? ORDER BY started_at DESC LIMIT 1", (project,)).fetchone()
    mtd = con.execute(
        "SELECT COALESCE(SUM(cost_cents),0) c FROM cost_events WHERE occurred_at >= ?",
        (_month_start(),)).fetchone()["c"]
    pol = con.execute("SELECT cap_cents FROM budget_policies WHERE id = 1").fetchone()
    src = con.execute(
        "SELECT source, kept, seen, muted FROM source_stats WHERE routine = ? ORDER BY source",
        (routine,)).fetchall()
    con.close()

    synth = project_dir / f"projects/{project}/literature/0_synthesis_literature.md"
    synth_headers = []
    if synth.is_file():
        synth_headers = [l for l in synth.read_text().splitlines() if l.startswith("## ")][:8]

    L = [f"# LitStream digest — {routine} — {now:%Y-%m-%d}", ""]
    L.append(f"**Project:** {project}  ·  **New papers since last digest:** {len(new_kept)}")
    if run:
        L.append(f"**Last run:** {run['status']}  ·  cost {_d(run['cost_cents'])}  "
                 f"·  {run['papers_new']} acquired")
    cap = f" / {_d(pol['cap_cents'])}" if pol else ""
    L.append(f"**Month-to-date spend:** {_d(mtd)}{cap}")
    L.append("")

    L.append("## New kept papers (ranked by triage)")
    if new_kept:
        for r in new_kept[:25]:
            score = f"{r['triage_score']:.2f}" if r["triage_score"] is not None else "—"
            pdf = "📄" if r["pdf_status"] == "fetched" else "·"
            L.append(f"- **[{score}]** {pdf} {r['title']}  ({r['journal'] or '—'}, {r['year'] or '—'})")
    else:
        L.append("_No new kept papers since the last digest._")
    L.append("")

    if synth_headers:
        L.append("## Synthesis sections")
        L += [f"- {h[3:]}" for h in synth_headers]
        L.append(f"\n_Full synthesis: `projects/{project}/literature/0_synthesis_literature.md`_")
        L.append("")

    if src:
        L.append("## Source quality")
        for r in src:
            rate = (r["kept"] / r["seen"]) if r["seen"] else 0
            flag = " (muted)" if r["muted"] else ""
            L.append(f"- {r['source']}: {r['kept']}/{r['seen']} kept ({rate:.0%}){flag}")
        L.append("")

    md = "\n".join(L)
    path = digests_dir / f"{routine}_{now:%Y%m%dT%H%M%SZ}.md"
    path.write_text(md)
    return path, md
