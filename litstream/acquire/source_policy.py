"""Source-quality auto-mute policy.

Per (routine, source), tracks how many fetched papers triage kept. Once a source
has been seen enough and its keep-rate falls below threshold, it is muted: future
runs skip fetching it entirely, saving both the fetch and the triage of its papers.
To avoid permanent lock-out, a muted source is re-probed every `reprobe_every` runs.

The policy rides on triage's judgment, so no extra LLM call decides sources. arXiv
for a CITE-seq routine self-mutes because its keep-rate is near zero (the relevant
work appears on bioRxiv and in journals, not arXiv).
"""

from __future__ import annotations

from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_stats (
    routine            TEXT NOT NULL,
    source             TEXT NOT NULL,
    seen               INTEGER NOT NULL DEFAULT 0,   -- papers this source returned
    kept               INTEGER NOT NULL DEFAULT 0,   -- of those, triage-kept (score≥thr)
    runs               INTEGER NOT NULL DEFAULT 0,
    muted              INTEGER NOT NULL DEFAULT 0,
    skips_since_probe  INTEGER NOT NULL DEFAULT 0,
    updated_at         TEXT,
    PRIMARY KEY (routine, source)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SourcePolicy:
    def __init__(self, conn, *, min_seen: int = 20, keep_threshold: float = 0.08,
                 reprobe_every: int = 4):
        self.conn = conn
        self.min_seen = min_seen
        self.keep_threshold = keep_threshold       # mute if keep-rate below this
        self.reprobe_every = reprobe_every
        conn.executescript(SCHEMA)

    def _row(self, routine: str, source: str):
        return self.conn.execute(
            "SELECT * FROM source_stats WHERE routine=? AND source=?",
            (routine, source)).fetchone()

    def decide(self, routine: str, candidates: list[str]) -> dict[str, str]:
        """Return {source: 'run'|'reprobe'|'muted_skip'} — side-effect free."""
        out: dict[str, str] = {}
        for s in candidates:
            row = self._row(routine, s)
            if row and row["muted"]:
                out[s] = "reprobe" if (row["skips_since_probe"] + 1) >= self.reprobe_every else "muted_skip"
            else:
                out[s] = "run"
        return out

    def update(self, routine: str, decisions: dict[str, str],
               seen: dict[str, int], kept: dict[str, int]) -> None:
        now = _now()
        for s, decision in decisions.items():
            row = self._row(routine, s)
            base = dict(seen=row["seen"] if row else 0, kept=row["kept"] if row else 0,
                        runs=row["runs"] if row else 0,
                        skips=row["skips_since_probe"] if row else 0)
            if decision in ("run", "reprobe"):
                new_seen = base["seen"] + seen.get(s, 0)
                new_kept = base["kept"] + kept.get(s, 0)
                rate = (new_kept / new_seen) if new_seen else 1.0
                muted = 1 if (new_seen >= self.min_seen and rate < self.keep_threshold) else 0
                self.conn.execute(
                    """INSERT INTO source_stats (routine, source, seen, kept, runs, muted,
                         skips_since_probe, updated_at) VALUES (?,?,?,?,?,?,0,?)
                       ON CONFLICT(routine, source) DO UPDATE SET
                         seen=excluded.seen, kept=excluded.kept, runs=excluded.runs,
                         muted=excluded.muted, skips_since_probe=0, updated_at=excluded.updated_at""",
                    (routine, s, new_seen, new_kept, base["runs"] + 1, muted, now))
            else:  # muted_skip
                self.conn.execute(
                    "UPDATE source_stats SET skips_since_probe=?, updated_at=? WHERE routine=? AND source=?",
                    (base["skips"] + 1, now, routine, s))
        self.conn.commit()

    def status(self, routine: str) -> list:
        return self.conn.execute(
            "SELECT * FROM source_stats WHERE routine=? ORDER BY source", (routine,)).fetchall()
