"""Global paper library + per-project references — the dedup core of Acquire.

A paper is stored once (metadata, abstract, PDF), deduped by canonical id, and
shared across projects. Each project keeps its own triage decision. This means two
projects that both surface the same paper never re-download or re-store it, and the
global library doubles as the seen-index.

    store = PaperStore("litstream.db", library_dir="library")
    pid = store.upsert(PaperRecord(doi="10.1038/…", title="…", abstract="…", source="biorxiv"))
    store.add_to_project("citeseq_apoe", pid)            # acquired for this project
    if not store.seen_in_project("citeseq_apoe", pid): ...
    store.set_triage("citeseq_apoe", pid, "RELEVANT", score=0.82, model="deepseek-chat")
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = doi.removeprefix("doi:").strip()
    return doi or None


def norm_arxiv(aid: str | None) -> str | None:
    if not aid:
        return None
    aid = aid.strip().lower().removeprefix("arxiv:")
    aid = re.sub(r"v\d+$", "", aid)            # drop version suffix
    return aid or None


@dataclass
class PaperRecord:
    title: str = ""
    abstract: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    s2_id: str | None = None
    year: int | None = None
    journal: str | None = None
    source: str = ""
    pdf_url: str | None = None

    def ids(self) -> dict:
        return {"doi": norm_doi(self.doi), "arxiv_id": norm_arxiv(self.arxiv_id),
                "pmid": (self.pmid or "").strip() or None,
                "s2_id": (self.s2_id or "").strip() or None}


def canonical_id(ids: dict) -> str | None:
    """Stable id from the strongest available identifier."""
    if ids.get("doi"):      return f"doi:{ids['doi']}"
    if ids.get("arxiv_id"): return f"arxiv:{ids['arxiv_id']}"
    if ids.get("pmid"):     return f"pmid:{ids['pmid']}"
    if ids.get("s2_id"):    return f"s2:{ids['s2_id']}"
    return None


class PaperStore:
    def __init__(self, db_path: str | Path, library_dir: str | Path = "library"):
        self.db_path = str(db_path)
        self.library_dir = Path(library_dir)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA_PATH.read_text())

    # -- dedup + upsert ------------------------------------------------------

    def _find_all_existing(self, ids: dict) -> list[sqlite3.Row]:
        """All DISTINCT existing papers sharing ANY id with `ids` (cross-source
        dedup). Usually 0 or 1; >1 means the incoming record bridges rows that
        were legitimately separate before (no shared id existed yet) and must be
        reconciled. Ordered by id strength (doi>arxiv>pmid>s2) so matches[0] is
        the strongest-identified survivor."""
        seen: set[str] = set()
        rows: list[sqlite3.Row] = []
        for col in ("doi", "arxiv_id", "pmid", "s2_id"):
            if ids.get(col):
                for row in self._conn.execute(
                        f"SELECT * FROM papers WHERE {col} = ?", (ids[col],)).fetchall():
                    if row["paper_id"] not in seen:
                        seen.add(row["paper_id"])
                        rows.append(row)
        return rows

    def _reconcile(self, matches: list[sqlite3.Row], now: str) -> sqlite3.Row:
        """Collapse rows the incoming ids unified into one survivor: fold in each
        loser's ids/metadata/sources, re-point its project links (the survivor's
        own decision wins on a (project, paper_id) collision), drop the loser row
        (ON DELETE CASCADE clears any leftover links). Returns the refreshed survivor."""
        survivor = matches[0]
        spid = survivor["paper_id"]
        merged = {c: survivor[c] for c in ("doi", "arxiv_id", "pmid", "s2_id",
                                           "title", "abstract", "year", "journal")}
        srcs = set(json.loads(survivor["sources"]))
        for loser in matches[1:]:
            for c in merged:
                if not merged[c]:
                    merged[c] = loser[c]
            srcs |= set(json.loads(loser["sources"]))
            self._conn.execute(
                "UPDATE OR IGNORE project_papers SET paper_id=? WHERE paper_id=?",
                (spid, loser["paper_id"]))
            self._conn.execute("DELETE FROM papers WHERE paper_id=?", (loser["paper_id"],))
        self._conn.execute(
            """UPDATE papers SET doi=?, arxiv_id=?, pmid=?, s2_id=?, title=?, abstract=?,
                 year=?, journal=?, sources=?, updated_at=? WHERE paper_id=?""",
            (*merged.values(), json.dumps(sorted(srcs)), now, spid))
        self._conn.commit()
        return self._conn.execute(
            "SELECT * FROM papers WHERE paper_id=?", (spid,)).fetchone()

    def upsert(self, rec: PaperRecord) -> str:
        """Insert a paper or merge into an existing one. Returns its paper_id."""
        now = _now()
        ids = rec.ids()
        matches = self._find_all_existing(ids)
        existing = (self._reconcile(matches, now) if len(matches) > 1
                    else (matches[0] if matches else None))
        if existing:
            pid = existing["paper_id"]
            srcs = set(json.loads(existing["sources"]))
            if rec.source:
                srcs.add(rec.source)
            # fill any missing fields from the new record; keep existing otherwise
            merged = {
                "doi": existing["doi"] or ids["doi"],
                "arxiv_id": existing["arxiv_id"] or ids["arxiv_id"],
                "pmid": existing["pmid"] or ids["pmid"],
                "s2_id": existing["s2_id"] or ids["s2_id"],
                "title": existing["title"] or rec.title,
                "abstract": existing["abstract"] or rec.abstract,
                "year": existing["year"] or rec.year,
                "journal": existing["journal"] or rec.journal,
            }
            self._conn.execute(
                """UPDATE papers SET doi=?, arxiv_id=?, pmid=?, s2_id=?, title=?,
                     abstract=?, year=?, journal=?, sources=?, updated_at=?
                   WHERE paper_id=?""",
                (*merged.values(), json.dumps(sorted(srcs)), now, pid))
            self._conn.commit()
            return pid

        pid = canonical_id(ids)
        if pid is None:
            raise ValueError(f"paper has no usable identifier: {rec.title[:60]!r}")
        self._conn.execute(
            """INSERT INTO papers (paper_id, doi, arxiv_id, pmid, s2_id, title,
                 abstract, year, journal, sources, first_seen, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pid, ids["doi"], ids["arxiv_id"], ids["pmid"], ids["s2_id"], rec.title,
             rec.abstract, rec.year, rec.journal,
             json.dumps([rec.source] if rec.source else []), now, now))
        self._conn.commit()
        return pid

    # -- per-project references ---------------------------------------------

    def add_to_project(self, project: str, paper_id: str) -> bool:
        """Link a paper to a project. Returns False if already present (seen)."""
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO project_papers (project, paper_id, status, acquired_at)
               VALUES (?, ?, 'screened', ?)""", (project, paper_id, _now()))
        self._conn.commit()
        return cur.rowcount > 0

    def seen_in_project(self, project: str, paper_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM project_papers WHERE project=? AND paper_id=?",
            (project, paper_id)).fetchone() is not None

    def set_triage(self, project: str, paper_id: str, label: str,
                   score: float | None = None, model: str | None = None,
                   status: str | None = None) -> None:
        self._conn.execute(
            """UPDATE project_papers SET triage_label=?, triage_score=?, triage_model=?,
                 status=COALESCE(?, status), decided_at=? WHERE project=? AND paper_id=?""",
            (label, score, model, status, _now(), project, paper_id))
        self._conn.commit()

    def set_pdf(self, paper_id: str, path: str, status: str = "fetched") -> None:
        self._conn.execute("UPDATE papers SET pdf_path=?, pdf_status=?, updated_at=? WHERE paper_id=?",
                           (path, status, _now(), paper_id))
        self._conn.commit()

    def project_papers(self, project: str, status: str | None = None) -> list[sqlite3.Row]:
        q = ("""SELECT p.*, pp.triage_label, pp.triage_score, pp.status AS proj_status
                FROM project_papers pp JOIN papers p ON p.paper_id = pp.paper_id
                WHERE pp.project = ?""")
        args: list = [project]
        if status:
            q += " AND pp.status = ?"; args.append(status)
        return self._conn.execute(q, args).fetchall()

    def stats(self) -> dict:
        n_papers = self._conn.execute("SELECT COUNT(*) c FROM papers").fetchone()["c"]
        n_links = self._conn.execute("SELECT COUNT(*) c FROM project_papers").fetchone()["c"]
        return {"papers": n_papers, "project_links": n_links}

    def close(self) -> None:
        self._conn.close()
