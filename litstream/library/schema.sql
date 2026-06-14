-- LitStream paper library — global, content-addressed, shared across projects.
-- Lives in the same litstream.db as the cost ledger.

-- One row per real-world paper, deduped by canonical id (see store.canonical_id).
-- Project-INDEPENDENT facts only: metadata, abstract, the PDF. Project-specific
-- triage/mining lives in project_papers.
CREATE TABLE IF NOT EXISTS papers (
    paper_id    TEXT PRIMARY KEY,     -- e.g. "doi:10.1038/s41592-020-01050-x"
    doi         TEXT,                 -- normalized lowercase, no URL prefix
    arxiv_id    TEXT,                 -- base id, no version
    pmid        TEXT,
    s2_id       TEXT,                 -- Semantic Scholar paperId
    title       TEXT,
    abstract    TEXT,
    year        INTEGER,
    journal     TEXT,
    sources     TEXT NOT NULL DEFAULT '[]',   -- JSON list: which APIs yielded it
    pdf_path    TEXT,                 -- library/<paper_id>.pdf once fetched
    pdf_status  TEXT NOT NULL DEFAULT 'none'   -- none|open|fetched|unavailable
                  CHECK (pdf_status IN ('none','open','fetched','unavailable')),
    first_seen  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
-- alternate-id lookups for cross-source dedup (a paper may arrive as arXiv+DOI
-- from one source and DOI+S2 from another)
CREATE INDEX IF NOT EXISTS idx_papers_doi   ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_pmid  ON papers(pmid);
CREATE INDEX IF NOT EXISTS idx_papers_s2    ON papers(s2_id);

-- Per-project view of a paper: the triage decision + lifecycle status. Two
-- projects can reference the same paper_id with different decisions.
CREATE TABLE IF NOT EXISTS project_papers (
    project       TEXT NOT NULL,
    paper_id      TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    triage_label  TEXT,               -- RELEVANT|BORDERLINE|NOT_RELEVANT
    triage_score  REAL,               -- optional rank score (triage RANKS, not hard-cuts)
    triage_model  TEXT,
    status        TEXT NOT NULL DEFAULT 'screened'
                    CHECK (status IN ('screened','kept','dropped','mined')),
    acquired_at   TEXT NOT NULL,
    decided_at    TEXT,
    PRIMARY KEY (project, paper_id)
);
CREATE INDEX IF NOT EXISTS idx_pp_project_status ON project_papers(project, status);
