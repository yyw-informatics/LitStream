"""Tests for the content-addressed paper library (litstream.library.store).

Focus: the deduplication / content-addressing core of PaperStore. A single
real-world paper that arrives from different sources under different identifier
classes (DOI / PMID / arXiv / Semantic Scholar) must collapse to ONE library
entry, and per-project linkage must be idempotent with honest "seen" accounting.

No network and no PDF downloads are exercised: every path under test is pure
store/dedup logic. Each test gets its own temp DB + temp library dir.
"""

from __future__ import annotations

import json

import pytest

from litstream.library.store import (
    PaperRecord,
    PaperStore,
    canonical_id,
    norm_arxiv,
    norm_doi,
)


@pytest.fixture
def store(tmp_path):
    s = PaperStore(str(tmp_path / "test.db"), library_dir=str(tmp_path / "lib"))
    yield s
    s.close()


# --------------------------------------------------------------------------
# Identifier normalization (pure functions)
# --------------------------------------------------------------------------

class TestNormDoi:
    def test_none_and_empty(self):
        assert norm_doi(None) is None
        assert norm_doi("") is None
        assert norm_doi("   ") is None

    def test_lowercased(self):
        assert norm_doi("10.1038/S41592-020-01050-X") == "10.1038/s41592-020-01050-x"

    def test_strips_https_doi_org_prefix(self):
        assert norm_doi("https://doi.org/10.1/abc") == "10.1/abc"
        assert norm_doi("http://doi.org/10.1/abc") == "10.1/abc"
        assert norm_doi("https://dx.doi.org/10.1/abc") == "10.1/abc"

    def test_strips_doi_scheme_prefix(self):
        assert norm_doi("doi:10.1/abc") == "10.1/abc"

    def test_surrounding_whitespace(self):
        assert norm_doi("  10.1/ABC  ") == "10.1/abc"


class TestNormArxiv:
    def test_none_and_empty(self):
        assert norm_arxiv(None) is None
        assert norm_arxiv("") is None

    def test_strips_arxiv_scheme_and_version(self):
        assert norm_arxiv("arXiv:2101.00001v2") == "2101.00001"
        assert norm_arxiv("2101.00001v13") == "2101.00001"

    def test_no_version_unchanged(self):
        assert norm_arxiv("2101.00001") == "2101.00001"


class TestCanonicalId:
    def test_priority_doi_over_all(self):
        ids = {"doi": "10.1/x", "arxiv_id": "2101.1", "pmid": "9", "s2_id": "s"}
        assert canonical_id(ids) == "doi:10.1/x"

    def test_priority_arxiv_when_no_doi(self):
        assert canonical_id({"arxiv_id": "2101.1", "pmid": "9", "s2_id": "s"}) == "arxiv:2101.1"

    def test_priority_pmid_when_no_doi_or_arxiv(self):
        assert canonical_id({"pmid": "9", "s2_id": "s"}) == "pmid:9"

    def test_priority_s2_last(self):
        assert canonical_id({"s2_id": "s"}) == "s2:s"

    def test_none_when_no_ids(self):
        assert canonical_id({}) is None
        assert canonical_id({"doi": None, "arxiv_id": None, "pmid": None, "s2_id": None}) is None


# --------------------------------------------------------------------------
# upsert: insert + canonical-id assignment
# --------------------------------------------------------------------------

class TestUpsertInsert:
    def test_returns_canonical_id_from_doi(self, store):
        pid = store.upsert(PaperRecord(doi="10.1038/ABC", title="T", source="biorxiv"))
        assert pid == "doi:10.1038/abc"
        assert store.stats()["papers"] == 1

    def test_stores_normalized_columns(self, store):
        pid = store.upsert(
            PaperRecord(doi="https://doi.org/10.1/X", arxiv_id="arXiv:2101.1v3", source="a")
        )
        row = store._conn.execute(
            "SELECT doi, arxiv_id FROM papers WHERE paper_id=?", (pid,)
        ).fetchone()
        assert row["doi"] == "10.1/x"
        assert row["arxiv_id"] == "2101.1"

    def test_source_recorded_as_json_list(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", source="biorxiv"))
        row = store._conn.execute(
            "SELECT sources FROM papers WHERE paper_id=?", (pid,)
        ).fetchone()
        assert json.loads(row["sources"]) == ["biorxiv"]

    def test_no_usable_identifier_raises(self, store):
        with pytest.raises(ValueError):
            store.upsert(PaperRecord(title="orphan with no ids"))

    def test_arxiv_only_canonical(self, store):
        assert store.upsert(PaperRecord(arxiv_id="2101.1", source="a")) == "arxiv:2101.1"

    def test_pmid_only_canonical(self, store):
        assert store.upsert(PaperRecord(pmid="123", source="a")) == "pmid:123"

    def test_s2_only_canonical(self, store):
        assert store.upsert(PaperRecord(s2_id="abc", source="a")) == "s2:abc"


# --------------------------------------------------------------------------
# Cross-source dedup: same paper, different identifier classes -> ONE entry
# --------------------------------------------------------------------------

class TestDedupByIdentifierClass:
    def test_same_doi_different_case_and_prefix_dedupes(self, store):
        a = store.upsert(PaperRecord(doi="10.1/ABC", title="A", source="s1"))
        b = store.upsert(PaperRecord(doi="https://doi.org/10.1/abc", title="A", source="s2"))
        assert a == b
        assert store.stats()["papers"] == 1

    def test_doi_then_pmid_referring_to_same_work(self, store):
        a = store.upsert(PaperRecord(doi="10.1/x", pmid="999", title="A", source="crossref"))
        b = store.upsert(PaperRecord(pmid="999", title="A", source="pubmed"))
        assert a == b
        assert store.stats()["papers"] == 1

    def test_doi_then_arxiv_referring_to_same_work(self, store):
        a = store.upsert(PaperRecord(doi="10.1/x", arxiv_id="2101.1", title="A", source="s1"))
        b = store.upsert(PaperRecord(arxiv_id="arXiv:2101.1v2", title="A", source="arxiv"))
        assert a == b
        assert store.stats()["papers"] == 1

    def test_doi_then_s2_referring_to_same_work(self, store):
        a = store.upsert(PaperRecord(doi="10.1/x", s2_id="S2ABC", title="A", source="s1"))
        b = store.upsert(PaperRecord(s2_id="S2ABC", title="A", source="semanticscholar"))
        assert a == b
        assert store.stats()["papers"] == 1

    def test_transitive_chain_doi_arxiv_pmid(self, store):
        # Each new record overlaps the running entry on exactly one id and
        # contributes a fresh one; all must collapse to a single paper.
        a = store.upsert(PaperRecord(doi="10.1/x", arxiv_id="2101.1", source="s1"))
        b = store.upsert(PaperRecord(arxiv_id="2101.1", pmid="999", source="s2"))
        c = store.upsert(PaperRecord(pmid="999", s2_id="S2", source="s3"))
        d = store.upsert(PaperRecord(s2_id="S2", source="s4"))
        assert a == b == c == d
        assert store.stats()["papers"] == 1

    def test_genuinely_different_papers_not_merged(self, store):
        store.upsert(PaperRecord(doi="10.1/aaa", source="s1"))
        store.upsert(PaperRecord(doi="10.1/bbb", source="s1"))
        assert store.stats()["papers"] == 2


# --------------------------------------------------------------------------
# upsert merge semantics: fill missing fields, accumulate sources
# --------------------------------------------------------------------------

class TestUpsertMerge:
    def test_merge_fills_missing_alternate_ids(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", title="A", source="s1"))
        store.upsert(PaperRecord(doi="10.1/x", pmid="999", arxiv_id="2101.1", source="s2"))
        row = store._conn.execute(
            "SELECT pmid, arxiv_id FROM papers WHERE paper_id=?", (pid,)
        ).fetchone()
        assert row["pmid"] == "999"
        assert row["arxiv_id"] == "2101.1"

    def test_merge_accumulates_sources_sorted_unique(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", source="biorxiv"))
        store.upsert(PaperRecord(doi="10.1/x", source="arxiv"))
        store.upsert(PaperRecord(doi="10.1/x", source="biorxiv"))  # duplicate source
        row = store._conn.execute(
            "SELECT sources FROM papers WHERE paper_id=?", (pid,)
        ).fetchone()
        assert json.loads(row["sources"]) == ["arxiv", "biorxiv"]

    def test_merge_does_not_overwrite_existing_metadata(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", title="Original", abstract="orig abs",
                                       year=2020, journal="Nature", source="s1"))
        store.upsert(PaperRecord(doi="10.1/x", title="Different", abstract="other",
                                 year=2099, journal="Cell", source="s2"))
        row = store._conn.execute(
            "SELECT title, abstract, year, journal FROM papers WHERE paper_id=?", (pid,)
        ).fetchone()
        assert row["title"] == "Original"
        assert row["abstract"] == "orig abs"
        assert row["year"] == 2020
        assert row["journal"] == "Nature"

    def test_merge_fills_metadata_when_originally_empty(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", title="", abstract="", source="s1"))
        store.upsert(PaperRecord(doi="10.1/x", title="Now Titled", abstract="now abs",
                                 year=2021, journal="Science", source="s2"))
        row = store._conn.execute(
            "SELECT title, abstract, year, journal FROM papers WHERE paper_id=?", (pid,)
        ).fetchone()
        assert row["title"] == "Now Titled"
        assert row["abstract"] == "now abs"
        assert row["year"] == 2021
        assert row["journal"] == "Science"


# --------------------------------------------------------------------------
# Project linkage: idempotency + honest "seen" accounting
# --------------------------------------------------------------------------

class TestProjectLinkage:
    def test_add_returns_true_for_new_link(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", source="s1"))
        assert store.add_to_project("proj", pid) is True

    def test_add_returns_false_when_already_linked(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", source="s1"))
        store.add_to_project("proj", pid)
        assert store.add_to_project("proj", pid) is False

    def test_idempotent_no_duplicate_links(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", source="s1"))
        store.add_to_project("proj", pid)
        store.add_to_project("proj", pid)
        assert store.stats()["project_links"] == 1

    def test_seen_in_project_reflects_state(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", source="s1"))
        assert store.seen_in_project("proj", pid) is False
        store.add_to_project("proj", pid)
        assert store.seen_in_project("proj", pid) is True

    def test_same_paper_two_projects_independent_links(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", source="s1"))
        assert store.add_to_project("projA", pid) is True
        assert store.add_to_project("projB", pid) is True
        assert store.stats()["project_links"] == 2
        # but still a single library entry
        assert store.stats()["papers"] == 1

    def test_seen_index_via_dedup_across_sources(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", pmid="999", source="crossref"))
        assert store.add_to_project("proj", pid) is True
        pid2 = store.upsert(PaperRecord(pmid="999", source="pubmed"))
        assert pid2 == pid
        assert store.seen_in_project("proj", pid2) is True
        assert store.add_to_project("proj", pid2) is False


# --------------------------------------------------------------------------
# Querying project papers + triage
# --------------------------------------------------------------------------

class TestProjectQuery:
    def test_project_papers_returns_added_paper_with_metadata(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", title="Title", abstract="abs", source="s1"))
        store.add_to_project("proj", pid)
        rows = store.project_papers("proj")
        assert len(rows) == 1
        assert rows[0]["paper_id"] == pid
        assert rows[0]["title"] == "Title"
        assert rows[0]["proj_status"] == "screened"

    def test_project_papers_empty_for_unknown_project(self, store):
        store.upsert(PaperRecord(doi="10.1/x", source="s1"))
        assert store.project_papers("nope") == []

    def test_project_papers_isolated_per_project(self, store):
        a = store.upsert(PaperRecord(doi="10.1/a", source="s1"))
        b = store.upsert(PaperRecord(doi="10.1/b", source="s1"))
        store.add_to_project("projA", a)
        store.add_to_project("projB", b)
        rows = store.project_papers("projA")
        assert [r["paper_id"] for r in rows] == [a]

    def test_set_triage_then_query_by_status(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", source="s1"))
        store.add_to_project("proj", pid)
        store.set_triage("proj", pid, "RELEVANT", score=0.82, model="deepseek-chat",
                         status="kept")
        kept = store.project_papers("proj", status="kept")
        assert len(kept) == 1
        assert kept[0]["triage_label"] == "RELEVANT"
        assert kept[0]["triage_score"] == pytest.approx(0.82)
        # status filter excludes non-matching rows
        assert store.project_papers("proj", status="dropped") == []

    def test_set_triage_without_status_preserves_status(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", source="s1"))
        store.add_to_project("proj", pid)
        store.set_triage("proj", pid, "BORDERLINE", score=0.5, model="m")
        rows = store.project_papers("proj")
        assert rows[0]["proj_status"] == "screened"  # COALESCE keeps prior status
        assert rows[0]["triage_label"] == "BORDERLINE"


# --------------------------------------------------------------------------
# PDF pointer
# --------------------------------------------------------------------------

class TestSetPdf:
    def test_set_pdf_updates_path_and_status(self, store):
        pid = store.upsert(PaperRecord(doi="10.1/x", source="s1"))
        store.set_pdf(pid, "library/doi_x.pdf", status="fetched")
        row = store._conn.execute(
            "SELECT pdf_path, pdf_status FROM papers WHERE paper_id=?", (pid,)
        ).fetchone()
        assert row["pdf_path"] == "library/doi_x.pdf"
        assert row["pdf_status"] == "fetched"


# --------------------------------------------------------------------------
# Persistence across reopen (content-addressed ids are stable on disk)
# --------------------------------------------------------------------------

class TestPersistence:
    def test_reopen_dedupes_against_persisted_papers(self, tmp_path):
        db = str(tmp_path / "test.db")
        lib = str(tmp_path / "lib")
        s1 = PaperStore(db, library_dir=lib)
        pid = s1.upsert(PaperRecord(doi="10.1/x", pmid="999", source="s1"))
        s1.add_to_project("proj", pid)
        s1.close()

        s2 = PaperStore(db, library_dir=lib)
        # Same work re-surfacing by PMID after reopen must dedupe, not duplicate.
        pid2 = s2.upsert(PaperRecord(pmid="999", source="s2"))
        assert pid2 == pid
        assert s2.stats()["papers"] == 1
        assert s2.seen_in_project("proj", pid2) is True
        s2.close()


class TestMergeCollisionReconcile:
    """Bridging-record reconciliation in PaperStore._find_all_existing / _reconcile.

    Setup: a DOI-only paper and a PMID-only paper are stored separately (the
    store legitimately cannot know they are the same work — no shared id yet).
    A third record then arrives carrying both that DOI and that PMID, proving
    the two prior entries are one paper.

    Expected behavior: the two entries reconcile into one library entry that
    carries both ids, the unioned sources, and the surviving project links — with
    no duplicate row left behind to corrupt the identifier index.
    """

    def test_bridging_record_reconciles_to_one_paper(self, store):
        a = store.upsert(PaperRecord(doi="10.1/x", title="X", source="a"))
        b = store.upsert(PaperRecord(pmid="999", title="X", source="b"))
        assert a != b
        assert store.stats()["papers"] == 2

        store.upsert(PaperRecord(doi="10.1/x", pmid="999", title="X", source="c"))

        # Reconciled: exactly one paper, carrying both ids and the unioned sources.
        assert store.stats()["papers"] == 1
        rows = store._conn.execute(
            "SELECT paper_id, doi, pmid, sources FROM papers WHERE pmid='999'"
        ).fetchall()
        assert len(rows) == 1                       # no duplicate pmid index
        survivor = rows[0]
        assert survivor["doi"] == "10.1/x" and survivor["pmid"] == "999"
        assert survivor["paper_id"] == "doi:10.1/x"  # strongest id wins
        assert set(json.loads(survivor["sources"])) == {"a", "b", "c"}

    def test_reconcile_preserves_project_links(self, store):
        # Each pre-existing entry was acquired into a different project; after a
        # bridging record collapses them, BOTH project links must survive on the
        # single survivor (no lost or duplicated project_papers rows).
        a = store.upsert(PaperRecord(doi="10.1/x", title="X", source="a"))
        b = store.upsert(PaperRecord(pmid="999", title="X", source="b"))
        store.add_to_project("proj_a", a)
        store.add_to_project("proj_b", b)

        store.upsert(PaperRecord(doi="10.1/x", pmid="999", title="X", source="c"))

        assert store.stats()["papers"] == 1
        survivor = store._conn.execute(
            "SELECT paper_id FROM papers").fetchone()["paper_id"]
        links = store._conn.execute(
            "SELECT project FROM project_papers WHERE paper_id=? ORDER BY project",
            (survivor,)).fetchall()
        assert [r["project"] for r in links] == ["proj_a", "proj_b"]
        # the loser's link was re-pointed, not orphaned or left dangling
        assert store.stats()["project_links"] == 2

    def test_reconcile_survivor_link_wins_on_collision(self, store):
        # If BOTH entries were in the SAME project, the survivor's own link is
        # kept (UPDATE OR IGNORE) and the collision doesn't error or duplicate.
        a = store.upsert(PaperRecord(doi="10.1/x", title="X", source="a"))
        b = store.upsert(PaperRecord(pmid="999", title="X", source="b"))
        store.add_to_project("shared", a)
        store.add_to_project("shared", b)
        assert store.stats()["project_links"] == 2  # distinct paper_ids, same project

        store.upsert(PaperRecord(doi="10.1/x", pmid="999", title="X", source="c"))

        assert store.stats()["papers"] == 1
        assert store.stats()["project_links"] == 1  # collapsed to the survivor's link
