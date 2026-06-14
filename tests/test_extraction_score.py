"""Tests for the A1 slot-level extraction scorer (litstream/eval/extraction_score.py).

Fully OFFLINE: a tiny `<stem>_evidence.json` sidecar + a tiny gold key are written into
tmp_path, no network / no model / no PDF. The canonical case is one true positive, one miss,
and one hallucination → P = R = F1 = 0.5, asserted exactly.
"""

from __future__ import annotations

import json

from litstream.eval.extraction_score import (
    aggregate, gene_aliases, load_key, load_sidecars, run, score, species_of,
)  # noqa: F401  (load_* / score exercised in the loader + determinism tests)


def _make_project(tmp_path, record: dict, project: str = "p"):
    """Write one evidence sidecar where the structurer would, and return the project dir."""
    lit = tmp_path / "projects" / project / "literature"
    lit.mkdir(parents=True)
    (lit / "paper1_evidence.json").write_text(json.dumps(record))
    return tmp_path


def _key(tmp_path, obj: dict):
    p = tmp_path / "key.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    return p


# canonical sidecar: FOXP3 is a true positive, MYC is a hallucination (not in gold).
SIDECAR = {
    "paper_id": "paper1", "relevance": "HIGH", "species": ["human"],
    "genes": [
        {"symbol": "FOXP3", "species": "human", "source_quote": "FOXP3+ Tregs"},   # TP
        {"symbol": "MYC", "species": "human", "source_quote": "MYC was high"},      # FP (hallucination)
    ],
}
# gold: FOXP3 (found) + CD25 (missed). MYC is not gold.
GOLD = {"paper_id": "paper1", "genes": ["FOXP3", "CD25"]}


def test_one_tp_one_miss_one_hallucination(tmp_path):
    pdir = _make_project(tmp_path, SIDECAR)
    rows, agg, n_papers, n_mined = run("p", pdir, _key(tmp_path, GOLD))
    assert (n_papers, n_mined) == (1, 1)

    # strict tier, genes field: TP=1 (FOXP3), FP=1 (MYC hallucination), FN=1 (CD25 miss)
    b = agg["strict"]["by_field"]["genes"]
    assert b == {"tp": 1, "fp": 1, "fn": 1}, b
    p, r, f = agg["strict"]["field_prf"]["genes"]
    assert (p, r, f) == (0.5, 0.5, 0.5)

    # overall micro mirrors the single field; macro too (one field only)
    assert agg["strict"]["micro"] == (0.5, 0.5, 0.5)
    assert agg["strict"]["macro"] == (0.5, 0.5, 0.5)


def test_normalized_tier_recovers_a_species_aware_gene_alias(tmp_path):
    rec = {"paper_id": "paper1", "relevance": "HIGH", "species": ["human"],
           "genes": [{"symbol": "IL2RA", "species": "human", "source_quote": "IL2RA"}]}
    pdir = _make_project(tmp_path, rec)
    _, agg, _, _ = run("p", pdir, _key(tmp_path, {"paper_id": "paper1", "genes": ["CD25"]}))
    assert agg["strict"]["by_field"]["genes"] == {"tp": 0, "fp": 1, "fn": 1}
    assert agg["normalized"]["by_field"]["genes"] == {"tp": 1, "fp": 0, "fn": 0}
    assert agg["normalized"]["micro"][2] > agg["strict"]["micro"][2]


def test_labeled_but_unmined_paper_is_all_misses(tmp_path):
    # gold labels a paper that has no sidecar → every gold item is a recall miss, no FPs.
    pdir = _make_project(tmp_path, SIDECAR)   # only paper1 is mined
    rows, agg, n_papers, n_mined = run(
        "p", pdir, _key(tmp_path, {"paper_id": "ghost", "genes": ["TP53", "EGFR"]}))
    assert (n_papers, n_mined) == (1, 0)
    b = agg["strict"]["by_field"]["genes"]
    assert b == {"tp": 0, "fp": 0, "fn": 2}, b


def test_only_labeled_fields_are_scored(tmp_path):
    # the sidecar has cell_types, but the gold key doesn't label that field → unmeasured,
    # not a recall miss. Only the labeled `genes` field produces rows.
    rec = dict(SIDECAR, cell_types=[{"name": "regulatory T cell", "source_quote": "Tregs"}])
    pdir = _make_project(tmp_path, rec)
    rows, agg, _, _ = run("p", pdir, _key(tmp_path, GOLD))
    assert set(agg["strict"]["by_field"]) == {"genes"}
    assert "cell_types" not in agg["strict"]["by_field"]


def test_species_routing_and_alias_seam():
    assert species_of({"species": ["human", "PBMC"]}) == "human"
    assert species_of({"species": ["Mus musculus"]}) == "mouse"
    assert species_of({"species": []}) == ""
    # casefold + alias table: a symbol and its surface alias share one concept set
    assert "cd25" in gene_aliases("IL2RA", "human")
    assert gene_aliases("Foxp3", "mouse") == {"foxp3", "scurfin"}


def test_loaders_round_trip(tmp_path):
    pdir = _make_project(tmp_path, SIDECAR)
    recs = load_sidecars("p", pdir)
    assert set(recs) == {"paper1"} and recs["paper1"]["relevance"] == "HIGH"
    gold = load_key(_key(tmp_path, GOLD))
    assert gold == {"paper1": {"genes": ["FOXP3", "CD25"]}}


def test_score_is_deterministic(tmp_path):
    pdir = _make_project(tmp_path, SIDECAR)
    recs, gold = load_sidecars("p", pdir), load_key(_key(tmp_path, GOLD))
    assert score(recs, gold, "strict") == score(recs, gold, "strict")
    assert aggregate(score(recs, gold, "strict")) == aggregate(score(recs, gold, "strict"))
