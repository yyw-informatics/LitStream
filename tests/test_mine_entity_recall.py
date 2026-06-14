"""Tests for the MINE entity-recall harness (litstream/eval/mine_entity_recall.py).

Deterministic: a StubNER with a tiny controlled lexicon over hand-written source text +
evidence markdown, so recall/extra_rate equal hand-computed values. No model, no network,
no PDF — score_paper takes source_text directly.
"""

from __future__ import annotations

import pytest

from litstream.eval.mine_entity_recall import (
    Entity, StubNER, make_backend, normalize, parse_evidence, score_paper,
)

LEXICON = {
    "gene": ["CD4", "FOXP3", "GZMB"],
    "cell_type": ["Treg", "B cell"],
    "species": ["human", "mouse"],
}
LABELS = ["gene", "cell_type", "species"]

SOURCE = ("We profiled CD4 and FOXP3 and GZMB in Treg and B cell populations "
          "in human samples.")

# Evidence: surfaces CD4+FOXP3 (MISSES GZMB and B cell) and claims a `mouse` ortholog
# that is NOT in the source (a deliberate unsupported/hallucinated entity).
EVIDENCE = """---
species: "human"
cell_types: ["Treg"]
relevance: "HIGH"
---
# Evidence
We found CD4 and FOXP3 expression in Treg cells. The authors note a mouse ortholog.
"""


@pytest.fixture
def rows():
    return {r["label"]: r for r in
            score_paper("p1", SOURCE, EVIDENCE, StubNER(LEXICON), LABELS)}


def test_gene_recall_penalizes_a_missed_entity(rows):
    g = rows["gene"]
    assert g["n_silver"] == 3          # CD4, FOXP3, GZMB in source
    assert g["n_mine"] == 2            # CD4, FOXP3 in evidence (GZMB missed)
    assert g["recall"] == 0.667        # 2/3
    assert g["extra_rate"] == 0.0      # nothing unsupported
    assert "gzmb" in g["missed"]


def test_cell_type_recall_half(rows):
    c = rows["cell_type"]
    assert c["n_silver"] == 2          # Treg, B cell in source
    assert c["n_mine"] == 1            # only Treg in evidence
    assert c["recall"] == 0.5
    assert c["extra_rate"] == 0.0
    assert "b cell" in c["missed"]


def test_species_extra_rate_flags_hallucination(rows):
    s = rows["species"]
    assert s["n_silver"] == 1          # human in source
    assert s["n_mine"] == 2            # human + mouse in evidence
    assert s["recall"] == 1.0          # human covered
    assert s["extra_rate"] == 0.5      # mouse is unsupported -> 1/2
    assert "mouse" in s["extra"]


def test_match_column_is_strict_by_default(rows):
    assert all(r["match"] == "strict" for r in rows.values())


class _TwoSidedNER:
    """Returns `silver_ents` for the source text (tagged with a sentinel) and `mine_ents`
    otherwise — lets us force a substring relationship (silver 'cd4 t cell' vs mine 'cd4')
    that strict matching misses but relaxed matching credits."""

    def __init__(self, silver_ents, mine_ents):
        self.silver_ents, self.mine_ents = silver_ents, mine_ents

    def extract(self, text, labels):
        return self.silver_ents if "SOURCE_MARKER" in text else self.mine_ents


def test_relaxed_match_credits_substring_overlap():
    backend = _TwoSidedNER(
        silver_ents=[Entity("CD4 T cell", "cell_type", None, "cd4 t cell")],
        mine_ents=[Entity("CD4", "cell_type", None, "cd4")],
    )
    src = "SOURCE_MARKER paper about lymphocyte subsets"      # no literal 'cd4'
    ev = "---\n---\nThe CD4 population was enriched."
    strict = {r["label"]: r for r in score_paper("p", src, ev, backend, ["cell_type"])}["cell_type"]
    relaxed = {r["label"]: r for r in
               score_paper("p", src, ev, backend, ["cell_type"], relaxed=True)}["cell_type"]
    # strict: 'cd4' != 'cd4 t cell' -> missed silver, and 'cd4' counts as unsupported
    assert strict["recall"] == 0.0 and strict["extra_rate"] == 1.0
    # relaxed: 'cd4' ⊂ 'cd4 t cell' -> silver covered, nothing left over
    assert relaxed["recall"] == 1.0 and relaxed["extra_rate"] == 0.0


def test_empty_sets_give_none_not_zero_division():
    # a label the lexicon never matches -> n_silver=0, n_mine=0 -> recall/extra None
    rows = {r["label"]: r for r in
            score_paper("p", "nothing here", "---\n---\nempty", StubNER(LEXICON), ["gene"])}
    assert rows["gene"]["n_silver"] == 0
    assert rows["gene"]["recall"] is None
    assert rows["gene"]["extra_rate"] is None


def test_parse_evidence_tolerant():
    fm, body = parse_evidence(EVIDENCE)
    assert fm["species"] == "human" and fm["cell_types"] == ["Treg"]
    assert "Evidence" in body
    # no frontmatter -> empty dict, whole text as body
    fm2, body2 = parse_evidence("just prose, no frontmatter")
    assert fm2 == {} and body2 == "just prose, no frontmatter"
    # malformed YAML -> empty dict, not a crash
    fm3, _ = parse_evidence("---\n: : bad : yaml :\n---\nbody")
    assert fm3 == {}


def test_normalize_collapses_and_casefolds():
    assert normalize("  CD8  T cell. ") == "cd8 t cell"
    assert normalize("FOXP3") == "foxp3"


def test_make_backend():
    assert isinstance(make_backend("stub"), StubNER)
    with pytest.raises(ValueError):
        make_backend("nope")
