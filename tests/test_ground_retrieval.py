"""Tests for find-then-verify grounding (litstream/eval/ground_retrieval.py). No model download,
no network: the OverlapVerifier and re-grounding are tested with an injected retriever; the
LangChain index plumbing is tested with a DeterministicFakeEmbedding.
"""

from __future__ import annotations

import pytest

from litstream_evidence.evidence_schema import empty_record
from litstream_evidence.ground_retrieval import (
    MiniCheckVerifier, OverlapVerifier, build_retriever, make_embeddings, make_verifier,
    reground_record,
)

SOURCE = ("Methods. CD4 T cells were 25 to 40 percent of CD4 T cells in healthy donors. "
          "FOXP3 marked regulatory T cells. We used CD19 for B cell gating.")


# ---- OverlapVerifier: words + the number teeth ---------------------------------

def test_overlap_supported_when_words_and_numbers_present():
    v = OverlapVerifier()
    ok, score = v.verify("CD4 T cells 25-40 % healthy",
                         "CD4 T cells were 25 to 40 percent in healthy donors")
    assert ok is True and score >= 0.5


def test_overlap_rejects_invented_number_even_if_words_match():
    v = OverlapVerifier()
    # words overlap, but the claimed 99 is NOT in the passage -> flagged
    ok, _ = v.verify("NK cells 99 % of PBMC",
                     "NK cells were present in PBMC samples from donors")
    assert ok is False


def test_overlap_rejects_low_word_overlap():
    v = OverlapVerifier(min_overlap=0.5)
    ok, _ = v.verify("regulatory T cell FOXP3 signature",
                     "the assay measured background fluorescence levels")
    assert ok is False


# ---- re-grounding a record (injected retriever -> deterministic) ----------------

def _make_record():
    rec = empty_record("p1")
    rec["relevance"] = "HIGH"
    rec["genes"] = [{"symbol": "FOXP3", "species": "", "source_quote": "made up"}]
    rec["frequencies"] = [
        {"cell_type": "CD4 T cells", "value": "25-40", "unit": "%",
         "of_population": "CD4 T cells", "condition": "healthy", "source_quote": "fabricated quote"},
        {"cell_type": "NK cells", "value": "99", "unit": "%", "of_population": "PBMC",
         "condition": "", "source_quote": "NK cells were 99%"},     # invented number
    ]
    return rec


class _Always:
    def __init__(self, result):
        self.result = result

    def verify(self, claim, passage):
        return (self.result, 1.0 if self.result else 0.0)


def test_reground_routes_value_fields_to_value_verifier():
    rec = empty_record("p")
    rec["genes"] = [{"symbol": "FOXP3", "species": "", "source_quote": "x"}]          # entity field
    rec["frequencies"] = [{"cell_type": "CD4 T cells", "value": "25", "unit": "%",
                           "source_quote": "x"}]                                       # value field
    # entity verifier says NO, value verifier says YES -> only the frequency should ground
    _, rep = reground_record(rec, lambda q: ["CD4 T cells 25 % FOXP3"],
                             verifier=_Always(False), value_verifier=_Always(True))
    assert rep["by_field"]["genes"]["grounded"] == 0          # entity used the (NO) entity verifier
    assert rep["by_field"]["frequencies"]["grounded"] == 1    # value used the (YES) value verifier
    # and the reverse routing
    _, rep2 = reground_record(rec, lambda q: ["..."], verifier=_Always(True), value_verifier=_Always(False))
    assert rep2["by_field"]["genes"]["grounded"] == 1
    assert rep2["by_field"]["frequencies"]["flagged"] == 1


def test_reground_routes_categorical_thresholds_to_entity_verifier():
    # within ONE value field, routing is per item: a numeric threshold -> value verifier, a
    # categorical 'CD3 positive' -> the entity/presence verifier (it carries no number to check).
    rec = empty_record("p")
    rec["gating_thresholds"] = [
        {"marker": "CD3", "operator": "", "value": "positive", "source_quote": "x"},   # categorical
        {"marker": "CD25", "operator": ">", "value": "500", "source_quote": "x"},       # numeric
    ]
    # entity verifier says YES, value verifier says NO
    _, rep = reground_record(rec, lambda q: ["CD3 positive, CD25 above 500 selected"],
                             verifier=_Always(True), value_verifier=_Always(False))
    gt = rep["by_field"]["gating_thresholds"]
    assert gt["value_checked"] == 1                 # only the numeric 'CD25 > 500' hit the value verifier
    assert gt["grounded"] == 1                      # categorical 'CD3 positive' grounded by presence (YES)
    assert gt["flagged"] == 1                       # numeric 'CD25 > 500' flagged by value verifier (NO)
    assert any(f["item"]["value"] == "500" for f in rep["flagged_items"])


def test_reground_grounds_real_items_and_flags_invented_ones():
    rec, report = reground_record(_make_record(), lambda q: [SOURCE], OverlapVerifier())
    assert report["by_field"]["genes"]["grounded"] == 1
    assert report["by_field"]["frequencies"]["grounded"] == 1      # the 25-40% one (in source)
    assert report["by_field"]["frequencies"]["flagged"] == 1       # the invented 99%
    assert any(f["item"]["cell_type"] == "NK cells" for f in report["flagged_items"])
    # a grounded item's source_quote was replaced with the REAL passage, not the fabrication
    cd4 = rec["frequencies"][0]
    assert "25" in cd4["source_quote"] and "fabricated" not in cd4["source_quote"]


def test_build_retriever_returns_passages_with_fake_embeddings():
    emb = make_embeddings("fake")
    retrieve = build_retriever(SOURCE * 5, emb, k=3, size=200, overlap=40)
    passages = retrieve("CD4 T cells frequency")
    assert 1 <= len(passages) <= 3
    assert all(isinstance(p, str) and p for p in passages)


def test_make_factories():
    assert isinstance(make_verifier("overlap"), OverlapVerifier)
    with pytest.raises(ValueError):
        make_verifier("nope")
    with pytest.raises(ValueError):
        make_embeddings("nope")


def test_minicheck_requires_predict_callable():
    with pytest.raises(NotImplementedError):
        MiniCheckVerifier().verify("claim", "passage")
    v = MiniCheckVerifier(predict=lambda c, p: "FOXP3" in p)
    assert v.verify("FOXP3 is a Treg marker", "the paper discusses FOXP3")[0] is True
    assert v.verify("FOXP3 is a Treg marker", "unrelated passage")[0] is False


def test_make_verifier_minicheck_is_lazy():
    v = make_verifier("minicheck")
    assert isinstance(v, MiniCheckVerifier)
    assert callable(v.predict)


def test_minicheck_number_teeth_override():
    # MiniCheck says "supported" for everything, but an invented quantity is still rejected
    always_yes = MiniCheckVerifier(predict=lambda c, p: True)
    assert always_yes.verify("NK cells 99 % of PBMC", "NK cells seen in PBMC")[0] is False  # 99 absent
    assert always_yes.verify("FOXP3 marks Tregs", "FOXP3 marks regulatory T cells")[0] is True  # no number


def test_reground_routes_claim_fields_to_strict_verifier_without_numbers():
    from litstream_evidence.ground_retrieval import _needs_strict
    # a proposition with NO number must still go to the strict (entailment) verifier, not presence
    assert _needs_strict("findings", "the method assumes negative binomial counts") is True
    rec = empty_record("p")
    rec["findings"] = [{"statement": "the method assumes negative binomial counts", "source_quote": "x"}]
    # entity verifier says YES, strict says NO -> claim is flagged (it used the strict verifier)
    _, rep = reground_record(rec, lambda q: ["..."], verifier=_Always(True), value_verifier=_Always(False))
    assert rep["by_field"]["findings"]["flagged"] == 1
    assert rep["by_field"]["findings"]["value_checked"] == 1     # routed to strict despite no number
    # reverse: strict says YES -> grounded
    rec2 = empty_record("p")
    rec2["study_aim"] = [{"statement": "to test whether X drives Y", "source_quote": "x"}]
    _, rep2 = reground_record(rec2, lambda q: ["..."], verifier=_Always(False), value_verifier=_Always(True))
    assert rep2["by_field"]["study_aim"]["grounded"] == 1


def test_new_wetlab_fields_route_by_kind():
    from litstream_evidence.evidence_schema import FIELD_KIND, NUMERIC_FIELDS
    from litstream_evidence.ground_retrieval import _is_numeric_claim
    # diseases + perturbations are entity mentions; cohort is a numeric (sample-size) claim
    assert FIELD_KIND["diseases"] == "entity" and FIELD_KIND["perturbations"] == "entity"
    assert "cohort" in NUMERIC_FIELDS and "perturbations" not in NUMERIC_FIELDS
    assert _is_numeric_claim("cohort", "healthy donors 120") is True          # has a count -> MiniCheck
    assert _is_numeric_claim("perturbations", "anti-CD3/CD28 stimulation") is False   # mention -> presence


def test_numbers_ignores_digits_inside_names():
    from litstream_evidence.ground_retrieval import _numbers
    assert _numbers("CD25 high, FOXP3+") == []        # digits inside names are not quantities
    assert _numbers("25-40% of CD4 T cells") == ["25", "40"]
