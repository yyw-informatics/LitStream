"""Tests for the A2 citation/attribution checker (litstream/eval/citation_check.py)."""

from __future__ import annotations

from litstream.eval.citation_check import (
    EXTERNAL_REVIEW, SUPPORTIVE, PARTIAL, CONTRADICTORY, IRRELEVANT,
    check_synthesis, parse_appendix_a, parse_claims, _norm_key,
)

SYNTHESIS = """---
project: demo
---

# Synthesis

Surface protein outperforms RNA for rare-cell detection [Zheng 2025].
ADTnorm preserves the negative peak required for gating [Zheng].
totalVI is the #1 integration method [Yin 2025], but it removes negative peaks [Zheng 2025].
Predicted proteins carry partial error and are not perfect [Zhao 2026].
The wider field frames this as a translation problem [External Review].

| Marker | Cell type | Papers |
|--------|-----------|--------|
| CD4 | CD4 T cells | [Zheng 2025] |

## Appendix A: Paper-by-Paper Summary

| # | Citation | Evidence files |
|---|----------|----------------|
| 1 | [Zheng 2025] | `zheng_adtnorm_evidence.md` + `doi_zheng_evidence.md` |
| 2 | [Yin 2025] | `yin_benchmark_evidence.md` |
| 3 | [Zhao 2026] | `zhao_scprotrans_evidence.md` |

## Appendix B: Gene Provenance

| CD4 | should not be parsed as a citation row |
"""

EVIDENCE = {
    "zheng_adtnorm_evidence.md": "ADTnorm preserves the negative peak; protein outperforms RNA. "
                                 "totalVI removes negative peaks and is detrimental for gating.",
    "doi_zheng_evidence.md": "Duplicate mining of ADTnorm: it preserves the negative peak required "
                             "for gating; protein outperforms RNA.",
    "yin_benchmark_evidence.md": "Benchmark of 28 clustering methods; totalVI ranks #1 for integration.",
    "zhao_scprotrans_evidence.md": "scProTrans predicts proteins from RNA with partial error.",
}


class _Keyword:
    """Verifier double with the ground_retrieval `.verify` shape plus a `.label` head."""

    def label(self, claim: str, passage: str) -> str:
        c, p = claim.lower(), passage.lower()
        if "removes negative peaks" in c and "preserves the negative peak" in p:
            return CONTRADICTORY
        if "partial" in c:
            return PARTIAL
        words = [w.strip(".,;#") for w in c.split() if len(w.strip(".,;#")) > 3]
        if not words:
            return IRRELEVANT
        hit = sum(w in p for w in words) / len(words)
        return SUPPORTIVE if hit >= 0.6 else IRRELEVANT

    def verify(self, claim: str, passage: str) -> tuple[bool, float]:
        ok = self.label(claim, passage) == SUPPORTIVE
        return (ok, 1.0 if ok else 0.0)


def test_norm_key_collapses_year_and_abbreviation():
    assert _norm_key("Zheng 2025") == "zheng"
    assert _norm_key("Zheng") == "zheng"
    assert _norm_key("van der Berg 2024") == "van"


def test_appendix_a_maps_citation_to_evidence_files():
    mapping = parse_appendix_a(SYNTHESIS)
    assert mapping["zheng"] == ["zheng_adtnorm_evidence.md", "doi_zheng_evidence.md"]
    assert mapping["yin"] == ["yin_benchmark_evidence.md"]
    assert mapping["zhao"] == ["zhao_scprotrans_evidence.md"]
    assert "cd4" not in mapping


def test_parse_claims_pairs_text_with_citations_and_strips_markers():
    claims = parse_claims(SYNTHESIS)
    by_text = {c.text: c for c in claims}
    multi = next(c for c in claims if c.text.startswith("totalVI is the #1"))
    assert {cit.key for cit in multi.citations} == {"yin", "zheng"}
    assert "[" not in multi.text
    ext = next(c for c in claims if c.citations and c.citations[0].external)
    assert ext.citations[0].raw == EXTERNAL_REVIEW


def test_external_review_is_skipped_never_resolved():
    pairings, _ = check_synthesis(SYNTHESIS, EVIDENCE, _Keyword())
    ext = [p for p in pairings if p.citation == EXTERNAL_REVIEW]
    assert len(ext) == 1
    assert ext[0].evidence_file == ""                 # never resolved through Appendix A
    assert "external review" in ext[0].note.lower()


def test_contradiction_is_labeled_and_fails_its_claim():
    pairings, _ = check_synthesis(SYNTHESIS, EVIDENCE, _Keyword())
    # "totalVI ... removes negative peaks [Zheng 2025]" vs Zheng's "preserves the negative peak"
    contra = [p for p in pairings
              if p.label == CONTRADICTORY and p.citation == "[Zheng 2025]"]
    assert contra, "expected a Contradictory pairing for the totalVI/Zheng claim"


def test_partial_label_flows_through():
    pairings, _ = check_synthesis(SYNTHESIS, EVIDENCE, _Keyword())
    assert any(p.label == PARTIAL for p in pairings)   # the "partial error" Zhao claim


def test_known_case_precision_recall_coverage():
    pairings, m = check_synthesis(SYNTHESIS, EVIDENCE, _Keyword())

    # Each [Zheng] citation fans out across both minings (two evidence files); the "preserves the
    # negative peak" claim is supported by both, so it is the one fully-supported cited claim.
    zheng_supported = [p for p in pairings
                       if p.citation in ("[Zheng 2025]", "[Zheng]") and p.label == SUPPORTIVE]
    assert {p.evidence_file for p in zheng_supported} == {"zheng_adtnorm_evidence.md",
                                                          "doi_zheng_evidence.md"}

    # The body data-table row ("CD4 | CD4 T cells | [Zheng 2025]") is parsed as a 5th cited claim,
    # adding 2 Irrelevant pairs. 2 Supportive of 10 scored pairs -> precision 0.2; 1 of 5 cited
    # claims fully supported -> recall 0.2. The Contradictory totalVI claim must not be recalled.
    assert m["n_scored_pairs"] == 10
    assert m["citation_precision"] == 0.2
    assert m["claim_recall"] == 0.2
    assert m["label_counts"] == {SUPPORTIVE: 2, PARTIAL: 1, CONTRADICTORY: 2, IRRELEVANT: 5}
    totalvi_claim = next(p.claim for p in pairings if p.label == CONTRADICTORY)
    assert all(p.label != SUPPORTIVE for p in pairings if p.claim == totalvi_claim)

    # Coverage: all four evidence files are reachable via Appendix A and each is cited by >=1 claim.
    assert m["evidence_total"] == 4
    assert m["coverage"] == 1.0
    assert m["evidence_uncited"] == []


def test_unresolved_citation_is_flagged_not_scored():
    # a citation absent from Appendix A resolves to nothing: flagged, evidence_file empty, not scored
    synth = "Some claim with no appendix entry [Ghost 2099].\n\n## Appendix A\n\n(no rows)\n"
    pairings, m = check_synthesis(synth, {}, _Keyword())
    ghost = [p for p in pairings if p.citation == "[Ghost 2099]"]
    assert ghost and ghost[0].evidence_file == "" and "unresolved" in ghost[0].note
    assert m["n_scored_pairs"] == 0


def test_provided_mapping_overrides_appendix_parsing():
    synth = "Protein outperforms RNA [Foo 2030].\n"
    mapping = {"foo": ["foo_evidence.md"]}
    pairings, m = check_synthesis(synth, {"foo_evidence.md": "protein outperforms rna here"},
                                  _Keyword(), mapping=mapping)
    assert pairings[0].evidence_file == "foo_evidence.md"
    assert pairings[0].label == SUPPORTIVE
    assert m["citation_precision"] == 1.0
