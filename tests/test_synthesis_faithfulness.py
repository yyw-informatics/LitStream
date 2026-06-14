"""Tests for the deterministic synthesis-faithfulness scorer
(litstream/eval/synthesis_faithfulness.py). Fully offline: a tiny synthesis plus tiny
`*_evidence.md` files written into tmp_path, no model, no network, no MiniCheck.

Three cases are asserted explicitly:
  1. a fabricated number is caught (fails number_grounded),
  2. a grounded claim passes all three checks, and
  3. the lexical blind spot, a valid paraphrase the rule-based check misses, so the
     documented limitation is pinned by a test.
"""

from __future__ import annotations

from litstream.eval.synthesis_faithfulness import (
    DEFAULT_OVERLAP_THRESHOLD,
    entity_grounded,
    lexical_overlap,
    number_grounded,
    run,
    score_synthesis,
    structural_metrics,
)
from litstream.eval.citation_check import parse_appendix_a, split_frontmatter


# A tiny synthesis: one grounded claim, one fabricated-number claim, one paraphrase, an uncited
# claim, and an Appendix-A table resolving each citation to an evidence file.
SYNTHESIS = """---
project: demo
---

# Synthesis

totalVI ranks #1 of 28 integration methods [Yin 2025].
ADTnorm preserves the negative peak required for gating [Zheng 2025].
Rare cells make up 0.4% of the population [Zheng 2025].
Antibody capture beats sequencing depth when resolving scarce immune populations [Zhao 2026].
This is widely regarded as a translation problem.

## Appendix A: Paper-by-Paper Summary

| # | Citation | Evidence files |
|---|----------|----------------|
| 1 | [Yin 2025] | `yin_evidence.md` |
| 2 | [Zheng 2025] | `zheng_evidence.md` |
| 3 | [Zhao 2026] | `zhao_evidence.md` |
"""

# Evidence wording is chosen to drive each case. zhao paraphrases the synthesis claim
# ("surface protein outperforms RNA") so the lexical check misses a valid attribution.
EVIDENCE = {
    "yin_evidence.md": "We benchmarked 28 integration methods; totalVI ranks #1 overall.",
    "zheng_evidence.md": "ADTnorm preserves the negative peak required for gating. "
                         "Rare cells account for 5.2% of the population in this cohort.",
    "zhao_evidence.md": "Surface protein outperforms RNA transcripts for detecting rare cell types.",
}


def _write_project(tmp_path, project="demo"):
    lit = tmp_path / f"projects/{project}/literature"
    lit.mkdir(parents=True)
    (lit / "0_synthesis_literature.md").write_text(SYNTHESIS)
    for name, text in EVIDENCE.items():
        (lit / name).write_text(text)
    return lit


# ---- the three deterministic checks in isolation -------------------------------

def test_number_grounded_passes_when_quantity_present():
    assert number_grounded("totalVI ranks #1 of 28 methods",
                           "We benchmarked 28 integration methods.") is True


def test_number_grounded_catches_fabricated_number():
    # claim says 0.4%, evidence says 5.2%, so number_grounded must fail
    assert number_grounded("Rare cells make up 0.4% of the population",
                           "Rare cells account for 5.2% of the population.") is False


def test_number_grounded_is_unit_aware():
    assert number_grounded("5% of cells", "we saw 5 cells") is False
    assert number_grounded("5% of cells", "we saw 5% of cells") is True


def test_number_grounded_percent_not_substring_matched():
    assert number_grounded("up 5%", "rose 15% overall") is False
    assert number_grounded("5% of cells", "0.5% of cells were rare") is False
    assert number_grounded("5% of cells", "exactly 5% of cells") is True


def test_number_grounded_trivial_without_numbers():
    assert number_grounded("protein outperforms RNA", "anything at all") is True


def test_entity_grounded_requires_each_entity():
    assert entity_grounded("FOXP3 marks Tregs", "FOXP3 is expressed in regulatory cells") is True
    assert entity_grounded("CD8 and FOXP3 co-express", "only CD8 is mentioned here") is False


def test_entity_grounded_accepts_marker_gene_aliases():
    assert entity_grounded("PTPRC marks naive T cells", "CD45RA is a naive T marker") is True
    assert entity_grounded("CD8A defines cytotoxic T cells", "CD8 T cells were gated") is True
    assert entity_grounded("CD20 EXPLORATORY", "B cells express CD20") is True


def test_lexical_overlap_is_a_pure_string_metric():
    high = lexical_overlap("ADTnorm preserves the negative peak required for gating",
                           "ADTnorm preserves the negative peak required for gating.")
    low = lexical_overlap("ADTnorm preserves the negative peak required for gating",
                          "Completely unrelated sentence about something else entirely.")
    assert high > 0.9
    assert low < high


# ---- end-to-end on the tiny synthesis ------------------------------------------

def _scores_by_prefix(scores):
    return {s.claim[:18]: s for s in scores}


def test_grounded_claim_passes_all_three():
    scores, _ = score_synthesis(SYNTHESIS, EVIDENCE)
    byp = _scores_by_prefix(scores)
    adtnorm = byp["ADTnorm preserves "]
    assert adtnorm.number_grounded and adtnorm.entity_grounded
    assert adtnorm.lexical_overlap >= DEFAULT_OVERLAP_THRESHOLD
    assert adtnorm.passed is True
    assert adtnorm.reason == ""


def test_fabricated_number_is_flagged():
    scores, m = score_synthesis(SYNTHESIS, EVIDENCE)
    byp = _scores_by_prefix(scores)
    rare = byp["Rare cells make up"]
    assert rare.number_grounded is False
    assert rare.passed is False
    assert "number not grounded" in rare.reason
    # and it surfaces in the aggregate flagged list
    flagged = {fl["claim"] for fl in m["faithfulness"]["flagged"]}
    assert any(c.startswith("Rare cells make up") for c in flagged)


def test_lexical_blind_spot_valid_paraphrase_is_missed():
    """Documented limitation: a valid attribution worded as a paraphrase is flagged unsupported.

    The Zhao claim ("Antibody capture beats sequencing depth when resolving scarce immune
    populations") is supported by the evidence ("Surface protein outperforms RNA transcripts for
    detecting rare cell types"), but the words barely overlap, so the rule-based check fails it on
    lexical_overlap. This false negative is inherent to the deterministic floor; the model-based
    citation check exists to catch it.
    """
    scores, _ = score_synthesis(SYNTHESIS, EVIDENCE)
    byp = _scores_by_prefix(scores)
    zhao = byp["Antibody capture b"]
    assert zhao.lexical_overlap < DEFAULT_OVERLAP_THRESHOLD
    assert zhao.passed is False                      # rule-based check flags a valid claim
    assert "lexical_overlap" in zhao.reason


def test_faithfulness_aggregate_rates():
    scores, m = score_synthesis(SYNTHESIS, EVIDENCE)
    f = m["faithfulness"]
    # four cited prose claims resolve to present evidence files (Yin, Zheng x2, Zhao)
    assert f["n_scored_claims"] == 4
    # only the fabricated-number claim fails number_grounded -> 3/4
    assert f["number_grounded_rate"] == 0.75
    # the fabricated-number and paraphrase claims fail, the others pass -> partial faithfulness
    assert f["faithfulness_rate"] is not None and 0.0 < f["faithfulness_rate"] < 1.0


# ---- structural integrity ------------------------------------------------------

def test_structural_uncited_and_coverage():
    _, body = split_frontmatter(SYNTHESIS)
    mapping = parse_appendix_a(SYNTHESIS)
    st = structural_metrics(body, mapping, EVIDENCE)
    # the "translation problem" clause carries no citation -> one uncited claim
    assert st["n_uncited_claims"] == 1
    assert st["uncited_claim_rate"] is not None and st["uncited_claim_rate"] > 0
    # every evidence file is cited -> full coverage, none dangling
    assert st["coverage"] == 1.0
    assert st["n_dangling"] == 0
    assert st["evidence_uncited"] == []


def test_dangling_citation_detected():
    synth = ("A claim resolving nowhere [Ghost 2099].\n\n"
             "## Appendix A\n\n| # | Citation | Evidence files |\n|---|---|---|\n")
    _, body = split_frontmatter(synth)
    st = structural_metrics(body, parse_appendix_a(synth), {})
    assert st["n_dangling"] == 1
    assert st["dangling_rate"] == 1.0


def test_count_consistency_detects_mismatch():
    # "3 markers" claims three but lists two co-located items -> inconsistent
    synth = "We highlight 3 markers CD4, CD8 [Zheng 2025].\n"
    _, body = split_frontmatter(synth)
    st = structural_metrics(body, {"zheng": ["zheng_evidence.md"]},
                            {"zheng_evidence.md": "CD4 and CD8 are markers."})
    assert st["count_checks"]
    assert st["count_consistent"] is False


# ---- determinism + runner ------------------------------------------------------

def test_rerun_is_identical():
    a_scores, a_m = score_synthesis(SYNTHESIS, EVIDENCE)
    b_scores, b_m = score_synthesis(SYNTHESIS, EVIDENCE)
    assert [(s.claim, s.passed, s.lexical_overlap) for s in a_scores] == \
           [(s.claim, s.passed, s.lexical_overlap) for s in b_scores]
    assert a_m == b_m


def test_run_reads_project_from_disk(tmp_path):
    _write_project(tmp_path)
    scores, m = run("demo", tmp_path)
    assert m["faithfulness"]["n_scored_claims"] == 4
    assert any(not s.passed for s in scores)


def test_run_no_synthesis_is_empty(tmp_path):
    (tmp_path / "projects/demo/literature").mkdir(parents=True)
    scores, m = run("demo", tmp_path)
    assert scores == [] and m == {}
