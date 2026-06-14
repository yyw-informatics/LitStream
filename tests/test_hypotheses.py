"""Tests for the hypothesis-candidate generator (litstream/hypotheses). Offline: no model, no network.
Grounding uses the stub or the offline lexical verifier; figures degrade without matplotlib."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from litstream.hypotheses.config import HypothesisConfig
from litstream.hypotheses.frame_extractor import extract_frames
from litstream.hypotheses.graph_builder import build_evidence_graph
from litstream.hypotheses.grounding import (
    FrameGrounder, LexicalFrameVerifier, StubGrounder, make_grounder, verify_frames,
)
from litstream.hypotheses.normalize import Normalizer, contexts_compatible
from litstream.hypotheses.pipeline import ContextBoundHypothesisGenerator, run_to_dir
from litstream.hypotheses.schema import BioContext, Entity, HypothesisCandidate
from litstream.hypotheses.templates import generate_candidates
from litstream.hypotheses.filters import filter_candidates
from litstream.hypotheses.ranker import rank_candidates

NORM = Normalizer()
STUB = StubGrounder()


# --- helpers ---------------------------------------------------------------------

def rec(paper_id, findings, **kw):
    base = dict(paper_id=paper_id, relevance=kw.pop("relevance", "HIGH"),
                species=kw.pop("species", ["human"]), tissue=kw.pop("tissue", ["PBMC"]),
                diseases=[], perturbations=[], genes=[], cell_types=[], surface_markers=[],
                frequencies=[], signatures=[], findings=findings)
    base.update(kw)
    return base


def gene(sym):
    return {"symbol": sym, "species": "human", "source_quote": ""}


def cell(name):
    return {"name": name, "source_quote": ""}


def pert(name):
    return {"name": name, "type": "", "source_quote": ""}


def disease(name):
    return {"name": name, "source_quote": ""}


def marker(m, g=""):
    return {"marker": m, "maps_to_gene": g, "source_quote": ""}


def sig(name, genes):
    return {"name": name, "genes": genes, "species": "human", "source_quote": ""}


def finding(stmt, quote=None):
    return {"statement": stmt, "source_quote": quote if quote is not None else stmt}


def run(records, cfg=None, grounder=STUB):
    cfg = cfg or HypothesisConfig()
    return ContextBoundHypothesisGenerator(cfg).run(records, grounder=grounder)


# --- normalization ---------------------------------------------------------------

def test_normalize_cell_types():
    assert NORM.cell_type("Treg").canonical_name == "regulatory T cell"
    assert NORM.cell_type("CD4 T").canonical_name == "CD4-positive T cell"


def test_normalize_surface_marker_bridge():
    e = NORM.surface_marker("CD25", "")
    assert e.type == "SURFACE_PROTEIN"
    assert e.attrs["maps_to_gene"] == "IL2RA"
    assert e.attrs["bridge_type"] == "surface_marker_to_gene"


def test_normalize_gene_and_greek_and_disease():
    assert NORM.gene("FOXP3", "human").type == "GENE_RNA"
    assert NORM.perturbation("IFN-β").canonical_name == "IFN-beta"
    assert NORM.disease("COVID-19").entity_id == "DISEASE:na:covid_19"
    assert NORM.disease("SARS-CoV-2 infection").entity_id == "DISEASE:na:covid_19"


def test_mouse_gene_lower_confidence():
    e = NORM.gene("Il2ra", "mouse")
    assert e.species == "mouse" and e.normalization_confidence < 1.0


# --- frame extraction ------------------------------------------------------------

def test_extract_interventional():
    r = rec("pA", [finding("IFN-beta increased FOXP3 in regulatory T cells")],
            perturbations=[pert("IFN-beta")], genes=[gene("FOXP3")], cell_types=[cell("regulatory T cell")])
    frames, _ = extract_frames([r], HypothesisConfig(), NORM)
    f = frames[0]
    assert f.direction == "increase"
    assert f.readout.canonical_name == "FOXP3"
    assert f.anchor.canonical_name == "IFN-beta"
    assert f.evidence_mode == "interventional"
    assert f.cell_type.canonical_name == "regulatory T cell"


def test_extract_case_control():
    r = rec("pB", [finding("Severe COVID-19 patients had elevated ISG15 in CD8 T cells")],
            diseases=[disease("COVID-19")], genes=[gene("ISG15")], cell_types=[cell("CD8 T")])
    frames, _ = extract_frames([r], HypothesisConfig(), NORM)
    f = frames[0]
    assert f.evidence_mode == "case_control"
    assert f.direction == "association_positive"
    assert f.readout.canonical_name == "ISG15"
    assert f.cell_type.canonical_name == "CD8-positive T cell"


def test_extract_descriptive_marker():
    r = rec("pC", [finding("FOXP3 and CD25 marked regulatory T cells")],
            genes=[gene("FOXP3")], surface_markers=[marker("CD25", "IL2RA")],
            cell_types=[cell("regulatory T cell")])
    frames, _ = extract_frames([r], HypothesisConfig(), NORM)
    assert frames and all(f.evidence_mode == "descriptive_marker" for f in frames)
    assert {f.readout.canonical_name for f in frames} == {"FOXP3", "CD25"}


def test_extract_abstains_on_no_readout():
    r = rec("pD", [finding("Patients were older than healthy controls")], diseases=[disease("COVID-19")])
    frames, diag = extract_frames([r], HypothesisConfig(), NORM)
    assert frames == []
    assert diag["frames_skipped"].get("missing_readout", 0) >= 1


# --- grounding ------------------------------------------------------------------

def _one_frame():
    r = rec("pA", [finding("IFN-beta increased FOXP3 in regulatory T cells")],
            perturbations=[pert("IFN-beta")], genes=[gene("FOXP3")], cell_types=[cell("regulatory T cell")])
    return extract_frames([r], HypothesisConfig(), NORM)[0][0]


def test_grounding_stub_entailed_retained():
    kept, diag = verify_frames([_one_frame()], HypothesisConfig(), STUB)
    assert len(kept) == 1 and kept[0].grounding_label == "entailed"


def test_grounding_contradicted_skipped():
    class FalseV:
        def verify(self, claim, passage):
            return (False, 0.3)
    kept, diag = verify_frames([_one_frame()], HypothesisConfig(), FrameGrounder(FalseV()))
    assert kept == []
    assert diag["skipped"][0]["reason"] == "grounding_failed"


def test_grounding_below_threshold_skipped():
    class HalfV:
        name = "half"
        def verify(self, frame):
            return replace(frame, grounding_label="entailed", grounding_score=0.5)
    cfg = HypothesisConfig(require_grounded_frames=False, min_grounding_score=0.8)
    kept, diag = verify_frames([_one_frame()], cfg, HalfV())
    assert kept == []
    assert diag["skipped"][0]["reason"] == "grounding_below_threshold"


def test_offline_lexical_grounder_handles_morphology():
    v = LexicalFrameVerifier()
    ok, score = v.verify("The paper reports that FOXP3 marks regulatory T cells.",
                         "FOXP3 and CD25 marked regulatory T cells.")
    assert ok


# --- context compatibility -------------------------------------------------------

def test_context_compat():
    cfg = HypothesisConfig()
    h_treg = BioContext(species=("human",), cell_type=("regulatory T cell",))
    h_treg2 = BioContext(species=("human",), cell_type=("Treg",))
    m_treg = BioContext(species=("mouse",), cell_type=("regulatory T cell",))
    h_cd4 = BioContext(species=("human",), cell_type=("CD4 T",))
    h_mono = BioContext(species=("human",), cell_type=("monocyte",))
    assert contexts_compatible(h_treg, h_treg2, cfg, NORM)[0]
    assert not contexts_compatible(h_treg, m_treg, cfg, NORM)[0]          # human vs mouse blocked
    ok, score, _ = contexts_compatible(h_treg, h_cd4, cfg, NORM)          # parent/child penalty
    assert ok and score < 1.0
    assert not contexts_compatible(h_treg, h_mono, cfg, NORM)[0]          # different lineage blocked


def test_t1_produces_candidate():
    recs = [
        rec("pA", [finding("IFN-beta increased FOXP3 in regulatory T cells")],
            perturbations=[pert("IFN-beta")], genes=[gene("FOXP3")], cell_types=[cell("regulatory T cell")]),
        rec("pB", [finding("FOXP3 marked regulatory T cells")],
            genes=[gene("FOXP3")], cell_types=[cell("regulatory T cell")]),
    ]
    res = run(recs)
    assert len(res.candidates) == 1
    c = res.candidates[0]
    assert c.motif == "perturbation_to_marker_state_completion"
    assert c.predicted_direction == "increase"
    assert c.anchor.canonical_name == "IFN-beta"


def test_t2_reversal_only_opposite_signs():
    recs = [
        rec("pC", [finding("Severe COVID-19 was associated with an increased ISG signature in CD8 T cells")],
            diseases=[disease("COVID-19")], cell_types=[cell("CD8 T")],
            signatures=[sig("ISG signature", ["ISG15", "MX1", "IFIT1"])]),
        rec("pD", [finding("JAK inhibitor decreased the ISG signature in CD8 T cells")],
            perturbations=[pert("JAK inhibitor")], cell_types=[cell("CD8 T")],
            signatures=[sig("ISG signature", ["ISG15", "MX1", "IFIT1"])]),
    ]
    res = run(recs)
    reversal = [c for c in res.candidates if c.motif == "disease_signature_reversal"]
    assert reversal, "opposite-sign disease vs perturbation should reverse"
    assert reversal[0].anchor.canonical_name == "JAK inhibitor"

    recs2 = [
        rec("pC", [finding("Severe COVID-19 was associated with an increased ISG signature in CD8 T cells")],
            diseases=[disease("COVID-19")], cell_types=[cell("CD8 T")],
            signatures=[sig("ISG signature", ["ISG15", "MX1", "IFIT1"])]),
        rec("pD", [finding("IFN-beta increased the ISG signature in CD8 T cells")],
            perturbations=[pert("IFN-beta")], cell_types=[cell("CD8 T")],
            signatures=[sig("ISG signature", ["ISG15", "MX1", "IFIT1"])]),
    ]
    res2 = run(recs2)
    assert not [c for c in res2.candidates if c.motif == "disease_signature_reversal"]


def test_t3_signature_requires_three_genes():
    genes3 = [gene("ISG15"), gene("MX1"), gene("IFIT1")]
    recs = [rec("pE",
                [finding("IFN-beta increased ISG15 in CD8 T cells"),
                 finding("IFN-beta increased MX1 in CD8 T cells"),
                 finding("IFN-beta increased IFIT1 in CD8 T cells")],
                perturbations=[pert("IFN-beta")], genes=genes3, cell_types=[cell("CD8 T")],
                signatures=[sig("ISG signature", ["ISG15", "MX1", "IFIT1"])])]
    res = run(recs)
    assert [c for c in res.candidates if c.motif == "signature_consolidation"]

    recs2 = [rec("pE",
                 [finding("IFN-beta increased ISG15 in CD8 T cells"),
                  finding("IFN-beta increased MX1 in CD8 T cells")],
                 perturbations=[pert("IFN-beta")], genes=[gene("ISG15"), gene("MX1")],
                 cell_types=[cell("CD8 T")], signatures=[sig("ISG signature", ["ISG15", "MX1", "IFIT1"])])]
    res2 = run(recs2)
    assert not [c for c in res2.candidates if c.motif == "signature_consolidation"]


def test_t4_bridge_warns():
    recs = [rec("pF", [finding("IL-2 increased IL2RA in regulatory T cells")],
                perturbations=[pert("IL-2")], genes=[gene("IL2RA")], cell_types=[cell("regulatory T cell")])]
    res = run(recs)
    bridges = [c for c in res.candidates if c.motif == "cite_seq_marker_bridge"]
    assert bridges
    assert any("bridge" in w.lower() for w in bridges[0].warnings)


def test_filter_generic_cell_type_dropped():
    recs = [rec("pG",
                [finding("Anti-CD3/CD28 increased CD69 in T cells"),
                 finding("CD69 marked T cells")],
                perturbations=[pert("anti-CD3/CD28")], genes=[gene("CD69")], cell_types=[cell("T cell")])]
    res = run(recs)
    assert res.diagnostics["candidates_filtered"].get("generic_cell_type", 0) >= 1
    assert all(" ".join(c.context.cell_type).lower() != "t cell" for c in res.candidates)


def test_filter_symbolic_restatement_dropped():
    base = [finding("IFN-beta increased ISG15 in CD8 T cells"),
            finding("IFN-beta increased MX1 in CD8 T cells"),
            finding("IFN-beta increased IFIT1 in CD8 T cells")]
    direct = base + [finding("IFN-beta increased the ISG signature in CD8 T cells")]
    recs = [rec("pE", direct, perturbations=[pert("IFN-beta")],
                genes=[gene("ISG15"), gene("MX1"), gene("IFIT1")], cell_types=[cell("CD8 T")],
                signatures=[sig("ISG signature", ["ISG15", "MX1", "IFIT1"])])]
    res = run(recs)
    assert not [c for c in res.candidates if c.motif == "signature_consolidation"]
    assert res.diagnostics["candidates_filtered"].get("not_locally_novel", 0) >= 1


def test_filter_observational_causal_language_blocked():
    g = build_evidence_graph([], HypothesisConfig(), NORM)
    anchor = NORM.disease("COVID-19")
    readout = NORM.gene("ISG15", "human")
    c = HypothesisCandidate(
        hypothesis_id="X", claim="COVID-19 causes higher ISG15 in CD8 T cells.",
        motif="perturbation_to_marker_state_completion", predicted_direction="association_positive",
        context=BioContext(species=("human",), cell_type=("CD8-positive T cell",)),
        anchor=anchor, mediator=readout, readouts=[readout],
        support_frame_ids=[], support_edge_ids=[], support_paper_ids=["pX"],
        evidence_mode_summary=["case_control"])
    kept, diag = filter_candidates([c], g, [], HypothesisConfig(), NORM)
    assert kept == []
    assert diag["by_reason"].get("observational_causal_language") == 1


def test_ranking_uses_min_grounding_and_clamps():
    recs = [
        rec("pA", [finding("IFN-beta increased FOXP3 in regulatory T cells")],
            perturbations=[pert("IFN-beta")], genes=[gene("FOXP3")], cell_types=[cell("regulatory T cell")]),
        rec("pB", [finding("FOXP3 marked regulatory T cells")],
            genes=[gene("FOXP3")], cell_types=[cell("regulatory T cell")]),
    ]
    res = run(recs)
    c = res.candidates[0]
    assert 0.0 <= c.scores["rank_score"] <= 1.0
    assert c.scores["risk_penalty"] >= 0.0


def test_ranking_min_not_mean():
    recs = [
        rec("pA", [finding("IFN-beta increased FOXP3 in regulatory T cells")],
            perturbations=[pert("IFN-beta")], genes=[gene("FOXP3")], cell_types=[cell("regulatory T cell")]),
        rec("pB", [finding("FOXP3 marked regulatory T cells")],
            genes=[gene("FOXP3")], cell_types=[cell("regulatory T cell")]),
    ]
    cfg = HypothesisConfig(require_grounded_frames=False, min_grounding_score=0.0)
    g = ContextBoundHypothesisGenerator(cfg)
    frames, _ = extract_frames(recs, cfg, NORM)
    grounded = []
    for f in frames:
        score = 1.0 if f.evidence_mode == "interventional" else 0.84
        grounded.append(replace(f, grounding_label="entailed", grounding_score=score))
    graph = build_evidence_graph(grounded, cfg, NORM, records=recs)
    raw = generate_candidates(graph, grounded, cfg, NORM)
    filt, _ = filter_candidates(raw, graph, grounded, cfg, NORM)
    ranked = rank_candidates(filt, graph, grounded, cfg, NORM)
    assert ranked
    assert ranked[0].scores["grounding_score"] == pytest.approx(0.84)  # min, not mean (0.92)


def _smoke_dir():
    return str(Path(__file__).parent / "fixtures" / "hypotheses" / "evidence")


def test_report_artifacts_and_interpretation_note(tmp_path):
    summary = run_to_dir(_smoke_dir(), tmp_path, HypothesisConfig(grounder="overlap"))
    assert summary["candidates"] == 1
    for name in ("hypotheses.jsonl", "hypotheses.csv", "hypotheses.md", "diagnostics.json",
                 "skipped_findings.csv", "hypothesis_graph.graphml"):
        assert (tmp_path / name).exists()
    lines = (tmp_path / "hypotheses.jsonl").read_text().splitlines()
    obj = json.loads(lines[0])
    assert obj["novelty_scope"] == "local_corpus_only" and obj["test_design"]
    header = (tmp_path / "hypotheses.csv").read_text().splitlines()[0]
    for col in ("rank", "hypothesis_id", "claim", "rank_score", "grounding_score"):
        assert col in header
    md = (tmp_path / "hypotheses.md").read_text()
    assert "hypothesis candidates, not validated discoveries" in md
    assert "```mermaid" in md


def test_zero_candidate_report_is_honest(tmp_path):
    ev = tmp_path / "ev"
    ev.mkdir()
    (ev / "empty_evidence.json").write_text(json.dumps(
        {"paper_id": "p", "relevance": "HIGH", "species": ["human"], "findings": []}))
    out = tmp_path / "out"
    summary = run_to_dir(ev, out, HypothesisConfig())
    assert summary["candidates"] == 0
    assert (out / "hypotheses.md").exists()
    assert (out / "diagnostics.json").exists()
    md = (out / "hypotheses.md").read_text()
    assert "No hypothesis candidates were generated" in md


def test_does_not_mutate_inputs(tmp_path):
    src = Path(_smoke_dir()) / "smoke_corpus.json"
    before = src.read_bytes()
    run_to_dir(_smoke_dir(), tmp_path, HypothesisConfig())
    assert src.read_bytes() == before


def test_mixed_direction_abstains():
    """'increased X but reduced Y' must not assign one sign to both readouts — abstain."""
    r = rec("p", [finding("IFN-beta increased FOXP3 but reduced CTLA4 in regulatory T cells")],
            perturbations=[pert("IFN-beta")], genes=[gene("FOXP3"), gene("CTLA4")],
            cell_types=[cell("regulatory T cell")])
    frames, diag = extract_frames([r], HypothesisConfig(), NORM)
    assert frames == []
    assert diag["frames_skipped"].get("ambiguous_mixed_direction", 0) == 1


@pytest.mark.parametrize("stmt", [
    "IFN-beta did not increase FOXP3 in regulatory T cells",
    "IFN-beta failed to increase FOXP3 in regulatory T cells",
    "IFN-beta showed no significant increase in FOXP3 in regulatory T cells",
])
def test_negated_direction_is_no_change(stmt):
    r = rec("p", [finding(stmt)], perturbations=[pert("IFN-beta")],
            genes=[gene("FOXP3")], cell_types=[cell("regulatory T cell")])
    frames, _ = extract_frames([r], HypothesisConfig(), NORM)
    assert len(frames) == 1
    assert frames[0].direction == "no_change"


def test_multiple_anchors_abstains():
    r = rec("p", [finding("IFN-beta increased FOXP3 while IL-2 increased CTLA4 in regulatory T cells")],
            perturbations=[pert("IFN-beta"), pert("IL-2")],
            genes=[gene("FOXP3"), gene("CTLA4")], cell_types=[cell("regulatory T cell")])
    frames, diag = extract_frames([r], HypothesisConfig(), NORM)
    assert frames == []
    assert diag["frames_skipped"].get("ambiguous_multiple_anchors", 0) == 1


def test_null_statement_does_not_crash():
    r = rec("p", [{"statement": None}, finding("IFN-beta increased FOXP3 in regulatory T cells")],
            perturbations=[pert("IFN-beta")], genes=[gene("FOXP3")], cell_types=[cell("regulatory T cell")])
    frames, _ = extract_frames([r], HypothesisConfig(), NORM)
    assert len(frames) == 1


def test_string_findings_field_ignored():
    r = rec("p", "IFN-beta increased FOXP3")          # malformed: a string, not a list
    frames, diag = extract_frames([r], HypothesisConfig(), NORM)
    assert frames == [] and diag["findings_seen"] == 0


def test_non_dict_records_no_crash():
    res = ContextBoundHypothesisGenerator(HypothesisConfig()).run([None, "junk", 42], grounder=STUB)
    assert res.candidates == []


def _t3_recs(pairs, sig_genes):
    return [rec(f"p{i}", [finding(f"IFN-beta {verb} {g} in {c}")],
                perturbations=[pert("IFN-beta")], genes=[gene(g)], cell_types=[cell(c)],
                signatures=[sig("ISG signature", sig_genes)])
            for i, (g, c, verb) in enumerate(pairs)]


def test_t3_blocks_cross_lineage():
    recs = _t3_recs([("ISG15", "regulatory T cell", "increased"),
                     ("MX1", "CD8 T cell", "increased"),
                     ("IFIT1", "monocyte", "increased")], ["ISG15", "MX1", "IFIT1"])
    res = run(recs)
    assert not [c for c in res.candidates if c.motif == "signature_consolidation"]


def test_t3_lists_only_concordant_genes():
    recs = _t3_recs([("ISG15", "CD8 T", "increased"), ("MX1", "CD8 T", "increased"),
                     ("IFIT1", "CD8 T", "increased"), ("OAS1", "CD8 T", "decreased")],
                    ["ISG15", "MX1", "IFIT1", "OAS1"])
    res = run(recs)
    cons = [c for c in res.candidates if c.motif == "signature_consolidation"]
    assert cons and cons[0].predicted_direction == "increase"
    assert "OAS1" not in cons[0].claim                 # discordant minority not called "concordant"
    assert "ISG15" in cons[0].claim
    assert len(cons[0].support_frame_ids) == 3
    assert all("oas1" not in fid for fid in cons[0].support_frame_ids)


def test_ranking_tiebreak_is_deterministic():
    def r(p, d, pr, stmt, c, sg):
        return rec(p, [finding(stmt)], diseases=[disease(d)] if d else [],
                   perturbations=[pert(pr)] if pr else [], cell_types=[cell(c)],
                   signatures=[sig(sg, ["A", "B", "C"])])
    recs = [r("p1", "COVID-19", None, "COVID-19 was associated with an increased ISG signature in CD8 T cells", "CD8 T", "ISG signature"),
            r("p2", None, "JAK inhibitor", "JAK inhibitor decreased the ISG signature in CD8 T cells", "CD8 T", "ISG signature"),
            r("p3", "lupus", None, "lupus was associated with an increased IFN signature in CD8 T cells", "CD8 T", "IFN signature"),
            r("p4", None, "baricitinib", "baricitinib decreased the IFN signature in CD8 T cells", "CD8 T", "IFN signature")]
    res = run(recs)
    rev = [c for c in res.candidates if c.motif == "disease_signature_reversal"]
    assert len(rev) == 2
    expected = sorted(rev, key=lambda c: (-c.scores["rank_score"], c.motif, c.anchor.entity_id, c.claim))
    assert [c.hypothesis_id for c in rev] == [c.hypothesis_id for c in expected]


def test_observational_denylist_expanded():
    g = build_evidence_graph([], HypothesisConfig(), NORM)
    anchor = NORM.disease("COVID-19")
    readout = NORM.gene("ISG15", "human")
    for verb in ("reduces", "reverses", "suppresses"):
        c = HypothesisCandidate(
            hypothesis_id="X", claim=f"COVID-19 {verb} ISG15 in CD8 T cells.",
            motif="perturbation_to_marker_state_completion", predicted_direction="association_positive",
            context=BioContext(species=("human",), cell_type=("CD8-positive T cell",)),
            anchor=anchor, mediator=readout, readouts=[readout],
            support_frame_ids=[], support_edge_ids=[], support_paper_ids=["pX"],
            evidence_mode_summary=["case_control"])
        kept, diag = filter_candidates([c], g, [], HypothesisConfig(), NORM)
        assert kept == [], f"{verb!r} should be blocked"


def test_filter_incompatible_context_defense_in_depth():
    """Defense-in-depth: a candidate whose support edges span incompatible contexts is dropped even
    if a (hypothetical) generator produced it."""
    hr = rec("ph", [finding("IFN-beta increased FOXP3 in regulatory T cells")],
             perturbations=[pert("IFN-beta")], genes=[gene("FOXP3")], cell_types=[cell("regulatory T cell")])
    mr = rec("pm", [finding("IFN-beta increased Foxp3 in monocytes")], species=["mouse"],
             perturbations=[pert("IFN-beta")], genes=[{"symbol": "Foxp3", "species": "mouse", "source_quote": ""}],
             cell_types=[cell("monocyte")])
    frames, _ = extract_frames([hr, mr], HypothesisConfig(), NORM)
    frames, _ = verify_frames(frames, HypothesisConfig(), STUB)   # grounded, so we reach the ctx gate
    graph = build_evidence_graph(frames, HypothesisConfig(), NORM, records=[hr, mr])
    eids = list(graph.graph["edges_by_id"])
    a = NORM.perturbation("IFN-beta")
    ro = NORM.gene("FOXP3", "human")
    c = HypothesisCandidate(
        hypothesis_id="X", claim="cross-context claim", motif="signature_consolidation",
        predicted_direction="increase", context=BioContext(species=("human",), cell_type=("regulatory T cell",)),
        anchor=a, mediator=None, readouts=[ro], support_frame_ids=list(graph.graph["frames_by_id"]),
        support_edge_ids=eids, support_paper_ids=["ph", "pm"], evidence_mode_summary=["interventional"])
    kept, diag = filter_candidates([c], graph, frames, HypothesisConfig(), NORM)
    assert kept == [] and diag["by_reason"].get("incompatible_context") == 1
