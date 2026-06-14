"""Transparent multi-axis ranking. Every component is stored on the candidate so the report can show
the breakdown. The final score is clamped to [0, 1].
Per-mediator / per-anchor caps and the global ``max_candidates`` truncation are applied after sorting,
so the strongest candidate in each group survives."""

from __future__ import annotations

import re

from .config import HypothesisConfig
from .filters import _CAUSAL_VERBS
from .graph_builder import relation_sign

_WEIGHTS = {
    "grounding_score": 0.30,
    "context_match_score": 0.20,
    "measurability_score": 0.20,
    "evidence_design_score": 0.15,
    "local_nonredundancy_score": 0.10,
    "specificity_score": 0.05,
}

_DESIGN = {"interventional": 1.0, "longitudinal": 0.8, "case_control": 0.65,
           "cross_sectional_association": 0.45, "descriptive_marker": 0.25,
           "descriptive_expression": 0.25, "unknown": 0.0}


def _modality(entity) -> str:
    return {"GENE_RNA": "rna", "SURFACE_PROTEIN": "surface_protein", "SIGNATURE": "signature",
            "CELL_FREQUENCY": "cell_frequency"}.get(entity.type, "unknown")


def rank_candidates(candidates, graph, frames, config: HypothesisConfig, norm=None):
    frames_by_id = graph.graph.get("frames_by_id", {})
    statements = [s["statement"] for s in graph.graph.get("finding_statements", [])]
    for c in candidates:
        _score_one(c, graph, frames_by_id, statements)

    # rank_score desc, with a deterministic content tie-breaker so equal-scored candidates get a
    # stable order (and stable HYP-ids) across processes regardless of PYTHONHASHSEED.
    candidates.sort(key=lambda c: (-c.scores.get("rank_score", 0.0), c.motif,
                                   c.anchor.entity_id, c.claim))
    capped = _apply_caps(candidates, config)
    return capped[: config.max_candidates]


def _score_one(c, graph, frames_by_id, statements):
    support = [frames_by_id[fid] for fid in c.support_frame_ids if fid in frames_by_id]
    grounding = min((f.grounding_score for f in support), default=0.0)   # use the weakest support
    ctx_match = _context_match(c)
    measurability = _measurability(c)
    design = _evidence_design(c)
    nonredundancy = _nonredundancy(c, graph, statements)
    specificity = _specificity(c)
    penalty = _risk_penalty(c, support)

    comps = {
        "grounding_score": grounding, "context_match_score": ctx_match,
        "measurability_score": measurability, "evidence_design_score": design,
        "local_nonredundancy_score": nonredundancy, "specificity_score": specificity,
        "risk_penalty": round(penalty, 3),
    }
    raw = sum(_WEIGHTS[k] * comps[k] for k in _WEIGHTS) - penalty
    comps["rank_score"] = round(max(0.0, min(1.0, raw)), 4)
    c.scores = {k: round(v, 4) for k, v in comps.items()}


def _context_match(c):
    ctx = c.context
    if "cell_type_parent_child" in c.warnings:
        return 0.6
    has_sp, has_ti = bool(ctx.species), bool(ctx.tissue)
    has_ct = any(x.strip() for x in ctx.cell_type)
    has_dp = bool(ctx.disease) or bool(ctx.perturbation)
    if has_sp and has_ct and has_ti and has_dp:
        return 1.0
    if has_sp and has_ct:
        return 0.8
    unknowns = sum(1 for x in (has_sp, has_ct, has_ti, has_dp) if not x)
    return 0.4 if unknowns >= 2 else 0.6


def _measurability(c):
    mods = [_modality(r) for r in c.readouts]
    assay = c.test_design.get("assay", "")
    cell_named = any(x.strip() for x in c.context.cell_type)
    has_comp = bool(c.context.comparator) or bool(c.context.condition)
    if "surface_protein" in mods and "CITE-seq" in assay:
        return 1.0
    if "rna" in mods and cell_named and has_comp:
        return 1.0
    if "rna" in mods and cell_named and any(rd.get("role") == "paired_readout"
                                            for rd in c.test_design.get("readouts", [])):
        return 1.0
    if "cell_frequency" in mods:
        return 0.9
    if "signature" in mods:
        genes = [rd for rd in c.test_design.get("readouts", [])
                 if rd.get("modality") == "signature" and len(rd.get("genes", [])) >= 3]
        return 0.8 if genes else 0.6
    if "rna" in mods and cell_named:
        return 0.6
    if "rna" in mods:
        return 0.6
    return 0.4 if mods else 0.0


def _evidence_design(c):
    modes = c.evidence_mode_summary or ["unknown"]
    effect = [m for m in modes if m not in ("descriptive_marker", "descriptive_expression")]
    if not effect:
        return 0.25                                  # all descriptive -> penalised floor
    return max(_DESIGN.get(m, 0.0) for m in effect)  # strongest effect-bearing edge


def _nonredundancy(c, graph, statements):
    u, v = c.meta.get("novel_subject"), c.meta.get("novel_object")
    if c.meta.get("bridge_only"):
        base = 0.4                                   # restates the RNA edge; the pairing is the novelty
    elif u and v and u in graph and v in graph and graph.has_edge(u, v):
        base = 0.0                                   # direct restatement (should have been filtered)
    else:
        endpoint_seen = (u in graph) or (v in graph)
        base = 0.7 if endpoint_seen else 1.0         # anchor/readout appears elsewhere -> 0.7
    # textual proximity to existing findings nudges it down
    sim = _max_sim(c.claim, statements)
    if sim > 0.6:
        base = min(base, 0.4)
    return base


def _specificity(c):
    ctx = c.context
    pts = 0
    pts += any(x.strip() for x in ctx.cell_type)
    pts += bool(ctx.disease) or bool(ctx.perturbation)
    pts += any(r.is_named for r in c.readouts)
    pts += bool(ctx.comparator)
    pts += bool(ctx.tissue)
    pts += any(r.type == "SIGNATURE" and len(r.attrs.get("genes", [])) >= 3 for r in c.readouts)
    return round(pts / 6.0, 3)


def _risk_penalty(c, support):
    p = 0.0
    observational = all(m in ("case_control", "cross_sectional_association") for m in c.evidence_mode_summary) \
        and bool(c.evidence_mode_summary)
    if observational and any(re.search(rf"(?<![a-z]){v}(?![a-z])", c.claim.lower()) for v in _CAUSAL_VERBS):
        p += 0.25
    if any("bridge" in w.lower() or "adt" in w.lower() for w in c.warnings):
        p += 0.20
    if len(set(c.support_paper_ids)) <= 1 and len(c.support_frame_ids) >= 2:
        p += 0.20                                    # multi-hop path from a single paper
    if any(w == "missing_comparator" for w in c.warnings):
        p += 0.15
    if c.mediator and c.mediator.normalization_confidence < 0.8:
        p += 0.15
    if any(getattr(r, "normalization_confidence", 1.0) < 0.8 for r in c.readouts):
        p += 0.15
    if not c.context.tissue:
        p += 0.10
    if len(set(c.support_paper_ids)) <= 1:
        p += 0.10
    return min(p, 0.95)


def _apply_caps(candidates, config: HypothesisConfig):
    per_mediator: dict[str, int] = {}
    per_anchor: dict[str, int] = {}
    out = []
    for c in candidates:                              # already sorted best-first
        mid = c.mediator.entity_id if c.mediator else f"_none::{c.motif}"
        aid = c.anchor.entity_id
        if per_mediator.get(mid, 0) >= config.max_candidates_per_mediator:
            continue
        if per_anchor.get(aid, 0) >= config.max_candidates_per_anchor:
            continue
        per_mediator[mid] = per_mediator.get(mid, 0) + 1
        per_anchor[aid] = per_anchor.get(aid, 0) + 1
        out.append(c)
    return out


def _tokens(text):
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 3}


def _max_sim(claim, statements):
    ct = _tokens(claim)
    if not ct:
        return 0.0
    best = 0.0
    for st in statements:
        stt = _tokens(st)
        if stt:
            best = max(best, len(ct & stt) / len(ct | stt))
    return best
