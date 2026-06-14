"""Hard filters plus local-novelty and restatement checks.

Aggressive by design: a candidate is dropped on the first failing constraint, and every drop is
recorded with its reason for the diagnostics report. Per-mediator / per-anchor caps are applied later
(in the ranker, after scoring) so the *best* candidate per group survives rather than an arbitrary one.
"""

from __future__ import annotations

import re

from .config import HypothesisConfig
from .graph_builder import relation_sign
from .normalize import Normalizer, contexts_compatible, normalize_text

# Action/causal/therapeutic verbs that must NOT appear in a claim built from observational-only
# support (the approved observational phrasing is "associated with higher/lower …"). Deliberately
# verbs of action, NOT the adjectives "higher"/"lower" (which the approved phrasing uses).
_CAUSAL_VERBS = ("causes", "cause", "drives", "drive", "induces", "induce", "rescues", "rescue",
                 "treats", "treat", "cures", "cure", "prevents", "prevent", "restores", "restore",
                 "increases", "increase", "decreases", "decrease", "reduces", "reduce",
                 "reverses", "reverse", "suppresses", "suppress", "inhibits", "inhibit",
                 "ameliorates", "ameliorate", "normalizes", "normalize", "abrogates", "abrogate",
                 "promotes", "promote", "triggers", "trigger", "blocks", "block",
                 "upregulates", "upregulate", "downregulates", "downregulate",
                 "augments", "augment", "enhances", "enhance", "depletes", "deplete")
_ASSAY_MODALITIES = ("rna", "surface_protein", "signature", "cell_frequency")


def filter_candidates(candidates, graph, frames, config: HypothesisConfig, norm: Normalizer | None = None):
    norm = norm or Normalizer()
    frames_by_id = graph.graph.get("frames_by_id", {})
    statements = [s["statement"] for s in graph.graph.get("finding_statements", [])]
    generic = {normalize_text(x) for x in config.generic_mediators}

    kept = []
    dropped = []
    reasons: dict[str, int] = {}

    def drop(c, reason):
        reasons[reason] = reasons.get(reason, 0) + 1
        dropped.append({"hypothesis_id": c.hypothesis_id, "claim": c.claim, "motif": c.motif,
                        "reason": reason})

    for c in candidates:
        support = [frames_by_id.get(fid) for fid in c.support_frame_ids]
        if any(f is None for f in support):
            drop(c, "missing_support_frame"); continue
        if config.require_grounded_frames and any(f.grounding_label != "entailed" for f in support):
            drop(c, "ungrounded_support"); continue
        if any(f.grounding_score < config.min_grounding_score for f in support):
            drop(c, "ungrounded_support"); continue

        if _support_contexts_incompatible(c, graph, config, norm):
            drop(c, "incompatible_context"); continue

        if c.predicted_direction in ("unknown", "no_change"):
            drop(c, "unknown_direction"); continue

        named_readouts = [r for r in c.readouts if r.is_named]
        if config.require_named_readout and not named_readouts:
            drop(c, "no_named_readout"); continue
        if config.require_named_cell_type and not any(ct.strip() for ct in c.context.cell_type):
            drop(c, "no_named_cell_type"); continue

        if not any(_modality(r) in _ASSAY_MODALITIES for r in c.readouts):
            drop(c, "no_single_cell_readout"); continue

        med_name = normalize_text(c.mediator.canonical_name) if c.mediator else ""
        cell_name = normalize_text(" ".join(c.context.cell_type))
        if med_name and med_name in generic:
            drop(c, "generic_mediator"); continue
        if cell_name in generic:
            drop(c, "generic_cell_type"); continue

        observational = all(m in ("case_control", "cross_sectional_association", "descriptive_marker",
                                  "descriptive_expression") for m in c.evidence_mode_summary)
        if observational and not config.allow_observational_causal_language:
            if any(re.search(rf"(?<![a-z]){v}(?![a-z])", c.claim.lower()) for v in _CAUSAL_VERBS):
                drop(c, "observational_causal_language"); continue

        has_adt = any(_modality(r) == "surface_protein" for r in c.readouts) or \
            any(rd.get("role") == "paired_readout" for rd in c.test_design.get("readouts", []))
        warned = any("bridge" in w.lower() or "adt" in w.lower() for w in c.warnings)
        if has_adt and not warned:
            drop(c, "unwarned_rna_protein_bridge"); continue

        if not any(config.relevance_ok(f.relevance) for f in support):
            drop(c, "all_support_below_relevance"); continue

        if _is_symbolic_restatement(c, graph):
            drop(c, "not_locally_novel"); continue

        if _is_textual_dup(c.claim, statements, config.token_jaccard_dup):
            drop(c, "not_locally_novel"); continue

        kept.append(c)

    diag = {"candidates_in": len(candidates), "candidates_kept": len(kept),
            "candidates_dropped": len(dropped), "by_reason": reasons, "dropped": dropped}
    return kept, diag


def _modality(entity) -> str:
    return {"GENE_RNA": "rna", "SURFACE_PROTEIN": "surface_protein", "SIGNATURE": "signature",
            "CELL_FREQUENCY": "cell_frequency"}.get(entity.type, "unknown")


def _support_contexts_incompatible(c, graph, config, norm) -> bool:
    """True if any pair of the candidate's support-edge contexts is incompatible (different species
    without allow_cross_species, or different cell lineage)."""
    edges_by_id = graph.graph.get("edges_by_id", {})
    ctxs = [edges_by_id[eid].context for eid in c.support_edge_ids if eid in edges_by_id]
    for i in range(len(ctxs)):
        for j in range(i + 1, len(ctxs)):
            if not contexts_compatible(ctxs[i], ctxs[j], config, norm)[0]:
                return True
    return False


def _is_symbolic_restatement(c, graph) -> bool:
    """True if the corpus already contains the exact edge this hypothesis asserts (same subject,
    object, sign). Bridge-only (T4) and reversal (T2) motifs aren't direct-edge claims, so they are
    judged by textual novelty / ranking instead."""
    if c.meta.get("bridge_only") or c.meta.get("reversal"):
        return False
    u = c.meta.get("novel_subject")
    v = c.meta.get("novel_object")
    sign = c.meta.get("novel_sign", 0)
    if not u or not v or u not in graph or v not in graph or not graph.has_edge(u, v):
        return False
    for _, _, d in graph.out_edges(u, data=True):
        if d.get("relation") and relation_sign(d["relation"]) == sign:
            edge = d["edge"]
            if edge.target_entity.entity_id == v:
                return True
    return False


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 3}


def _is_textual_dup(claim: str, statements: list[str], threshold: float) -> bool:
    ct = _tokens(claim)
    if not ct:
        return False
    for st in statements:
        st_tok = _tokens(st)
        if not st_tok:
            continue
        j = len(ct & st_tok) / len(ct | st_tok)
        if j > threshold:
            return True
    return False
