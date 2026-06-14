"""Candidate generation with auditable biological motifs, deterministic claim rendering, and
single-cell test-design synthesis.

Each generator composes only across contexts that :func:`normalize.contexts_compatible` accepts and
emits :class:`HypothesisCandidate` objects with full provenance. Wording never asserts causality from
observational support and never asserts therapy from a disease-reversal motif.
"""

from __future__ import annotations

from .config import HypothesisConfig
from .graph_builder import relation_sign
from .normalize import Normalizer, contexts_compatible, normalize_text
from .schema import (
    BioContext, Entity, EvidenceEdge, HypothesisCandidate, make_entity_id,
)

_EFFECT_RELATIONS = ("INCREASES_READOUT", "DECREASES_READOUT",
                     "ASSOCIATED_WITH_HIGHER_READOUT", "ASSOCIATED_WITH_LOWER_READOUT")
_MARKER_RELATIONS = ("DEFINES_OR_MARKS", "DESCRIPTIVE_EXPRESSION")
_OBSERVATIONAL_MODES = ("case_control", "cross_sectional_association")


def generate_candidates(graph, frames, config: HypothesisConfig, norm: Normalizer | None = None):
    norm = norm or Normalizer()
    cands: list[HypothesisCandidate] = []
    cands += _t1_perturbation_to_marker_state(graph, config, norm)
    cands += _t2_disease_signature_reversal(graph, config, norm)
    cands += _t3_signature_consolidation(graph, config, norm)
    cands += _t4_cite_seq_marker_bridge(graph, config, norm)
    return _dedup(cands)


def _all_edges(graph) -> list[EvidenceEdge]:
    return [d["edge"] for _, _, d in graph.edges(data=True)]


def _t1_perturbation_to_marker_state(graph, config, norm):
    out: list[HypothesisCandidate] = []
    edges = _all_edges(graph)
    by_source = _index_by_source(edges)
    for e1 in edges:
        if e1.source_entity.type != "PERTURBATION" or e1.relation not in _EFFECT_RELATIONS:
            continue
        m = e1.target_entity
        for e2 in by_source.get(m.entity_id, []):
            if e2.relation not in _MARKER_RELATIONS:
                continue
            s = e2.target_entity
            if s.type not in ("CELL_TYPE", "CELL_STATE", "SIGNATURE", "PHENOTYPE"):
                continue
            ok, cscore, notes = contexts_compatible(e1.context, e2.context, config, norm)
            if not ok:
                continue
            if graph.has_edge(e1.source_entity.entity_id, s.entity_id):
                continue
            cell = s.canonical_name if s.type in ("CELL_TYPE", "CELL_STATE") else _ctx_cell(e1, e2)
            ctx = _compose_context(e1, e2, cell)
            observational = e1.evidence_mode in _OBSERVATIONAL_MODES
            direction = e1.direction
            warns = list(notes)
            if observational:
                warns.append("observational_support_only")
            paired, bridge_warn = _paired_adt(graph, s.entity_id, norm)
            warns += bridge_warn
            claim = _render_t1(ctx, e1.source_entity, m, cell, direction, observational, e1.context.comparator)
            cand = HypothesisCandidate(
                hypothesis_id="", claim=claim,
                motif="perturbation_to_marker_state_completion",
                predicted_direction=direction, context=ctx,
                anchor=e1.source_entity, mediator=m, readouts=[m],
                support_frame_ids=[e1.frame_id, e2.frame_id],
                support_edge_ids=[e1.edge_id, e2.edge_id],
                support_paper_ids=_papers(e1, e2),
                evidence_mode_summary=[e1.evidence_mode, e2.evidence_mode],
                warnings=_uniq(warns),
            )
            cand.meta = {"novel_subject": e1.source_entity.entity_id, "novel_object": s.entity_id,
                         "novel_sign": relation_sign(e1.relation)}
            cand.test_design = _test_design(cand, graph, norm, paired)
            out.append(cand)
    return out


def _t2_disease_signature_reversal(graph, config, norm):
    out: list[HypothesisCandidate] = []
    edges = _all_edges(graph)
    dis = {}
    pert = {}
    for e in edges:
        if e.relation not in _EFFECT_RELATIONS:
            continue
        if e.source_entity.type == "DISEASE" and e.target_entity.type in ("SIGNATURE", "GENE_RNA"):
            dis.setdefault(e.target_entity.entity_id, []).append(e)
        elif e.source_entity.type == "PERTURBATION" and e.target_entity.type in ("SIGNATURE", "GENE_RNA"):
            pert.setdefault(e.target_entity.entity_id, []).append(e)
    for s_id in sorted(set(dis) & set(pert)):       # sorted: set-iteration order is hash-seeded
        for de in dis[s_id]:
            for pe in pert[s_id]:
                if relation_sign(de.relation) == 0 or relation_sign(pe.relation) == 0:
                    continue
                if relation_sign(de.relation) != -relation_sign(pe.relation):
                    continue                                  # need opposite signs
                ok, cscore, notes = contexts_compatible(de.context, pe.context, config, norm)
                if not ok:
                    continue
                s = de.target_entity
                cell = _ctx_cell(de, pe)
                ctx = _compose_context(de, pe, cell)
                dis_dir = "increase" if relation_sign(de.relation) > 0 else "decrease"
                claim = _render_t2(ctx, pe.source_entity, de.source_entity, s, cell, dis_dir)
                cand = HypothesisCandidate(
                    hypothesis_id="", claim=claim, motif="disease_signature_reversal",
                    predicted_direction=pe.direction, context=ctx,
                    anchor=pe.source_entity, mediator=de.source_entity, readouts=[s],
                    support_frame_ids=[de.frame_id, pe.frame_id],
                    support_edge_ids=[de.edge_id, pe.edge_id],
                    support_paper_ids=_papers(de, pe),
                    evidence_mode_summary=[de.evidence_mode, pe.evidence_mode],
                    warnings=_uniq(list(notes)),
                )
                cand.meta = {"novel_subject": pe.source_entity.entity_id,
                             "novel_object": s.entity_id, "novel_sign": relation_sign(pe.relation),
                             "reversal": True}
                cand.test_design = _test_design(cand, graph, norm, [])
                out.append(cand)
    return out


def _t3_signature_consolidation(graph, config, norm):
    out: list[HypothesisCandidate] = []
    sig_index = graph.graph.get("signature_index", {})
    if not sig_index:
        return out
    edges = _all_edges(graph)
    groups: dict[tuple[str, str], list[tuple[EvidenceEdge, dict]]] = {}
    for e in edges:
        if e.source_entity.type != "PERTURBATION" or e.target_entity.type != "GENE_RNA":
            continue
        if e.relation not in ("INCREASES_READOUT", "DECREASES_READOUT"):
            continue
        for sig in sig_index.get(normalize_text(e.target_entity.canonical_name), []):
            groups.setdefault((e.source_entity.entity_id, sig["name"]), []).append((e, sig))
    for (pert_id, sig_name), items in sorted(groups.items()):     # sorted: deterministic order
        per_gene: dict[str, EvidenceEdge] = {}
        for e, _sig in items:
            g = normalize_text(e.target_entity.canonical_name)
            if g not in per_gene or e.parser_confidence > per_gene[g].parser_confidence:
                per_gene[g] = e
        ref = max(per_gene.values(), key=lambda e: (e.parser_confidence, e.edge_id))
        compatible = [e for e in per_gene.values()
                      if contexts_compatible(ref.context, e.context, config, norm)[0]]
        if len(compatible) < config.signature_min_genes:
            continue
        ups = [e for e in compatible if e.direction == "increase"]
        downs = [e for e in compatible if e.direction == "decrease"]
        frac_up = len(ups) / len(compatible)
        if max(frac_up, 1 - frac_up) < config.signature_majority_frac:
            continue                                          # conflicting signs -> skip
        direction = "increase" if frac_up >= 0.5 else "decrease"
        concordant = ups if direction == "increase" else downs    # only the genes that agree
        sig_genes = next((it[1].get("genes") for it in items), [])
        sig_ent = norm.signature(sig_name, sig_genes, " ".join(ref.context.species))
        cell = " ".join(ref.context.cell_type)
        ctx = _compose_context(ref, ref, cell)
        member_genes = [e.target_entity.canonical_name for e in concordant]
        claim = _render_t3(ctx, ref.source_entity, sig_name, cell, direction, member_genes)
        cand = HypothesisCandidate(
            hypothesis_id="", claim=claim, motif="signature_consolidation",
            predicted_direction=direction, context=ctx,
            anchor=ref.source_entity, mediator=None, readouts=[sig_ent],
            support_frame_ids=[e.frame_id for e in concordant],
            support_edge_ids=[e.edge_id for e in concordant],
            support_paper_ids=_uniq([e.paper_id for e in concordant]),
            evidence_mode_summary=_uniq([e.evidence_mode for e in concordant]),
            warnings=[f"consolidated_from_{len(concordant)}_concordant_member_genes"],
        )
        cand.meta = {"novel_subject": ref.source_entity.entity_id,
                     "novel_object": sig_ent.entity_id,
                     "novel_sign": 1 if direction == "increase" else -1}
        cand.test_design = _test_design(cand, graph, norm, [])
        out.append(cand)
    return out


def _t4_cite_seq_marker_bridge(graph, config, norm):
    out: list[HypothesisCandidate] = []
    for e in _all_edges(graph):
        if e.source_entity.type != "PERTURBATION" or e.target_entity.type != "GENE_RNA":
            continue
        if e.relation not in ("INCREASES_READOUT", "DECREASES_READOUT"):
            continue
        gene = e.target_entity
        marker = gene.attrs.get("measured_as_marker") or norm.marker_for_gene(gene.canonical_name)
        if not marker:
            continue
        cell = " ".join(e.context.cell_type)
        ctx = _compose_context(e, e, cell)
        marker_ent = norm.surface_marker(marker, gene.canonical_name, " ".join(e.context.species))
        claim = _render_t4(ctx, e.source_entity, gene, marker, cell, e.direction)
        warn = (f"{marker} ADT is a paired readout inferred through marker-gene mapping; "
                "protein change is not directly supported.")
        cand = HypothesisCandidate(
            hypothesis_id="", claim=claim, motif="cite_seq_marker_bridge",
            predicted_direction=e.direction, context=ctx,
            anchor=e.source_entity, mediator=gene, readouts=[gene, marker_ent],
            support_frame_ids=[e.frame_id], support_edge_ids=[e.edge_id],
            support_paper_ids=[e.paper_id], evidence_mode_summary=[e.evidence_mode],
            warnings=[warn, "rna_to_surface_protein_bridge"],
        )
        cand.meta = {"novel_subject": e.source_entity.entity_id, "novel_object": gene.entity_id,
                     "novel_sign": relation_sign(e.relation), "bridge_only": True}
        cand.test_design = _test_design(cand, graph, norm, [(marker_ent, "paired_readout")])
        out.append(cand)
    return out


def _setting(ctx) -> str:
    """Species + tissue prefix for claims; the cell type goes in the explicit 'in {cell}' clause so
    it isn't repeated."""
    bits = []
    if ctx.species:
        bits.append(" ".join(ctx.species))
    if ctx.tissue:
        bits.append(" ".join(ctx.tissue))
    return " ".join(bits) or "the studied context"


def _render_t1(ctx, pert, mediator, cell, direction, observational, comparator):
    where = f" in {cell}" if cell else ""
    cx = _setting(ctx)
    if observational:
        word = "higher" if direction in ("increase", "association_positive") else "lower"
        return (f"In {cx}, {pert.canonical_name} is predicted to be associated with a {word} "
                f"{mediator.canonical_name}-associated readout{where}.")
    word = "increase" if direction == "increase" else "decrease"
    comp = f" relative to {comparator}" if comparator else ""
    return (f"In {cx}, {pert.canonical_name} is predicted to {word} a "
            f"{mediator.canonical_name}-associated readout{where}{comp}.")


def _render_t2(ctx, pert, disease, sig, cell, dis_dir):
    where = f" in {cell}" if cell else ""
    cx = _setting(ctx)
    kind = "signature" if sig.type == "SIGNATURE" else "expression"
    return (f"In {cx}, {pert.canonical_name} is predicted to reduce the {disease.canonical_name}-"
            f"associated {dis_dir} in {sig.canonical_name} {kind}{where}.")


def _render_t3(ctx, pert, sig_name, cell, direction, member_genes):
    where = f" in {cell}" if cell else ""
    cx = _setting(ctx)
    word = "increase" if direction == "increase" else "decrease"
    genes = ", ".join(member_genes[:6])
    return (f"In {cx}, {pert.canonical_name} is predicted to {word} the {sig_name} signature{where} "
            f"(concordant member genes: {genes}).")


def _render_t4(ctx, pert, gene, marker, cell, direction):
    where = f" in {cell}" if cell else ""
    cx = _setting(ctx)
    word = "increase" if direction == "increase" else "decrease"
    return (f"In {cx}, {pert.canonical_name} is predicted to {word} {gene.canonical_name} RNA{where}; "
            f"{marker} ADT is proposed as a paired CITE-seq readout.")


def _test_design(cand: HypothesisCandidate, graph, norm, extra_paired):
    ctx = cand.context
    species = " ".join(ctx.species) or "human"
    tissue = " ".join(ctx.tissue) or "PBMC"
    cell = " ".join(ctx.cell_type) or "the target cell population"
    has_adt = bool(extra_paired) or any(r.type == "SURFACE_PROTEIN" for r in cand.readouts)
    assay = "CITE-seq" if has_adt else "scRNA-seq or CITE-seq"

    if ctx.perturbation:
        p = ctx.perturbation[0]
        conditions = [f"{p}-stimulated", ctx.comparator or "unstimulated control"]
    elif ctx.disease:
        conditions = [ctx.disease[0], ctx.comparator or "matched control"]
    else:
        conditions = [ctx.condition or "condition", ctx.comparator or "control"]

    readouts: list[dict] = []
    for r in cand.readouts:
        if r.type == "SIGNATURE":
            readouts.append({"modality": "signature", "entity": r.canonical_name,
                             "genes": r.attrs.get("genes", [])})
        elif r.type == "SURFACE_PROTEIN":
            readouts.append({"modality": "surface_protein", "entity": r.canonical_name,
                             "role": "paired_readout"})
        elif r.type == "CELL_FREQUENCY":
            readouts.append({"modality": "cell_frequency", "entity": r.canonical_name})
        else:
            readouts.append({"modality": "rna", "entity": r.canonical_name})
    for ent, role in extra_paired:
        if not any(rd.get("entity") == ent.canonical_name for rd in readouts):
            readouts.append({"modality": "surface_protein", "entity": ent.canonical_name, "role": role})

    analysis = [
        "annotate the target cell population",
        "aggregate expression by biological replicate (donor/sample) within the cell type",
        "compare the condition effect with a donor/sample-aware model (not per-cell)",
        "report effect size and confidence interval",
    ]
    if has_adt:
        analysis.append("treat any RNA-to-surface-protein bridge as exploratory unless direct ADT "
                        "evidence exists")
    return {"assay": assay, "sample": f"{species} {tissue}", "conditions": conditions,
            "cell_population": cell, "readouts": readouts, "analysis": analysis}


def _paired_adt(graph, cell_id, norm):
    """Surface markers that mark the candidate's cell type in the corpus → proposed paired ADT
    readouts (with the bridge caveat). This is how a T1 candidate in Tregs picks up 'CD25 ADT'."""
    paired: list[tuple[Entity, str]] = []
    warns: list[str] = []
    if cell_id not in graph:
        return paired, warns
    for u, _v, d in graph.in_edges(cell_id, data=True):
        e = d["edge"]
        if e.relation not in _MARKER_RELATIONS:
            continue
        src = e.source_entity
        marker_name = None
        if src.type == "SURFACE_PROTEIN":
            marker_name = src.canonical_name
        elif src.type == "GENE_RNA":
            marker_name = norm.marker_for_gene(src.canonical_name)
        if marker_name:
            ent = norm.surface_marker(marker_name, "", " ".join(e.context.species))
            paired.append((ent, "paired_readout"))
            warns.append(f"{marker_name} ADT is a paired readout inferred through marker-gene "
                         "mapping; protein change is not directly supported.")
    return paired, warns


def _index_by_source(edges):
    idx: dict[str, list[EvidenceEdge]] = {}
    for e in edges:
        idx.setdefault(e.source_entity.entity_id, []).append(e)
    return idx


def _ctx_cell(e1, e2):
    for e in (e1, e2):
        if e.context.cell_type:
            return e.context.cell_type[0]
    return ""


def _compose_context(e1, e2, cell):
    def u(a, b):
        return tuple(dict.fromkeys([x for x in (a + b) if x]))
    c1, c2 = e1.context, e2.context
    return BioContext(
        species=u(c1.species, c2.species), tissue=u(c1.tissue, c2.tissue),
        disease=u(c1.disease, c2.disease),
        cell_type=(cell,) if cell else u(c1.cell_type, c2.cell_type),
        cell_state=u(c1.cell_state, c2.cell_state),
        perturbation=u(c1.perturbation, c2.perturbation),
        condition=c1.condition or c2.condition,
        comparator=c1.comparator or c2.comparator,
        timepoint=c1.timepoint or c2.timepoint,
    )


def _papers(*edges):
    return _uniq([e.paper_id for e in edges])


def _uniq(seq):
    return list(dict.fromkeys(seq))


def _dedup(cands):
    seen: dict[tuple, HypothesisCandidate] = {}
    for c in cands:
        key = (c.motif, c.anchor.entity_id, tuple(sorted(r.entity_id for r in c.readouts)),
               c.predicted_direction, " ".join(c.context.cell_type))
        if key in seen:
            ex = seen[key]
            ex.support_frame_ids = _uniq(ex.support_frame_ids + c.support_frame_ids)
            ex.support_edge_ids = _uniq(ex.support_edge_ids + c.support_edge_ids)
            ex.support_paper_ids = _uniq(ex.support_paper_ids + c.support_paper_ids)
        else:
            c.hypothesis_id = _key_id(key)
            seen[key] = c
    return list(seen.values())


def _key_id(key) -> str:
    import hashlib
    return "cand_" + hashlib.sha1("::".join(str(k) for k in key).encode()).hexdigest()[:10]
