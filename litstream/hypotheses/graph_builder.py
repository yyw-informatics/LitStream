"""Build a typed, signed evidence graph from grounded frames.

``networkx.MultiDiGraph`` keyed by ``Entity.entity_id``; the :class:`EvidenceEdge` is stored as the
edge ``"edge"`` attribute (plus flat ``relation`` / ``direction`` / ``sign`` for cheap filtering). All
evidence is preserved — parallel edges from different papers coexist (MultiDiGraph). Auxiliary indexes
(frames-by-id, edges-by-id, signature membership, finding statements) ride on ``graph.graph`` so the
generators get everything from the one object.
"""

from __future__ import annotations

from typing import Any

from .config import HypothesisConfig
from .normalize import Normalizer, normalize_text
from .schema import EvidenceEdge, FindingFrame, RelationType

_DESCRIPTIVE_MODES = ("descriptive_marker", "descriptive_expression")


def relation_from_frame(frame: FindingFrame) -> RelationType | None:
    """Map a frame to its evidence-edge relation. Descriptive modes first (their direction is
    'unknown'); then signed/association directions. ``no_change`` produces no edge."""
    if frame.evidence_mode == "descriptive_marker":
        return "DEFINES_OR_MARKS"
    if frame.evidence_mode == "descriptive_expression":
        return "DESCRIPTIVE_EXPRESSION"
    if frame.direction == "increase":
        return "INCREASES_READOUT"
    if frame.direction == "decrease":
        return "DECREASES_READOUT"
    if frame.direction == "association_positive":
        return "ASSOCIATED_WITH_HIGHER_READOUT"
    if frame.direction == "association_negative":
        return "ASSOCIATED_WITH_LOWER_READOUT"
    return None


_SIGN = {
    "INCREASES_READOUT": 1, "ASSOCIATED_WITH_HIGHER_READOUT": 1,
    "DECREASES_READOUT": -1, "ASSOCIATED_WITH_LOWER_READOUT": -1,
    "DEFINES_OR_MARKS": 0, "DESCRIPTIVE_EXPRESSION": 0,
}


def relation_sign(relation: str | None) -> int:
    return _SIGN.get(relation or "", 0)


def _endpoints(frame: FindingFrame):
    """(source_entity, target_entity) for the frame's edge, or None if it can't be placed."""
    if frame.evidence_mode in _DESCRIPTIVE_MODES:
        if frame.cell_type is None:
            return None
        return (frame.readout, frame.cell_type)            # readout marks/expresses-in cell type
    if frame.anchor is None:
        return None
    return (frame.anchor, frame.readout)                   # driver -> readout


def build_signature_index(records: list[dict], norm: Normalizer) -> dict[str, list[dict]]:
    """{normalized gene -> [{'name','genes','species'}...]} from each record's ``signatures``."""
    idx: dict[str, list[dict]] = {}
    for rec in records or []:
        sp = " ".join(rec.get("species", []) or [])
        for sig in rec.get("signatures", []) or []:
            if not isinstance(sig, dict):
                continue
            genes = [g for g in (sig.get("genes") or []) if g]
            if not genes:
                continue
            entry = {"name": sig.get("name", ""), "genes": genes, "species": sp}
            for g in genes:
                idx.setdefault(normalize_text(g), []).append(entry)
    return idx


def build_evidence_graph(
    frames: list[FindingFrame], config: HypothesisConfig,
    norm: Normalizer | None = None, records: list[dict] | None = None,
):
    import networkx as nx
    norm = norm or Normalizer()
    g = nx.MultiDiGraph()
    edges_by_id: dict[str, EvidenceEdge] = {}
    frames_by_id: dict[str, FindingFrame] = {}
    statements: list[dict] = []

    for fr in frames:
        frames_by_id[fr.frame_id] = fr
        statements.append({"paper_id": fr.paper_id, "statement": fr.raw_statement})
        ends = _endpoints(fr)
        relation = relation_from_frame(fr)
        if ends is None or relation is None:
            continue
        src, tgt = ends
        _add_node(g, src)
        _add_node(g, tgt)
        edge = EvidenceEdge(
            edge_id=fr.frame_id, source_entity=src, relation=relation, target_entity=tgt,
            frame_id=fr.frame_id, paper_id=fr.paper_id, context=fr.context,
            evidence_mode=fr.evidence_mode, grounding_score=fr.grounding_score,
            parser_confidence=fr.parser_confidence, source_quote=fr.source_quote,
            raw_statement=fr.raw_statement, direction=fr.direction, warnings=fr.warnings,
        )
        edges_by_id[edge.edge_id] = edge
        g.add_edge(src.entity_id, tgt.entity_id, key=edge.edge_id, edge=edge,
                   relation=relation, direction=fr.direction, sign=relation_sign(relation))

    g.graph["frames"] = frames
    g.graph["frames_by_id"] = frames_by_id
    g.graph["edges_by_id"] = edges_by_id
    g.graph["finding_statements"] = statements
    g.graph["signature_index"] = build_signature_index(records or [], norm)
    g.graph["adequacy"] = _adequacy(g)
    return g


def _add_node(g, ent) -> None:
    g.add_node(ent.entity_id, entity=ent, label=ent.canonical_name, type=ent.type)


def _adequacy(g) -> dict[str, Any]:
    """Pre-generation graph stats — surfaced in diagnostics so a too-sparse graph reads as an honest
    non-result rather than a silent empty table."""
    n_edges = g.number_of_edges()
    pivots = [n for n in g.nodes if g.in_degree(n) >= 1 and g.out_degree(n) >= 1]
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": n_edges,
        "n_pivot_nodes": len(pivots),
        "pivot_ids": pivots,
    }
