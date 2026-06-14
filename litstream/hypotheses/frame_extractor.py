"""Deterministic frame extraction (no LLM).

For each ``findings`` statement we use the record's own entity lists as the candidate vocabulary
(genes, markers, cell types, diseases, perturbations, signatures), detect which entities the finding
text names, classify direction + evidence mode from fixed lexicons, pick the readout(s), and emit one
:class:`FindingFrame` per readout. Ambiguous findings *abstain* and are logged with a reason — trust
over recall.
"""

from __future__ import annotations

import re
from typing import Any

from .config import HypothesisConfig
from .normalize import Normalizer, fold_greek
from .schema import BioContext, Entity, FindingFrame, slug

INCREASE_TERMS = ["increase", "increased", "increases", "increasing", "elevated", "higher",
                  "upregulated", "up-regulated", "upregulation", "induced", "induces", "expanded",
                  "enriched", "enhanced", "augmented", "greater", "more abundant"]
DECREASE_TERMS = ["decrease", "decreased", "decreases", "decreasing", "reduced", "reduction",
                  "lower", "downregulated", "down-regulated", "downregulation", "depleted",
                  "contracted", "suppressed", "diminished", "loss of", "fewer", "less abundant"]
ASSOCIATION_TERMS = ["associated with", "correlated with", "correlation", "linked to",
                     "predictive of", "related to", "associated"]
NEG_ASSOCIATION_CUES = ["inversely", "negatively", "anti-correlated", "negative correlation"]
NO_CHANGE_TERMS = ["no change", "unchanged", "not significantly different", "did not alter",
                   "no significant effect", "no difference", "not altered"]
NEGATED_EFFECT_PATTERNS = [
    r"\b(?:did|does|do)\s+not\s+(?:\w+\s+){0,2}__TERM__\b",
    r"\bnot\s+(?:significantly\s+)?(?:\w+\s+){0,2}__TERM__\b",
    r"\bno\s+(?:significant\s+)?__TERM__\b",
    r"\bfailed\s+to\s+(?:\w+\s+){0,2}__TERM__\b",
    r"\bfails\s+to\s+(?:\w+\s+){0,2}__TERM__\b",
    r"\bwithout\s+(?:\w+\s+){0,2}__TERM__\b",
]
MARKER_TERMS = ["marks", "marked by", "marked", "marker of", "markers of", "expressed by",
                "characterized by", "defined by", "identifies", "identified by", "delineates"]
EXPRESSION_TERMS = ["express", "expressed", "expresses", "expression of", "expressing"]

TIMEPOINT_CUES = ["day ", "days ", "week", "month", "baseline", "follow-up", "followup",
                  "before", "after", "over time", "trajectory", "longitudinal", "timepoint",
                  "time point", "kinetics", "dynamics over"]
CASE_CONTROL_CUES = ["severe", "mild", "moderate", "critical", "patient", "patients", "healthy",
                     "control", "controls", "responder", "non-responder", "nonresponder",
                     "vaccinated", "unvaccinated", "versus", " vs ", "compared with", "compared to",
                     "relative to", "case", "infected", "convalescent"]
FREQUENCY_CUES = ["frequency", "frequencies", "proportion", "proportions", "abundance", "percentage",
                  "percent", "fraction", "%", "numbers of", "counts of"]


def _searchable(text: str) -> str:
    return re.sub(r"\s+", " ", fold_greek(str(text or ""))).strip().casefold()


def _has_any(text: str, terms: list[str]) -> bool:
    return any(t in text for t in terms)


def _has_negated_effect(text: str, terms: list[str]) -> bool:
    """True when a direction cue is explicitly negated ("did not increase", "failed to reduce")."""
    for term in terms:
        escaped = re.escape(term)
        for pat in NEGATED_EFFECT_PATTERNS:
            if re.search(pat.replace("__TERM__", escaped), text):
                return True
    return False


def _mentions_form(text: str, form: str) -> bool:
    """Word-boundary match of one surface form, tolerant of a trailing plural 's'."""
    f = _searchable(form)
    if not f or len(f) < 2:
        return False
    pat = re.escape(f)
    rx = rf"(?<![a-z0-9]){pat}s?(?![a-z0-9])"
    return re.search(rx, text) is not None


class _RecordEntities:
    """Normalized entities for one record, each paired with the surface forms to look for in text."""

    def __init__(self, record: dict, norm: Normalizer):
        sp = " ".join(record.get("species", []) or [])
        self.genes: list[tuple[Entity, set[str]]] = []
        for g in record.get("genes", []) or []:
            sym = g.get("symbol") if isinstance(g, dict) else str(g)
            if not sym:
                continue
            gsp = (g.get("species") or sp) if isinstance(g, dict) else sp
            ent = norm.gene(sym, gsp)
            self.genes.append((ent, {sym, ent.canonical_name}))
        self.markers: list[tuple[Entity, set[str]]] = []
        for m in record.get("surface_markers", []) or []:
            name = m.get("marker") if isinstance(m, dict) else str(m)
            if not name:
                continue
            ent = norm.surface_marker(name, m.get("maps_to_gene", "") if isinstance(m, dict) else "", sp)
            self.markers.append((ent, {name, ent.canonical_name}))
        self.cell_types: list[tuple[Entity, set[str]]] = []
        for c in record.get("cell_types", []) or []:
            name = c.get("name") if isinstance(c, dict) else str(c)
            if not name:
                continue
            ent = norm.cell_type(name, sp)
            self.cell_types.append((ent, {name, ent.canonical_name} | norm_cell_aliases(norm, ent.canonical_name)))
        self.diseases: list[tuple[Entity, set[str]]] = []
        for d in record.get("diseases", []) or []:
            name = d.get("name") if isinstance(d, dict) else str(d)
            if not name:
                continue
            ent = norm.disease(name)
            self.diseases.append((ent, {name, ent.canonical_name}))
        self.perturbations: list[tuple[Entity, set[str]]] = []
        for p in record.get("perturbations", []) or []:
            name = p.get("name") if isinstance(p, dict) else str(p)
            if not name:
                continue
            ent = norm.perturbation(name)
            self.perturbations.append((ent, {name, ent.canonical_name}))
        self.signatures: list[tuple[Entity, set[str]]] = []
        for s in record.get("signatures", []) or []:
            name = s.get("name") if isinstance(s, dict) else str(s)
            if not name:
                continue
            ent = norm.signature(name, s.get("genes", []) if isinstance(s, dict) else [], sp)
            self.signatures.append((ent, {name, ent.canonical_name}))


def norm_cell_aliases(norm: Normalizer, canonical: str) -> set[str]:
    """All spellings whose canonical is `canonical` (so 'Treg' in text matches a 'regulatory T cell'
    record entity)."""
    from .normalize import _cell_index
    alias2canon, _ = _cell_index()
    return {a for a, c in alias2canon.items() if c == canonical}


def _present(text: str, items: list[tuple[Entity, set[str]]]) -> list[Entity]:
    return [ent for ent, forms in items if any(_mentions_form(text, f) for f in forms)]


def _modality_of(ent: Entity) -> str:
    return {"GENE_RNA": "rna", "SURFACE_PROTEIN": "surface_protein", "SIGNATURE": "signature",
            "CELL_FREQUENCY": "cell_frequency"}.get(ent.type, "unknown")


def extract_frames(
    records: list[dict], config: HypothesisConfig, norm: Normalizer | None = None
) -> tuple[list[FindingFrame], dict]:
    norm = norm or Normalizer()
    frames: list[FindingFrame] = []
    skipped: list[dict] = []
    findings_seen = 0
    reasons: dict[str, int] = {}

    def skip(pid: str, text: str, quote: str, reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1
        skipped.append({"paper_id": pid, "finding_text": text, "reason": reason, "source_quote": quote})

    for rec in records:
        relevance = rec.get("relevance", "NOT_USEFUL")
        if relevance not in ("HIGH", "MODERATE", "LOW"):
            continue
        pid = rec.get("paper_id", "")
        ents = _RecordEntities(rec, norm)
        species = tuple(rec.get("species", []) or [])
        tissue = tuple(rec.get("tissue", []) or [])
        findings_list = rec.get("findings") or []
        if not isinstance(findings_list, list):
            findings_list = []                       # a malformed scalar contributes zero findings
        for idx, finding in enumerate(findings_list):
            if isinstance(finding, dict):
                stmt = finding.get("statement") or ""
                quote = finding.get("source_quote") or ""
            elif isinstance(finding, str):
                stmt, quote = finding, finding
            else:
                continue
            if not stmt.strip():
                continue
            findings_seen += 1
            text = _searchable(stmt)

            new = _extract_from_finding(rec, pid, idx, stmt, quote, text, ents, species, tissue,
                                        relevance, norm, config)
            if isinstance(new, str):           # a skip reason
                skip(pid, stmt, quote, new)
            elif new:
                frames.extend(new)
            else:
                skip(pid, stmt, quote, "no_frame")

    diag = {
        "records_used": sum(1 for r in records if r.get("relevance") in ("HIGH", "MODERATE", "LOW")),
        "findings_seen": findings_seen,
        "frames_extracted": len(frames),
        "frames_skipped": reasons,
        "skipped": skipped,
    }
    return frames, diag


def _extract_from_finding(
    rec: dict, pid: str, idx: int, stmt: str, quote: str, text: str,
    ents: _RecordEntities, species: tuple[str, ...], tissue: tuple[str, ...],
    relevance: str, norm: Normalizer, config: HypothesisConfig,
) -> list[FindingFrame] | str | None:
    genes = _present(text, ents.genes)
    markers = _present(text, ents.markers)
    signatures = _present(text, ents.signatures)
    cell_types = _present(text, ents.cell_types)
    diseases = _present(text, ents.diseases)
    perturbations = _present(text, ents.perturbations)

    readouts: list[Entity] = genes + markers + signatures

    is_no_change = _has_any(text, NO_CHANGE_TERMS)
    is_inc = _has_any(text, INCREASE_TERMS)
    is_dec = _has_any(text, DECREASE_TERMS)
    neg_inc = _has_negated_effect(text, INCREASE_TERMS)
    neg_dec = _has_negated_effect(text, DECREASE_TERMS)
    if neg_inc or neg_dec:
        is_no_change = True
        if neg_inc:
            is_inc = False
        if neg_dec:
            is_dec = False
    is_assoc = _has_any(text, ASSOCIATION_TERMS)
    is_marker = _has_any(text, MARKER_TERMS)
    is_expr = _has_any(text, EXPRESSION_TERMS)

    has_pert = bool(perturbations)
    has_timepoint = _has_any(text, TIMEPOINT_CUES)
    has_casecontrol = bool(diseases) or _has_any(text, CASE_CONTROL_CUES)
    has_freq = _has_any(text, FREQUENCY_CUES)

    ct_inferred = False
    if cell_types:
        frame_cells = cell_types
    elif len(ents.cell_types) == 1:
        frame_cells = [ents.cell_types[0][0]]
        ct_inferred = True
    else:
        frame_cells = []

    if not readouts and has_freq and frame_cells and (is_inc or is_dec or is_assoc):
        ct = frame_cells[0]
        from .schema import make_entity_id
        freq = Entity(entity_id=make_entity_id("CELL_FREQUENCY", ct.species, ct.canonical_name + " frequency"),
                      type="CELL_FREQUENCY", canonical_name=f"{ct.canonical_name} frequency",
                      raw_names=(), species=ct.species, attrs={"of_cell_type": ct.canonical_name})
        readouts = [freq]

    if not readouts:
        return "missing_readout"

    if is_no_change:
        direction = "no_change"
        mode = "interventional" if has_pert else ("case_control" if has_casecontrol else "unknown")
    elif (is_marker or (is_expr and not (is_inc or is_dec))) and not (is_inc or is_dec or is_assoc):
        if not frame_cells:
            return "descriptive_without_cell_type"
        mode = "descriptive_marker" if is_marker else "descriptive_expression"
        direction = "unknown"
    elif is_inc and is_dec:
        return "ambiguous_mixed_direction"
    elif has_pert and (is_inc or is_dec):
        mode = "interventional"
        direction = "increase" if is_inc and not is_dec else ("decrease" if is_dec else "increase")
    elif has_timepoint and (is_inc or is_dec):
        mode = "longitudinal"
        direction = "increase" if is_inc and not is_dec else "decrease"
    elif has_casecontrol and (is_inc or is_dec or is_assoc):
        mode = "case_control"
        if is_inc and not is_dec:
            direction = "association_positive"
        elif is_dec and not is_inc:
            direction = "association_negative"
        else:
            direction = "association_negative" if _has_any(text, NEG_ASSOCIATION_CUES) else "association_positive"
    elif is_assoc:
        mode = "cross_sectional_association"
        direction = "association_negative" if _has_any(text, NEG_ASSOCIATION_CUES) else "association_positive"
    else:
        return "unparsed_direction"

    anchor: Entity | None = None
    if mode in ("interventional", "longitudinal") and perturbations:
        if len(perturbations) > 1:
            return "ambiguous_multiple_anchors"
        anchor = perturbations[0]
    elif mode == "case_control" and diseases:
        if len(diseases) > 1:
            return "ambiguous_multiple_anchors"
        anchor = diseases[0]

    comparator = _parse_comparator(text)
    timepoint = _parse_timepoint(stmt)
    warnings: list[str] = []
    if ct_inferred:
        warnings.append("cell_type_inferred")
    if mode == "case_control" and not comparator:
        warnings.append("missing_comparator")
    if mode in ("interventional",) and config.require_comparator_for_interventional_claims and not comparator:
        comparator = comparator or "unstimulated control"

    parser_conf = 1.0
    if ct_inferred:
        parser_conf -= 0.15
    if "missing_comparator" in warnings:
        parser_conf -= 0.1
    if is_inc and is_dec:
        parser_conf -= 0.15
        warnings.append("mixed_direction_terms")

    cell_name = frame_cells[0].canonical_name if frame_cells else ""
    frame_disease = tuple({d.canonical_name for d in diseases})
    frame_pert = tuple({p.canonical_name for p in perturbations}) if mode in ("interventional", "longitudinal") else ()

    out: list[FindingFrame] = []
    for r_i, readout in enumerate(readouts):
        ctx = BioContext(
            species=species, tissue=tissue, disease=frame_disease,
            cell_type=(cell_name,) if cell_name else (), perturbation=frame_pert,
            condition=_parse_condition(text), comparator=comparator, timepoint=timepoint,
        )
        atomic = _atomic_claim(anchor, readout, mode, direction, cell_name, comparator, frame_disease)
        frame = FindingFrame(
            frame_id=f"{pid}::f{idx}::{slug(readout.canonical_name)}::{r_i}",
            paper_id=pid, raw_statement=stmt, source_quote=quote, relevance=relevance,
            readout=readout, readout_modality=_modality_of(readout), direction=direction,
            context=ctx, evidence_mode=mode, parser_confidence=round(max(parser_conf, 0.1), 3),
            atomic_claim=atomic, anchor=anchor,
            cell_type=frame_cells[0] if frame_cells else None,
            warnings=tuple(warnings),
        )
        out.append(frame)
    return out


def _parse_comparator(text: str) -> str | None:
    for cue, comp in [("compared with", None), ("compared to", None), ("versus", None),
                      (" vs ", None), ("relative to", None)]:
        i = text.find(cue)
        if i >= 0:
            tail = text[i + len(cue):].strip()
            words = tail.split()
            if words:
                return " ".join(words[:4]).strip(" .,;")
    if "healthy control" in text or "healthy donor" in text:
        return "healthy control"
    if "unstimulated" in text:
        return "unstimulated control"
    return None


def _parse_condition(text: str) -> str | None:
    for cond in ["unstimulated", "stimulated", "activated", "resting", "severe", "mild", "infected"]:
        if cond in text:
            return cond
    return None


def _parse_timepoint(stmt: str) -> str | None:
    m = re.search(r"\b(day|week|month)\s*\d+\b", stmt, re.I)
    if m:
        return m.group(0)
    for tp in ["baseline", "follow-up", "convalescent"]:
        if tp in stmt.lower():
            return tp
    return None


def _atomic_claim(anchor: Entity | None, readout: Entity, mode: str, direction: str,
                  cell: str, comparator: str | None, disease: tuple[str, ...]) -> str:
    rn = readout.canonical_name
    measure = "expression" if readout.type in ("GENE_RNA", "SURFACE_PROTEIN") else \
        ("signature" if readout.type == "SIGNATURE" else "level")
    where = f" in {cell}" if cell else ""
    if mode in ("interventional", "longitudinal"):
        verb = {"increase": "increased", "decrease": "decreased", "no_change": "did not change"}.get(direction, "changed")
        who = anchor.canonical_name if anchor else "the perturbation"
        return f"The paper reports that {who} {verb} {rn} {measure}{where}."
    if mode == "case_control":
        word = {"association_positive": "higher", "association_negative": "lower",
                "no_change": "unchanged"}.get(direction, "altered")
        who = (disease[0] if disease else "the disease condition")
        comp = f" compared with {comparator}" if comparator else ""
        return f"The paper reports that {who} was associated with {word} {rn} {measure}{where}{comp}."
    if mode == "cross_sectional_association":
        word = "higher" if direction == "association_positive" else "lower"
        return f"The paper reports that {word} {rn} {measure} was observed{where}."
    if mode == "descriptive_marker":
        return f"The paper reports that {rn} marks {cell or 'the cell type'}."
    if mode == "descriptive_expression":
        return f"The paper reports that {cell or 'the cell type'} expresses {rn}."
    return f"The paper reports a finding about {rn}{where}."
