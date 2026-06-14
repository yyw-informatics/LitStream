"""Canonical extraction-record fields — the single source of truth, shared with MINE.

These field names MUST match exactly what the MINE extractor emits, so the same scorer
grades both benchmark runs and real MINE output. The JSON mirror is record_schema.example.json.

Per scored field we define:
  - key(item): turns one item-object into the string we match on
  - match:     which matcher family grades it ("entity" substring / "number" overlap)
  - benchmark: which public dataset's gold stands in for it

  genes             -> match item["symbol"]      · BioRED gene entities
  species           -> the string itself         · BioRED species entities
  cell_types        -> match item["name"]        · JNLPBA cell-type entities
  frequencies       -> "value unit"              · MeasEval quantities (+ measured entity)
  gating_thresholds -> str(value)                · MeasEval quantities

  surface_markers, signatures -> no public benchmark; kept in the schema, not scored
  relevance, tissue           -> internal, not graded
"""

from __future__ import annotations

from .schema import normalize

# field names — verbatim from the agreed schema
PAPER_ID = "paper_id"
RELEVANCE = "relevance"
SPECIES = "species"
TISSUE = "tissue"
GENES = "genes"
CELL_TYPES = "cell_types"
SURFACE_MARKERS = "surface_markers"
FREQUENCIES = "frequencies"
GATING_THRESHOLDS = "gating_thresholds"
SIGNATURES = "signatures"


def _gene_key(it):
    return it.get("symbol", "") if isinstance(it, dict) else str(it)


def _species_key(it):
    if isinstance(it, str):
        return it
    return it.get("name") or it.get("symbol") or ""


def _cell_key(it):
    return it.get("name", "") if isinstance(it, dict) else str(it)


def _freq_key(it):
    if not isinstance(it, dict):
        return str(it)
    return f"{it.get('value', '')} {it.get('unit', '') or ''}".strip()


def _gate_key(it):
    if not isinstance(it, dict):
        return str(it)
    return f"{it.get('operator', '') or ''}{it.get('value', '')}"  # operator is meaning-bearing


# field -> how to grade it
SCORED = {
    GENES: {"key": _gene_key, "match": "entity"},
    SPECIES: {"key": _species_key, "match": "entity"},
    CELL_TYPES: {"key": _cell_key, "match": "entity"},
    FREQUENCIES: {"key": _freq_key, "match": "number"},
    GATING_THRESHOLDS: {"key": _gate_key, "match": "number"},
}
GAP = (SURFACE_MARKERS, SIGNATURES)
UNGRADED = (RELEVANCE, TISSUE)


def matchable(record: dict, field: str) -> set[str]:
    """The set of normalized strings to compare for one field of one record."""
    key = SCORED[field]["key"]
    out: set[str] = set()
    for item in record.get(field, []) or []:
        s = normalize(key(item))
        if s:
            out.add(s)
    return out
