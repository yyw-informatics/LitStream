"""The evidence record — the structure extraction fills.

One source of truth for two consumers: the structured-output step (`structure_evidence.py`) that
emits this, and the benchmark scorer that grades against it. Every extracted item carries a
`source_quote` so grounding can later check it is actually backed by the paper (catching invented
entries).

`JSON_SCHEMA` (below) is a standard JSON Schema usable directly as a function-calling / JSON-mode
response format for the LLM extractor. `validate()` is a small dependency-free checker for the same
shape.
"""

from __future__ import annotations

RELEVANCE = ["HIGH", "MODERATE", "LOW", "NOT_USEFUL"]

ITEM_REQUIRED: dict[str, list[str]] = {
    "diseases":          ["name", "source_quote"],
    "perturbations":     ["name", "source_quote"],
    "genes":             ["symbol", "source_quote"],
    "cell_types":        ["name", "source_quote"],
    "surface_markers":   ["marker", "source_quote"],
    "frequencies":       ["cell_type", "value", "source_quote"],
    "cohort":            ["group", "n", "source_quote"],
    "gating_thresholds": ["marker", "source_quote"],
    "signatures":        ["name", "genes", "source_quote"],
    "study_aim":         ["statement", "source_quote"],
    "findings":          ["statement", "source_quote"],
}
LIST_FIELDS = list(ITEM_REQUIRED)
SCALAR_LIST_FIELDS = ["species", "tissue"]

# Field kinds determine which grounding verifier should handle each extracted item.
FIELD_KIND: dict[str, str] = {
    "diseases":          "entity",
    "perturbations":     "entity",
    "genes":             "entity",
    "cell_types":        "entity",
    "surface_markers":   "entity",
    "frequencies":       "number",
    "cohort":            "number",
    "gating_thresholds": "number",
    "signatures":        "entity",
    "study_aim":         "claim",
    "findings":          "claim",
}
NUMERIC_FIELDS = {f for f, kind in FIELD_KIND.items() if kind == "number"}
CLAIM_FIELDS = {f for f, kind in FIELD_KIND.items() if kind == "claim"}
SKIP_ITEM_KEYS = {"source_quote", "maps_to_gene"}


def empty_record(paper_id: str = "") -> dict:
    rec: dict = {"paper_id": paper_id, "relevance": "NOT_USEFUL"}
    for f in SCALAR_LIST_FIELDS + LIST_FIELDS:
        rec[f] = []
    return rec


def validate(rec: dict) -> list[str]:
    """Return a list of problems (empty list = valid). Dependency-free."""
    if not isinstance(rec, dict):
        return ["record is not an object"]
    errs: list[str] = []
    if rec.get("relevance") not in RELEVANCE:
        errs.append(f"relevance must be one of {RELEVANCE}, got {rec.get('relevance')!r}")
    for f in SCALAR_LIST_FIELDS:
        if not isinstance(rec.get(f, []), list):
            errs.append(f"{f} must be a list of strings")
    for f, required in ITEM_REQUIRED.items():
        items = rec.get(f, [])
        if not isinstance(items, list):
            errs.append(f"{f} must be a list"); continue
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                errs.append(f"{f}[{i}] is not an object"); continue
            for req in required:
                if req not in it:
                    errs.append(f"{f}[{i}] missing '{req}'")
    return errs


def _arr(props: dict, required: list[str]) -> dict:
    return {"type": "array", "items": {"type": "object", "additionalProperties": True,
                                       "properties": props, "required": required}}

_STR = {"type": "string"}
_NUM = {"type": ["number", "string"]}
_STRLIST = {"type": "array", "items": {"type": "string"}}

JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "paper_id": _STR,
        "relevance": {"type": "string", "enum": RELEVANCE},
        "species": _STRLIST,
        "tissue": _STRLIST,
        "diseases": _arr({"name": _STR, "source_quote": _STR}, ["name", "source_quote"]),
        "perturbations": _arr({"name": _STR, "type": _STR, "source_quote": _STR},
                              ["name", "source_quote"]),
        "genes": _arr({"symbol": _STR, "species": _STR, "source_quote": _STR},
                      ["symbol", "source_quote"]),
        "cell_types": _arr({"name": _STR, "source_quote": _STR}, ["name", "source_quote"]),
        "surface_markers": _arr({"marker": _STR, "maps_to_gene": _STR, "source_quote": _STR},
                                ["marker", "source_quote"]),
        "frequencies": _arr({"cell_type": _STR, "value": _NUM, "unit": _STR,
                             "of_population": _STR, "condition": _STR, "source_quote": _STR},
                            ["cell_type", "value", "source_quote"]),
        "cohort": _arr({"group": _STR, "n": _NUM, "unit": _STR, "source_quote": _STR},
                       ["group", "n", "source_quote"]),
        "gating_thresholds": _arr({"marker": _STR, "operator": _STR, "value": _NUM,
                                   "source_quote": _STR}, ["marker", "source_quote"]),
        "signatures": _arr({"name": _STR, "genes": _STRLIST, "species": _STR, "source_quote": _STR},
                           ["name", "genes", "source_quote"]),
        "study_aim": _arr({"statement": _STR, "source_quote": _STR}, ["statement", "source_quote"]),
        "findings": _arr({"statement": _STR, "source_quote": _STR}, ["statement", "source_quote"]),
    },
    "required": ["paper_id", "relevance"],
}
