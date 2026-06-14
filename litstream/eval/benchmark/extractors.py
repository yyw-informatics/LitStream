"""Pluggable fact-finders.

Each extractor takes one document's raw text and returns a record in the canonical
schema (record_schema.example.json / fields.py): genes/species/cell_types/frequencies/
gating_thresholds/surface_markers/signatures, each item carrying a source_quote (kept
for provenance checks; the scorer ignores it and matches on the key field).

  - "baseline": regex + dictionaries, no model, runs instantly.
  - "mine":     MINE's extraction as a single model call.
"""

from __future__ import annotations

import re

from .schema import normalize

# genes (BioRED)
_GENE_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*[0-9][A-Za-z0-9]*\b|\b[A-Z]{2,}[0-9]*\b")
_GENE_STOP = {normalize(w) for w in (
    "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSION", "CONCLUSIONS", "BACKGROUND",
    "DNA", "RNA", "MRNA", "CDNA", "PCR", "QT", "LQTS", "ATG", "GTG", "ECG", "EKG",
    "USA", "II", "III", "IV", "ID", "OR", "AND", "WT", "KO", "SD", "CI", "HR")}

# species (BioRED)
_SPECIES_LEX = ["human", "humans", "patient", "patients", "man", "woman", "men", "women",
                "homo sapiens", "mouse", "mice", "murine", "mus musculus", "rat", "rats",
                "rattus", "zebrafish", "drosophila", "yeast", "newborn", "proband",
                "child", "children", "infant", "fetal", "fetus"]

# cell types (JNLPBA) — bounded {0,40} repetition to avoid catastrophic backtracking
_CELLTYPE_LEX = ["lymphocytes", "lymphocyte", "monocytes", "monocyte", "macrophages",
                 "macrophage", "neutrophils", "neutrophil", "fibroblasts", "fibroblast",
                 "granulocytes", "eosinophils", "basophils", "thymocytes", "thymocyte",
                 "splenocytes", "leukocytes", "leukocyte", "erythrocytes", "platelets",
                 "megakaryocytes", "hepatocytes", "keratinocytes", "myocytes",
                 "neurons", "neuron", "osteoblasts", "osteoclasts", "pbmc", "pbmcs"]
_CELLTYPE_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9+\-]{0,40}\s+(?:cells?|cyte|cytes|blasts?|phils?|clasts?)\b",
    re.IGNORECASE)

# numbers / quantities (MeasEval)
_QTY_RE = re.compile(
    r"\d+(?:\.\d+)?\s?(?:%|°\s?[CF]|[A-Za-zµμ]+(?:\s?/\s?[A-Za-zµμ]+)?)?")


def _ctx(text: str, start: int, end: int, pad: int = 40) -> str:
    a, b = max(0, start - pad), min(len(text), end + pad)
    return re.sub(r"\s+", " ", text[a:b]).strip()


def _split_qty(s: str) -> tuple[str, str]:
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(.*)", s)
    return (m.group(1), m.group(2).strip()) if m else (s.strip(), "")


def baseline_extract(text: str) -> dict:
    genes: dict[str, dict] = {}
    for m in _GENE_RE.finditer(text):
        sym = m.group(0)
        if normalize(sym) in _GENE_STOP or len(normalize(sym)) < 3:
            continue
        genes.setdefault(normalize(sym), {
            "symbol": sym, "species": None, "source_quote": _ctx(text, m.start(), m.end())})

    species = sorted({w for w in _SPECIES_LEX
                      if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", text, re.I)})

    cells: dict[str, dict] = {}
    for w in _CELLTYPE_LEX:
        mm = re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", text, re.I)
        if mm:
            cells.setdefault(normalize(w),
                             {"name": w, "source_quote": _ctx(text, mm.start(), mm.end())})
    for m in _CELLTYPE_RE.finditer(text):
        name = m.group(0)
        cells.setdefault(normalize(name),
                         {"name": name, "source_quote": _ctx(text, m.start(), m.end())})

    freqs: dict[str, dict] = {}
    for m in _QTY_RE.finditer(text):
        val, unit = _split_qty(m.group(0))
        freqs.setdefault(normalize(m.group(0)), {
            "cell_type": None, "value": val, "unit": unit or None, "of_population": None,
            "condition": None, "source_quote": _ctx(text, m.start(), m.end())})

    return {
        "relevance": None, "species": species, "tissue": [],
        "genes": list(genes.values()),
        "cell_types": list(cells.values()),
        "surface_markers": [],                      # no public benchmark covers this field
        "frequencies": list(freqs.values()),
        "gating_thresholds": [],                    # baseline emits none on generic text
        "signatures": [],                           # no public benchmark covers this field
    }


import json

_MINE_SYSTEM = (
    "You are an information-extraction engine for biomedical text (single-cell / "
    "immunology / CITE-seq). From the passage, extract EVERY concrete entity of the "
    "listed kinds that is EXPLICITLY stated. This is extract-everything mode: do not "
    "filter for relevance or topic — pull all items present. Be specific: exact gene "
    "symbols as written; exact numbers with their unit and the population they describe; "
    "gating thresholds with their operator. Never invent items not in the text. Give a "
    "short verbatim source_quote (copied from the passage) for every item. Respond with "
    "ONE JSON object only — no prose, no markdown fences."
)
_MINE_SCHEMA = (
    'JSON shape (use these exact keys; [] if none):\n'
    '{"species":["human"],"tissue":["PBMC"],'
    '"genes":[{"symbol":"FOXP3","species":"human","source_quote":"..."}],'
    '"cell_types":[{"name":"regulatory T cell","source_quote":"..."}],'
    '"surface_markers":[{"marker":"CD25","maps_to_gene":"IL2RA","source_quote":"..."}],'
    '"frequencies":[{"cell_type":"Treg","value":5.2,"unit":"%",'
    '"of_population":"CD4 T cells","condition":"healthy","source_quote":"..."}],'
    '"gating_thresholds":[{"marker":"CD25","operator":">","value":"high","source_quote":"..."}],'
    '"signatures":[{"name":"Treg signature","genes":["FOXP3","IL2RA"],'
    '"species":"human","source_quote":"..."}]}'
)

_MODEL_CACHE: dict = {}


def _get_model(name: str):
    """Build (once) a project TaskModel by name from litstream/config/task_models.yaml."""
    if name not in _MODEL_CACHE:
        import os
        from pathlib import Path

        import yaml

        from litstream.config.env import load_env
        from litstream.tasks.models import build_model
        load_env()
        cfg = Path(__file__).resolve().parents[2] / "config" / "task_models.yaml"
        specs = yaml.safe_load(cfg.read_text())["task_models"]
        spec = next(s for s in specs if s["name"] == name)
        _MODEL_CACHE[name] = build_model(spec, os.environ.get)
    return _MODEL_CACHE[name]


def _loads_loose(raw: str):
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        cut = raw.find("{")
        raw = raw[cut:] if cut >= 0 else raw
    try:
        return json.loads(raw)
    except Exception:
        i, j = raw.find("{"), raw.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(raw[i:j + 1])
            except Exception:
                return None
    return None


def _to_record(obj: dict | None) -> dict:
    """Coerce a model's JSON into the canonical record, with safe defaults."""
    obj = obj if isinstance(obj, dict) else {}

    def dicts(k):
        return [x for x in (obj.get(k) or []) if isinstance(x, dict)]

    def strs(k):
        out = []
        for x in (obj.get(k) or []):
            s = x if isinstance(x, str) else (x.get("name") or x.get("symbol") or "")
            if s:
                out.append(str(s))
        return out

    return {
        "relevance": obj.get("relevance"),
        "species": strs("species"), "tissue": strs("tissue"),
        "genes": dicts("genes"), "cell_types": dicts("cell_types"),
        "surface_markers": dicts("surface_markers"),
        "frequencies": dicts("frequencies"),
        "gating_thresholds": dicts("gating_thresholds"),
        "signatures": dicts("signatures"),
    }


def mine_extract(text: str, model: str | None = None) -> dict:
    """MINE's fact-finding as one model call. `model` is a name in task_models.yaml
    (default 'deepseek') — swap it to compare models on cost vs quality."""
    m = _get_model(model or "deepseek")
    prompt = f"{_MINE_SCHEMA}\n\nPassage:\n\"\"\"\n{text}\n\"\"\"\n\nJSON:"
    res = m.complete(prompt, system=_MINE_SYSTEM, max_tokens=2000, temperature=0.0)
    return _to_record(_loads_loose(res.text))


EXTRACTORS = {"baseline": baseline_extract, "mine": mine_extract}
