"""Cell Ontology (CL) exact-synonym lookup for cell-type matching.

Curated, deterministic synonym source. Parses cl-simple.obo once into a frozen,
auditable JSON map (concept_id -> accepted spellings), then expands a cell-type string
into its CL concept's full spelling set so e.g. MINE's "NK cell" matches gold "natural
killer cell" and "Treg" matches "regulatory T cell".

Only `EXACT` synonyms (plus the term name) are used — CL's BROAD/NARROW/RELATED synonyms
are deliberately ignored, so distinct cell types are never conflated. Spellings that map to
more than one concept are dropped (ambiguity guard).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import normalize

_DATA = Path(__file__).resolve().parent / "data" / "cl"
_OBO = _DATA / "cl-simple.obo"
_FROZEN = _DATA / "cl_synonyms.json"   # committed artifact: concept_id -> [spellings]

_SYN_RE = re.compile(r'"([^"]+)"\s+EXACT')


def _parse_obo(path: Path) -> dict[str, list[str]]:
    concepts: dict[str, list[str]] = {}
    cid: str | None = None
    spellings: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "[Term]" or (line.startswith("[") and line.endswith("]")):
            if cid and spellings:
                concepts[cid] = sorted(spellings)
            cid, spellings = None, set()
        elif line.startswith("id: CL:"):
            cid = line[4:].strip()
        elif line.startswith("name:"):
            spellings.add(normalize(line[5:]))
        elif line.startswith("synonym:"):
            m = _SYN_RE.search(line)            # EXACT synonyms only
            if m:
                spellings.add(normalize(m.group(1)))
    if cid and spellings:
        concepts[cid] = sorted(spellings)
    return concepts


def _build_frozen() -> dict[str, list[str]]:
    concepts = _parse_obo(_OBO)
    _FROZEN.write_text(json.dumps(concepts, indent=0, sort_keys=True))
    return concepts


_CONCEPTS: dict[str, list[str]] | None = None
_SPELL2ID: dict[str, str] | None = None


def _index() -> tuple[dict, dict]:
    global _CONCEPTS, _SPELL2ID
    if _CONCEPTS is None:
        _CONCEPTS = json.loads(_FROZEN.read_text()) if _FROZEN.exists() else _build_frozen()
        spell2ids: dict[str, set[str]] = {}
        for cid, sps in _CONCEPTS.items():
            for s in sps:
                spell2ids.setdefault(s, set()).add(cid)
        # ambiguity guard: a spelling shared by >1 concept is dropped (can't disambiguate)
        _SPELL2ID = {s: next(iter(ids)) for s, ids in spell2ids.items() if len(ids) == 1}
    return _CONCEPTS, _SPELL2ID


def _candidates(s: str):
    """Surface + light morphological variants so plurals resolve ('T cells' -> 'T cell')."""
    seen = []
    for c in (s, re.sub(r"\bcells\b", "cell", s), s[:-1] if s.endswith("s") else s):
        if c and c not in seen:
            seen.append(c)
            yield c


def concept_aliases(surface: str) -> tuple[str, set[str]]:
    """(concept_key, accepted spellings) for a cell-type string. If it isn't a known CL
    term, returns a singleton keyed by the surface so it stays its own answer."""
    concepts, spell2id = _index()
    s = normalize(surface)
    for cand in _candidates(s):
        cid = spell2id.get(cand)
        if cid:
            return cid, set(concepts[cid]) | {s}
    return f"~{s}", {s}


def aliases_by_id(cl_id: str, label: str = "") -> set[str]:
    """Accepted spellings for a CL concept given its ID (e.g. 'CL:0000636') + label.
    Used when the gold already carries the ontology ID (cellxgene)."""
    concepts, _ = _index()
    out = set(concepts.get(cl_id, []))
    if label:
        out.add(normalize(label))
    return out


if __name__ == "__main__":   # rebuild the frozen map and spot-check a few lookups
    c = _build_frozen()
    print(f"parsed {len(c)} CL concepts -> {_FROZEN}")
    for q in ("NK cells", "Treg", "regulatory T cell", "dendritic cell", "macrophage"):
        k, al = concept_aliases(q)
        print(f"  {q!r:22} -> {k}: {sorted(al)}")
