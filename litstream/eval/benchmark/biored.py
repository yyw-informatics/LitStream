"""BioRED loader — PubTator format -> Documents with gold genes + species.

Each abstract block is:
    PMID|t|title
    PMID|a|abstract
    PMID <tab> start <tab> end <tab> surface <tab> TYPE <tab> id   (entity lines)
    PMID <tab> RELATION <tab> ... (relation lines — skipped: col[1] is a word, not an int)
    (blank line ends the block)
We keep GeneOrGeneProduct -> gene and OrganismTaxon -> species.

ALIAS GROUPING: BioRED records every surface form of one entity as a separate row, but
rows for the same entity share a concept ID (col 6) — e.g. 'ADCY5' and 'adenylate cyclase
5' both carry the same gene ID; 'patient'/'human'/'men' all carry the human taxon ID. Rows
are grouped by that ID into one gold concept holding all its spellings, so a prediction
that matches any spelling scores the concept once (no double-counting, no symbol-vs-name
miss). Unlinked rows ('-'/empty id) stay singleton concepts.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .fields import GENES, SPECIES
from .schema import Document, normalize

_TYPE_TO_SLOT = {"GeneOrGeneProduct": GENES, "OrganismTaxon": SPECIES}

# NCBI taxonomy ID -> canonical species name. BioRED tags organism MENTIONS (incl.
# 'patient'/'men') with a taxon ID but the literal species word may never appear in the
# text; add the canonical name as an alias so MINE's 'human'/'mouse' can match the concept.
_TAXON_NAME = {
    "9606": "human", "10090": "mouse", "10116": "rat", "10029": "chinese hamster",
    "4932": "yeast", "7227": "drosophila", "7955": "zebrafish", "9598": "chimpanzee",
    "9544": "macaque", "9913": "bovine", "9823": "pig", "9031": "chicken", "8355": "xenopus",
}


def _finalize(groups: dict) -> dict:
    """slot -> {concept_id -> {surfaces}}  ==>  slot -> [frozenset(spellings), ...].
    For species, add the taxon's canonical name so a concept seen only as 'patients'
    still matches a prediction of 'human'."""
    out: dict = {}
    for slot, by_id in groups.items():
        concepts = []
        for cid, surfaces in by_id.items():
            spellings = set(surfaces)
            if slot == SPECIES and cid in _TAXON_NAME:
                spellings.add(normalize(_TAXON_NAME[cid]))
            concepts.append(frozenset(spellings))
        out[slot] = concepts
    return out


def load(path: Path) -> list[Document]:
    docs: list[Document] = []
    cur: Document | None = None
    groups: dict = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            if cur:
                cur.gold = _finalize(groups)
                docs.append(cur)
            cur, groups = None, {}
            continue
        if "|t|" in line:
            pmid, _, title = line.split("|", 2)
            cur = Document(id=pmid, text=title)
            groups = defaultdict(lambda: defaultdict(set))
        elif "|a|" in line and cur:
            cur.text += " " + line.split("|", 2)[2]
        elif "\t" in line and cur:
            cols = line.split("\t")
            if len(cols) >= 6 and cols[1].isdigit():   # entity line (not a relation)
                slot = _TYPE_TO_SLOT.get(cols[4])
                surf = normalize(cols[3])
                if slot and surf:
                    cid = cols[5].strip()
                    key = cid if cid and cid != "-" else f"~{surf}"  # unlinked -> singleton
                    groups[slot][key].add(surf)
    if cur:
        cur.gold = _finalize(groups)
        docs.append(cur)
    return docs
