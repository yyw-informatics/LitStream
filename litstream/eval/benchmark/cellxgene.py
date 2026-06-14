"""CZ CELLxGENE loader — an in-domain cell-type benchmark from real single-cell studies.

Reads the cached Curation-API crawl (fetch.ensure_cellxgene): one record per collection,
with the study text (title + description + dataset titles) and the set of Cell Ontology
cell types curated for that collection's datasets — modern single-cell papers with expert
CL-labeled cell types, close to MINE's real input.

Gold is keyed by CL ID and alias-expanded via the CL synonym layer, so 'natural killer
cell' also accepts MINE's 'NK cell'. Because the gold carries ontology IDs, there is no
over-annotation problem.

Caveat: the text is study-level and rarely names every cell type present in the data, so
RECALL here is a partial-recall measure (a recall floor); PRECISION is the cleaner signal
— "of the cell types MINE pulled, how many are really in the study."
"""

from __future__ import annotations

import json
from pathlib import Path

from .cell_ontology import aliases_by_id
from .fields import CELL_TYPES
from .schema import Document


def load(path) -> list[Document]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    docs: list[Document] = []
    for r in records:
        concepts = [frozenset(s) for cid, label in zip(r["cl_ids"], r["cl_labels"])
                    if (s := aliases_by_id(cid, label))]
        if concepts and (r.get("text") or "").strip():
            docs.append(Document(id=r["collection_id"], text=r["text"],
                                 gold={CELL_TYPES: concepts}))
    return docs
