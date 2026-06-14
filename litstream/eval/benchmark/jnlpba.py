"""JNLPBA loader — IOB token-tagged format -> Documents with gold cell types.

Format (one token per line):
    -DOCSTART-<tab>O          marks the start of a new abstract
    token<tab>B-cell_type     first token of a cell-type mention
    token<tab>I-cell_type     continuation token
    token<tab>O               outside any entity
    (blank line ends a sentence)

We split on -DOCSTART- into abstracts (404 in the test set), join tokens back into
text, and stitch B-/I- runs into whole cell-type mentions for the gold set. Other
entity types (protein, DNA, etc.) are parsed too but only cell_type is scored.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .cell_ontology import concept_aliases
from .fields import CELL_TYPES
from .schema import Document, normalize


def _to_concepts(surfaces: set[str]) -> list[frozenset[str]]:
    """Group cell-type surfaces by CL concept and expand each into its accepted spellings,
    so an answer written as 'natural killer cells' also accepts MINE's 'NK cell'. Surfaces
    of the same concept merge; unknown surfaces stay singletons."""
    by_concept: dict[str, set[str]] = {}
    for s in surfaces:
        key, spellings = concept_aliases(s)
        by_concept.setdefault(key, set()).update(spellings)
    return [frozenset(v) for v in by_concept.values()]

# JNLPBA tag suffix -> our field. MINE's `cell_types` field captures any named cell
# population, including immortalized cell lines (HL60, U937, HepG2, …). JNLPBA tags
# those separately as `cell_line`; we fold them into the cell_types gold so MINE's
# correct cell-line extractions count as hits instead of being dropped.
_SUFFIX_TO_SLOT = {"cell_type": CELL_TYPES, "cell_line": CELL_TYPES}


def _flush_entity(tokens: list[str], suffix: str, gold: dict[str, set[str]]) -> None:
    slot = _SUFFIX_TO_SLOT.get(suffix)
    if slot and tokens:
        gold[slot].add(normalize(" ".join(tokens)))


def load(path: Path) -> list[Document]:
    docs: list[Document] = []
    words: list[str] = []
    gold: dict[str, set[str]] = defaultdict(set)
    ent_tokens: list[str] = []
    ent_suffix = ""
    doc_n = 0

    def finish() -> None:
        nonlocal words, gold, ent_tokens, ent_suffix
        _flush_entity(ent_tokens, ent_suffix, gold)
        ent_tokens, ent_suffix = [], ""
        if words:
            docs.append(Document(id=f"jnlpba_{doc_n:04d}", text=" ".join(words),
                                 gold={k: _to_concepts(v) for k, v in gold.items()}))
        words = []
        gold = defaultdict(set)

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if line.startswith("-DOCSTART-"):
            finish(); doc_n += 1
            continue
        if not line.strip():
            _flush_entity(ent_tokens, ent_suffix, gold)  # entity can't cross sentences
            ent_tokens, ent_suffix = [], ""
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        token, tag = parts[0], parts[1]
        words.append(token)
        if tag.startswith("B-"):
            _flush_entity(ent_tokens, ent_suffix, gold)
            ent_tokens, ent_suffix = [token], tag[2:]
        elif tag.startswith("I-") and ent_suffix == tag[2:]:
            ent_tokens.append(token)
        else:  # O, or an I- mismatch -> close any open entity
            _flush_entity(ent_tokens, ent_suffix, gold)
            ent_tokens, ent_suffix = [], ""
    finish()
    return [d for d in docs if d.text.strip()]
