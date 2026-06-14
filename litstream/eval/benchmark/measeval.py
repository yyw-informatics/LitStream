"""MeasEval loader — paired text/ + tsv/ files -> Documents with gold numbers.

Each paragraph is a .txt file; its annotations are a .tsv with columns:
    docId  annotSet  annotType  startOffset  endOffset  annotId  text  other
We take the `text` of every row whose annotType is "Quantity" as the gold number set
(e.g. "2617.4 m", "2619.6 and 2614.7 m"), scoring the quantity-finding part of MeasEval
(not the measured-entity/property links).

Scored with the token-overlap matcher (score.number_*), not exact-string.
"""

from __future__ import annotations

from pathlib import Path

from .fields import FREQUENCIES
from .schema import Document, normalize


def load(root: Path, split: str = "eval") -> list[Document]:
    root = Path(root)
    tsv_dir, txt_dir = root / split / "tsv", root / split / "text"
    docs: list[Document] = []
    for tsv in sorted(tsv_dir.glob("*.tsv")):
        doc_id = tsv.stem
        txt = txt_dir / f"{doc_id}.txt"
        if not txt.exists():
            continue
        quantities: set[str] = set()
        for i, raw in enumerate(tsv.read_text(encoding="utf-8").splitlines()):
            if i == 0 or not raw.strip():  # header / blank
                continue
            cols = raw.split("\t")
            if len(cols) >= 7 and cols[2] == "Quantity":
                quantities.add(normalize(cols[6]))
        docs.append(Document(id=doc_id, text=txt.read_text(encoding="utf-8"),
                             gold={FREQUENCIES: quantities}))
    return docs
