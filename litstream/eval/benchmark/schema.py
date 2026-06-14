"""The common per-document shape plus text normalization.

A `Document` is one unit to score (an abstract or paragraph): its text and the
expert-marked gold answers as normalized strings, grouped by the canonical field
names in fields.py (only the fields a given dataset covers are filled).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    text: str
    gold: dict[str, set[str]] = field(default_factory=dict)


def normalize(text: str) -> str:
    """Lowercase, squeeze whitespace, strip edge punctuation. Makes 'CD8.' == 'cd8'.
    Shared by gold-loading, extraction, and scoring so all three compare like-for-like."""
    t = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    return t.strip(" .,:;()[]{}")
