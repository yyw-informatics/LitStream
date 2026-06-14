"""Extract plain text from a PDF so non-Claude providers can read papers.

Claude's Read tool ingests PDFs natively; other providers need text. For a fair
cross-provider benchmark we feed every model the same extracted text.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text(pdf_path: str | Path, max_chars: int = 45_000) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    total = 0
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n".join(parts)[:max_chars]
