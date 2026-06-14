"""Find-then-verify grounding (RAG) for the evidence converter.

The converter (`structure_evidence.py`) makes the model emit a `source_quote` per fact, but models
invent or paraphrase quotes, and exact-substring matching over messy PDF text over-flags. For each
extracted fact, this retrieves the most-relevant passage from the paper, then verifies that the fact
is actually supported by it. This tolerates PDF noise and paraphrase while still
catching made-up facts, especially invented numbers.

  retrieve: LangChain text chunks -> embeddings -> InMemoryVectorStore -> top-k passages.
  verify:   OverlapVerifier (cheap) requires the claim's words AND every number to be in the
            retrieved passage. MiniCheckVerifier is the entailment-model alternative.

Both the embeddings and the verifier are pluggable.

    python -m litstream_evidence.ground_retrieval --project lg_smoke \
        --project-dir /tmp/lg_smoke_proj --embeddings hf --verifier overlap
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Callable, Protocol

from langchain_core.embeddings import Embeddings

from litstream_evidence.pdf_text import extract_text
from litstream_evidence.evidence_schema import CLAIM_FIELDS, LIST_FIELDS, NUMERIC_FIELDS, SKIP_ITEM_KEYS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

_STOP = {"the", "and", "for", "with", "were", "was", "are", "this", "that", "from",
         "into", "which", "not", "have", "has", "used", "using", "our", "your", "per"}
_SKIP_FIELDS = SKIP_ITEM_KEYS
# The schema declares which fields carry numeric values or propositions.
VALUE_FIELDS = NUMERIC_FIELDS


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def _content_words(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOP]


def _numbers(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])", text or "")


def _is_numeric_claim(field: str, claim: str) -> bool:
    """A numeric-field item whose claim actually carries a standalone number. A 'CD25 > 500' threshold
    qualifies; a categorical 'CD3 positive'/'CD4 high' does not (it's a mention)."""
    return field in VALUE_FIELDS and bool(_numbers(claim))


def _needs_strict(field: str, claim: str) -> bool:
    """Route to the strict entailment verifier (MiniCheck) rather than the presence verifier when the
    item is a claim-kind proposition (ALWAYS — a proposition needs entailment, not a mention) OR a
    numeric-field item that carries a standalone number. Categorical numeric items and entity
    mentions fall through to presence."""
    return field in CLAIM_FIELDS or _is_numeric_claim(field, claim)


def _item_text(item: dict) -> str:
    """The claim to verify: the item's values, not the extracted quote."""
    parts: list[str] = []
    for k, v in item.items():
        if k in _SKIP_FIELDS:
            continue
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v is not None and str(v):
            parts.append(str(v))
    return " ".join(parts)


def _chunk(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    text = text or ""
    if len(text) <= size:
        return [text] if text.strip() else []
    out, i, step = [], 0, max(1, size - overlap)
    while i < len(text):
        out.append(text[i:i + size])
        i += step
    return [c for c in out if c.strip()]


class Verifier(Protocol):
    def verify(self, claim: str, passage: str) -> tuple[bool, float]: ...


class OverlapVerifier:
    """Cheap, no-model check: supported if enough of the claim's content words appear in the
    retrieved passage AND every number in the claim appears in it. The number requirement does the
    real work — it catches invented frequencies and thresholds, a known weak spot for generalist
    checkers."""

    def __init__(self, min_overlap: float = 0.5):
        self.min_overlap = min_overlap

    def verify(self, claim: str, passage: str) -> tuple[bool, float]:
        cw = set(_content_words(claim))
        if not cw:
            return (False, 0.0)
        overlap = len(cw & set(_content_words(passage))) / len(cw)
        plow = (passage or "").lower()
        numbers_ok = all(n in plow for n in _numbers(claim))
        return (overlap >= self.min_overlap and numbers_ok, round(overlap, 3))


class MiniCheckVerifier:
    """Entailment fact-checker (MiniCheck) — judges 'does the passage support the claim?'. Paired
    with the number-check: a claim whose quantities aren't in the passage is rejected outright
    (numbers are MiniCheck's known weak spot), otherwise MiniCheck decides. Pass
    `predict(claim, passage) -> bool`; the default (`make_verifier('minicheck')`) loads MiniCheck
    lazily on first use."""

    def __init__(self, predict: Callable[[str, str], bool] | None = None):
        self.predict = predict

    def verify(self, claim: str, passage: str) -> tuple[bool, float]:
        if self.predict is None:
            raise NotImplementedError(
                "MiniCheckVerifier needs a predict(claim, passage)->bool callable.")
        if not all(n in (passage or "").lower() for n in _numbers(claim)):
            return (False, 0.0)
        ok = bool(self.predict(claim, passage))
        return (ok, 1.0 if ok else 0.0)


def _lazy_minicheck(model_name: str = "flan-t5-large") -> Callable[[str, str], bool]:
    """A predict(claim, passage)->bool that lazy-loads the MiniCheck model on first call (cached)."""
    state: dict = {}

    def predict(claim: str, passage: str) -> bool:
        scorer = state.get("scorer")
        if scorer is None:
            from minicheck.minicheck import MiniCheck
            scorer = state["scorer"] = MiniCheck(model_name=model_name)
        labels, _, _, _ = scorer.score(docs=[passage], claims=[claim])
        return bool(labels[0])

    return predict


def make_verifier(name: str = "overlap", model: str = "flan-t5-large") -> Verifier:
    if name == "overlap":
        return OverlapVerifier()
    if name == "minicheck":
        return MiniCheckVerifier(_lazy_minicheck(model))
    raise ValueError(f"unknown verifier {name!r} (use 'overlap' or 'minicheck')")


class _FakeEmbeddings(Embeddings):
    """Deterministic, numpy-free embeddings (sha256 -> vector)."""

    def __init__(self, size: int = 16):
        self.size = size

    def _vec(self, text: str) -> list[float]:
        import hashlib
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        raw = (digest * (self.size // len(digest) + 1))[: self.size]
        return [b / 255.0 for b in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_embeddings(name: str = "hf"):
    if name == "fake":
        return _FakeEmbeddings()
    if name == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"unknown embeddings {name!r} (use 'fake' or 'hf')")


def build_retriever(source_text: str, embeddings, k: int = 3,
                    size: int = 800, overlap: int = 120) -> Callable[[str], list[str]]:
    """Index the paper's passages in a LangChain InMemoryVectorStore; return retrieve(query)->passages."""
    from langchain_core.vectorstores import InMemoryVectorStore
    chunks = _chunk(source_text, size, overlap)
    if not chunks:
        return lambda q: []
    store = InMemoryVectorStore.from_texts(chunks, embeddings)
    kk = min(k, len(chunks))
    return lambda q: [d.page_content for d in store.similarity_search(q, k=kk)]


def _best_passage(claim: str, passages: list[str]) -> str:
    if not passages:
        return ""
    cw = set(_content_words(claim))
    best = max(passages, key=lambda p: len(cw & set(_content_words(p))))
    return re.sub(r"\s+", " ", best).strip()[:240]


def reground_record(record: dict, retrieve: Callable[[str], list[str]], verifier: Verifier,
                    value_verifier: Verifier | None = None) -> tuple[dict, dict]:
    """For each item: retrieve passages, verify support; on success replace `source_quote` with the
    real supporting passage, else flag it. Propositions and numeric items with explicit numbers go to
    `value_verifier`; entity mentions and categorical thresholds go to `verifier`. `by_field` reports
    grounded/flagged counts plus `value_checked`. Returns the updated record and report."""
    value_verifier = value_verifier or verifier
    report: dict = {"grounded": 0, "flagged": 0, "by_field": {}, "flagged_items": []}
    for field in LIST_FIELDS:
        ok = bad = nstrict = 0
        for item in record.get(field, []):
            claim = _item_text(item)
            strict = _needs_strict(field, claim)
            nstrict += strict
            v = value_verifier if strict else verifier
            passages = retrieve(claim)
            supported, _ = v.verify(claim, "  ...  ".join(passages))
            if supported:
                item["source_quote"] = _best_passage(claim, passages)
                ok += 1
            else:
                bad += 1
                report["flagged_items"].append({"field": field, "item": item})
        report["by_field"][field] = {"grounded": ok, "flagged": bad, "value_checked": nstrict}
        report["grounded"] += ok
        report["flagged"] += bad
    return record, report


def reground_from_text(record: dict, source_text: str, embeddings, verifier: Verifier,
                       value_verifier: Verifier | None = None, k: int = 3) -> tuple[dict, dict]:
    return reground_record(record, build_retriever(source_text, embeddings, k), verifier, value_verifier)


def run(project: str, project_dir: Path, embeddings, verifier: Verifier,
        value_verifier: Verifier | None = None, k: int = 3) -> list[dict]:
    lit = project_dir / f"projects/{project}/literature"
    papers = project_dir / f"projects/{project}/papers"
    json_files = sorted(lit.glob("*_evidence.json")) if lit.is_dir() else []
    rows: list[dict] = []
    for jf in json_files:
        stem = jf.name[: -len("_evidence.json")]
        pdf = papers / f"{stem}.pdf"
        if not pdf.exists():
            log(f"{stem}: no source PDF — skipping"); continue
        rec = json.loads(jf.read_text())
        rec, report = reground_from_text(rec, extract_text(pdf), embeddings, verifier, value_verifier, k)
        (lit / f"{stem}_evidence.regrounded.json").write_text(json.dumps(rec, indent=2))
        rows.append({"paper": stem, "grounded": report["grounded"], "flagged": report["flagged"]})
        log(f"{stem}: {report['grounded']} grounded / {report['flagged']} flagged "
            f"→ {stem}_evidence.regrounded.json")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-ground evidence.json items by retrieve-then-verify")
    ap.add_argument("--project", required=True)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--embeddings", default="hf", choices=["hf", "fake"])
    ap.add_argument("--entity-verifier", default="overlap", choices=["overlap", "minicheck"],
                    help="verifier for entity items (genes/cell_types/markers)")
    ap.add_argument("--value-verifier", default="minicheck", choices=["overlap", "minicheck"],
                    help="verifier for numeric claims (frequencies/gating_thresholds)")
    ap.add_argument("--minicheck-model", default="flan-t5-large",
                    help="MiniCheck model (e.g. flan-t5-large, roberta-large)")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    rows = run(args.project, Path(args.project_dir).resolve(), make_embeddings(args.embeddings),
               make_verifier(args.entity_verifier, args.minicheck_model),
               make_verifier(args.value_verifier, args.minicheck_model), args.k)
    if not rows:
        log("no *_evidence.json found — run structure_evidence first"); return
    g = sum(r["grounded"] for r in rows); f = sum(r["flagged"] for r in rows)
    print(f"\n  re-grounded {len(rows)} paper(s): {g} grounded, {f} flagged "
          f"(entities={args.entity_verifier}, values={args.value_verifier}). "
          f"Regrounded JSON written next to each *_evidence.json.\n")


if __name__ == "__main__":
    main()
