"""Load real abstract-screening data from CSMeD-FT for the triage evaluator.

CSMeD (NeurIPS 2023) consolidates citation-screening datasets for systematic
reviews. Its CSMeD-FT subset ships as a bundled CSV (no credentials, no download
beyond the repo) with real Cochrane review documents: title, abstract, and an
expert include/exclude decision, grouped by review. Each review's eligibility
criteria become the triage CONTEXT; its documents become labeled items.

We map: included -> RELEVANT (keep), excluded -> NOT_RELEVANT (drop).

Caveat: CSMeD-FT decisions are full-text-stage screening decisions used here as a
proxy for abstract relevance — close enough to benchmark the triage mechanism and
get model-routing metrics, which is the goal. It is NOT single-cell/CITE-seq, so
it tests the classifier, not topic fit.

Source repo (cloned under reference/csmed): WojciechKusa/systematic-review-datasets
"""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

csv.field_size_limit(sys.maxsize)  # CSMeD rows carry full text

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = ROOT / "reference" / "csmed" / "data" / "CSMeD" / "CSMeD-FT.zip"


@dataclass
class ReviewGroup:
    review_id: str
    context: str
    items: list[dict]  # {id, title, abstract, gold}


def _build_context(meta: dict, max_criteria_chars: int = 1500) -> str:
    crit = (meta.get("criteria_text") or meta.get("criteria") or "").strip()
    if len(crit) > max_criteria_chars:
        crit = crit[:max_criteria_chars] + " …"
    return (
        "This is a systematic-review citation-screening task. Decide whether each "
        "paper meets the review's eligibility criteria.\n\n"
        f"REVIEW: {meta.get('title','').strip()}\n"
        f"TYPE: {meta.get('review_type','').strip()}\n\n"
        f"ELIGIBILITY CRITERIA:\n{crit}"
    )


def load_reviews(*, zip_path: Path = DEFAULT_ZIP, split: str = "dev",
                 review_ids: list[str] | None = None, min_docs: int = 15,
                 max_docs_per_review: int | None = None,
                 require_abstract: bool = True) -> list[ReviewGroup]:
    """Read CSMeD-FT from the bundled zip and return per-review labeled groups."""
    if not zip_path.exists():
        raise FileNotFoundError(
            f"CSMeD-FT zip not found at {zip_path}. Clone the repo first:\n"
            "  git clone --depth 1 https://github.com/WojciechKusa/systematic-review-datasets.git "
            f"{ROOT/'reference'/'csmed'}")

    csv_name = f"CSMeD-FT/CSMeD-FT-{split}.csv"
    meta_name = f"CSMeD-FT/CSMeD-FT-{split}_reviews_metadata.json"
    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_name) as fh:
            rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
        with z.open(meta_name) as fh:
            meta = json.load(fh)

    # group rows by review
    by_review: dict[str, list[dict]] = {}
    for r in rows:
        if require_abstract and not (r.get("abstract") or "").strip():
            continue
        by_review.setdefault(r["review_id"], []).append(r)

    wanted = review_ids or [rid for rid, docs in by_review.items() if len(docs) >= min_docs]
    groups: list[ReviewGroup] = []
    for rid in wanted:
        docs = by_review.get(rid, [])
        if not docs or rid not in meta:
            continue
        if max_docs_per_review:
            docs = docs[:max_docs_per_review]
        items = [{
            "id": d["document_id"],
            "title": (d.get("title") or "").strip(),
            "abstract": (d.get("abstract") or "").strip(),
            "gold": "RELEVANT" if d["decision"] == "included" else "NOT_RELEVANT",
        } for d in docs]
        groups.append(ReviewGroup(rid, _build_context(meta[rid]), items))
    return groups


if __name__ == "__main__":  # quick inspection
    gs = load_reviews(review_ids=["CD006612", "CD005563"])
    for g in gs:
        n_rel = sum(i["gold"] == "RELEVANT" for i in g.items)
        print(f"{g.review_id}: {len(g.items)} items ({n_rel} relevant), "
              f"context {len(g.context)} chars")
