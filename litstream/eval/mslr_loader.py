"""Load MSLR-2022 multi-document medical-study summarization data for synthesis eval.

MSLR (the MSLR-2022 shared task; DeYoung/Wallace et al.) groups many input studies
(Title + Abstract) under one ReviewID, each mapping to one expert-written Target
review summary: many abstracts in, one cited review out.

Mirrors csmed_loader's ReviewGroup shape, but MSLR is summarization, not triage:
each group carries the gold `target` summary, not per-item include/exclude labels.
Only reviews present in both the inputs and targets files are returned.

    python -m litstream.eval.mslr_loader [split]      # inspect a split
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

csv.field_size_limit(sys.maxsize)  # MSLR abstracts are long

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "litstream" / "eval" / "benchmark" / "data" / "mslr"


@dataclass
class ReviewGroup:
    review_id: str
    target: str                                       # gold expert review summary
    inputs: list[dict] = field(default_factory=list)  # [{pmid, title, abstract}]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"MSLR data not found at {path}. Fetch it first:\n"
            "  python -m litstream.eval.benchmark.fetch   # pulls cochrane\n"
            "or call litstream.eval.benchmark.fetch.ensure_mslr('cochrane'|'ms2').")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_reviews(*, subset: str = "cochrane", split: str = "dev",
                 review_ids: list[str] | None = None, min_docs: int = 1,
                 max_docs_per_review: int | None = None,
                 require_abstract: bool = True) -> list[ReviewGroup]:
    """Read MSLR <subset>/<split> and return per-review groups joined on ReviewID.

    min_docs drops reviews with fewer than N studies (set >1 to require
    multi-document reviews; cochrane/dev has 51 singletons). require_abstract skips
    title-only input rows (304 in cochrane/dev). max_docs_per_review caps the studies
    fed per review to bound cost (largest review has 145 studies).
    """
    base = DATA / subset
    inputs = _read_csv(base / f"{split}-inputs.csv")
    targets = _read_csv(base / f"{split}-targets.csv")

    by_review: dict[str, list[dict]] = defaultdict(list)
    for r in inputs:
        if require_abstract and not (r.get("Abstract") or "").strip():
            continue
        by_review[r["ReviewID"]].append(r)

    target_of = {r["ReviewID"]: (r.get("Target") or "").strip() for r in targets}

    wanted = review_ids if review_ids is not None else list(target_of)
    groups: list[ReviewGroup] = []
    for rid in wanted:
        tgt = target_of.get(rid)
        docs = by_review.get(rid, [])
        if not tgt or len(docs) < min_docs:
            continue
        if max_docs_per_review:
            docs = docs[:max_docs_per_review]
        items = [{"pmid": d.get("PMID", ""), "title": (d.get("Title") or "").strip(),
                  "abstract": (d.get("Abstract") or "").strip()} for d in docs]
        groups.append(ReviewGroup(rid, tgt, items))
    return groups


if __name__ == "__main__":  # quick inspection
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    gs = load_reviews(split=split, min_docs=2)
    if not gs:
        print(f"no reviews for split={split}")
        sys.exit(0)
    n_studies = sum(len(g.inputs) for g in gs)
    tgt_len = sorted(len(g.target) for g in gs)[len(gs) // 2]
    print(f"MSLR cochrane/{split}: {len(gs)} multi-doc reviews (min_docs=2), "
          f"{n_studies} studies, median target {tgt_len} chars")
    for g in gs[:3]:
        print(f"  {g.review_id}: {len(g.inputs)} studies -> target {len(g.target)} chars")
