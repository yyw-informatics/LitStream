"""Benchmark triage models on real CSMeD-FT abstract-screening data.

    mamba run -n litstream python -m litstream.eval.csmed_eval

Defaults to two balanced Cochrane reviews (~62 real abstracts, expert labels) so
the run stays cheap. Each backend is scored on KEEP precision/recall + 3-class
accuracy, with cost recorded in the ledger. Override the reviews/split via args.
"""

from __future__ import annotations

import sys

from litstream.config.env import load_env
from litstream.ledger.cost import CostLedger
from litstream.eval.csmed_loader import load_reviews, ROOT
from litstream.eval.triage_eval import evaluate_groups, _print

DEFAULT_REVIEWS = ["CD006612", "CD005563"]  # balanced: 17/17 and 15/13


if __name__ == "__main__":
    load_env()
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    reviews = sys.argv[2].split(",") if len(sys.argv) > 2 else DEFAULT_REVIEWS

    groups_obj = load_reviews(split=split, review_ids=reviews)
    groups = [(g.context, g.items) for g in groups_obj]
    n_items = sum(len(g.items) for g in groups_obj)
    n_rel = sum(i["gold"] == "RELEVANT" for g in groups_obj for i in g.items)
    print(f"\n  CSMeD-FT benchmark · split={split} · reviews={','.join(reviews)}")
    print(f"  {len(groups_obj)} reviews, {n_items} abstracts ({n_rel} relevant / "
          f"{n_items - n_rel} not) — real Cochrane expert labels")

    config = ROOT / "litstream" / "config" / "task_models.yaml"
    db = ROOT / "litstream.db"
    ledger = CostLedger(str(db))
    ledger.set_policy(cap_usd=50.0)
    run_id = ledger.start_run(project="csmed_benchmark", routine=None, invocation="manual")
    scores = evaluate_groups(groups, config_path=config, ledger=ledger, run_id=run_id)
    ledger.finish_run(run_id, status="completed")
    _print(scores, n_items)
    print(f"  ledger: {db}\n")
