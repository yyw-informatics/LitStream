"""Cost-vs-performance benchmark for the triage task on CSMeD (ground-truth labels).

The headline demonstration: strategic model selection cuts cost dramatically while
holding accuracy — because on a bounded classification task, a cheap model matches a
premium one. Runs several models (free local → cheap API → premium) over real
Cochrane abstracts with expert include/exclude labels, and reports keep-F1 + 3-class
accuracy against $ per 1,000 papers screened.

    mamba run -n litstream python -m litstream.eval.cost_vs_performance
    mamba run -n litstream python -m litstream.eval.cost_vs_performance CD013635 \
        local-qwen,deepseek,claude-haiku,claude-sonnet
"""

from __future__ import annotations

import sys

from litstream.config.env import load_env
from litstream.ledger.cost import CostLedger
from litstream.eval.csmed_loader import load_reviews, ROOT
from litstream.eval.triage_eval import evaluate_groups

DEFAULT_REVIEWS = ["CD013635"]                 # 90 abstracts, balanced (42 incl / 48 excl)
DEFAULT_MODELS = ["local-qwen", "deepseek", "claude-haiku", "claude-sonnet"]


def main() -> None:
    load_env()
    reviews = sys.argv[1].split(",") if len(sys.argv) > 1 else DEFAULT_REVIEWS
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else DEFAULT_MODELS

    groups_obj = load_reviews(split="test", review_ids=reviews) or load_reviews(
        split="dev", review_ids=reviews)
    groups = [(g.context, g.items) for g in groups_obj]
    n = sum(len(g.items) for g in groups_obj)
    n_rel = sum(i["gold"] == "RELEVANT" for g in groups_obj for i in g.items)
    print(f"\n  COST vs PERFORMANCE — triage on CSMeD (expert labels)")
    print(f"  {n} abstracts ({n_rel} relevant / {n - n_rel} not) · models: {', '.join(models)}\n")

    config = ROOT / "litstream" / "config" / "task_models.yaml"
    ledger = CostLedger(str(ROOT / "litstream.db"))
    ledger.set_policy(cap_usd=50.0)
    run_id = ledger.start_run(project="cost_vs_perf", routine=None, invocation="manual")
    scores = evaluate_groups(groups, config_path=config, ledger=ledger, run_id=run_id, only=models)
    ledger.finish_run(run_id)

    rows = []
    for s in scores:
        usd_per_1k = (s.cost_cents / s.n) * 1000 / 100 if s.n else 0.0
        rows.append((s, usd_per_1k))
    best_f1 = max((s.keep_f1 for s, _ in rows), default=0)
    cheapest_ok = min((u for s, u in rows if best_f1 - s.keep_f1 <= 0.05 and u > 0), default=None)

    print(f"  {'model':<15}{'keep-F1':>9}{'3-class acc':>13}{'$/1k papers':>13}{'vs best F1':>12}")
    print("  " + "─" * 62)
    for s, u in sorted(rows, key=lambda r: r[1]):
        delta = s.keep_f1 - best_f1
        print(f"  {s.backend:<15}{s.keep_f1:>9.2f}{s.acc3:>12.0%}{'$'+format(u,'.3f'):>13}"
              f"{('' if delta==0 else format(delta,'+.2f')):>12}")
    print()
    if cheapest_ok is not None:
        prem = max(rows, key=lambda r: r[1])      # most expensive
        if prem[1] > 0:
            save = (1 - cheapest_ok / prem[1]) * 100
            print(f"  Headline: the cheapest model within 0.05 F1 of the best costs "
                  f"${cheapest_ok:.3f}/1k vs the premium ${prem[1]:.3f}/1k "
                  f"→ ~{save:.0f}% cheaper at ~equal accuracy.\n")


if __name__ == "__main__":
    main()
