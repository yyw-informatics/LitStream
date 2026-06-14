"""Run-to-run variance benchmark — isolate stochastic (re-run) noise from topic noise.

Re-runs each (model, topic) cell REPEATS times at the PRODUCTION temperature (0) and
reports the std of keep-F1 across repeats. Two contrasting topics (one hard, one easy)
× all 6 models. Comparing this run-std against the cross-topic std (~0.10 from the main
benchmark) shows which source of uncertainty dominates — and therefore whether the way
to tighten the estimate is more runs or more topics.

    caffeinate -i mamba run -n litstream python -m litstream.eval.run_variance
"""

from __future__ import annotations

import collections
import csv
import datetime as dt
import statistics
import traceback
from pathlib import Path

from litstream.config.env import load_env
from litstream.ledger.cost import CostLedger
from litstream.eval.csmed_loader import load_reviews, ROOT
from litstream.eval.triage_eval import evaluate_groups

RESULTS = ROOT / "results"
DB = str(ROOT / "litstream.db")
CONFIG = ROOT / "litstream" / "config" / "task_models.yaml"

MODELS = ["local-qwen", "deepseek", "claude-haiku", "claude-sonnet", "gpt-5", "gpt-5.5"]
TOPICS = [("test", "CD013635")]   # one (hard) topic — full 6-model run-variance, resumable
REPEATS = 5
MAX_DOCS = 30


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    log(f"wrote {path.name} ({len(rows)} rows)")


def main() -> None:
    load_env()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ledger = CostLedger(DB)
    ledger.set_policy(cap_usd=50.0)
    mtd0 = ledger.month_to_date_cents() / 100
    log(f"=== RUN-VARIANCE START · MTD ${mtd0:.2f}/$50 · {len(MODELS)} models × "
        f"{len(TOPICS)} topics × {REPEATS} repeats ===")

    rows: list[dict] = []
    done: set = set()
    existing = RESULTS / "run_variance.csv"
    if existing.exists():
        for r in csv.DictReader(open(existing)):
            for k in ("rep", "n"):
                r[k] = int(r[k])
            for k in ("keep_f1", "keep_rec", "keep_prec", "acc3", "cost_usd"):
                r[k] = float(r[k])
            rows.append(r)
            done.add((r["topic"], r["model"], r["rep"]))
        log(f"resuming: {len(rows)} cells already done, will skip them")

    for split, rid in TOPICS:
        gs = load_reviews(split=split, review_ids=[rid], max_docs_per_review=MAX_DOCS)
        if not gs:
            log(f"{rid}: not found in {split}"); continue
        groups = [(g.context, g.items) for g in gs]
        n = sum(len(g.items) for g in gs)
        for model in MODELS:
            for rep in range(REPEATS):
                if (rid, model, rep) in done:
                    continue
                if not ledger.preflight().ok:
                    log("BUDGET cap hit — stopping"); break
                try:
                    run_id = ledger.start_run(f"runvar_{rid}_{model}_{rep}", None, "manual")
                    scores = evaluate_groups(groups, config_path=CONFIG, ledger=ledger,
                                             run_id=run_id, only=[model])
                    ledger.finish_run(run_id)
                    if not scores:
                        log(f"{rid} {model} rep{rep}: no score (build failed)"); continue
                    s = scores[0]
                    rows.append({"topic": rid, "model": model, "rep": rep, "n": s.n,
                                 "keep_f1": round(s.keep_f1, 4), "keep_rec": round(s.keep_rec, 4),
                                 "keep_prec": round(s.keep_prec, 4), "acc3": round(s.acc3, 4),
                                 "cost_usd": round(s.cost_cents / 100, 5)})
                    log(f"{rid} {model} rep{rep}: F1={s.keep_f1:.3f}")
                    write_csv(RESULTS / "run_variance.csv", rows)
                except Exception as e:
                    log(f"{rid} {model} rep{rep} ERROR: {e}\n{traceback.format_exc()}")

    # per-(model,topic) run-variance summary
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["model"], r["topic"])].append(r["keep_f1"])
    summ = []
    for (m, t), f1s in agg.items():
        summ.append({"model": m, "topic": t, "repeats": len(f1s),
                     "mean_f1": round(statistics.mean(f1s), 4),
                     "run_std": round(statistics.pstdev(f1s), 4) if len(f1s) > 1 else 0.0,
                     "min": min(f1s), "max": max(f1s), "range": round(max(f1s) - min(f1s), 4)})
    summ.sort(key=lambda r: (r["model"], r["topic"]))
    write_csv(RESULTS / "run_variance_summary.csv", summ)

    run_stds = [r["run_std"] for r in summ]
    mean_run_std = statistics.mean(run_stds) if run_stds else 0.0
    lines = ["# Run-to-run variance (production temp=0)", "",
             f"mean run-std across cells: **{mean_run_std:.4f}**  ·  "
             f"cross-topic std (main benchmark): ~0.10", "",
             "| model | topic | mean F1 | run-std | range | repeats |",
             "|---|---|---|---|---|---|"]
    for r in summ:
        lines.append(f"| {r['model']} | {r['topic']} | {r['mean_f1']} | {r['run_std']} | "
                     f"{r['range']} | {r['repeats']} |")
    TOPIC_STD = 0.10   # measured cross-topic std from the 5-topic main benchmark
    ratio = (mean_run_std / TOPIC_STD) if TOPIC_STD else float("nan")
    verdict = ("Run noise is much smaller than topic variance, so the benchmark's uncertainty "
               "is dominated by which topics you sample — tighten the estimate with more TOPICS, "
               "not more repeats." if mean_run_std < 0.4 * TOPIC_STD else
               "Run noise is non-trivial relative to topic variance — report both run-variance "
               "and topic-variance.")
    lines += ["", f"**Result (data-driven):** mean run-std = {mean_run_std:.3f} vs cross-topic "
              f"std ≈ {TOPIC_STD:.2f} (ratio {ratio:.2f}). {verdict}"]
    (RESULTS / "run_variance_SUMMARY.md").write_text("\n".join(lines))
    log(f"wrote run_variance_SUMMARY.md · mean run-std {mean_run_std:.4f}")

    mtd1 = ledger.month_to_date_cents() / 100
    log(f"=== DONE · MTD ${mtd1:.2f}/$50 (this run +${mtd1 - mtd0:.2f}) ===")


if __name__ == "__main__":
    main()
