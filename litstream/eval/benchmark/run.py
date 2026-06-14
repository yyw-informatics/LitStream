"""Run a benchmark and write a labeled precision/recall/F1 report.

  python3 -m litstream.eval.benchmark.run --dataset biored --n 20
  python3 -m litstream.eval.benchmark.run --dataset jnlpba --n all
  python3 -m litstream.eval.benchmark.run --dataset all --n 50 --extractor baseline

THE SCALE KNOB
  --n 10 | 100 | all      how many documents to score (default 20)
  --sample random|first   which ones; random uses a fixed seed so the same sample
                          comes back every run (fair for comparing models) (default random)
  --seed 12345            the fixed number behind 'random' (default 12345)
  --at-once 8             how many to run at once (parallelism for the model extractor)

Reports are per canonical field (fields.py). On MeasEval, frequencies and
gating_thresholds share one undifferentiated pool of quantities — the benchmark can't
tell them apart — so they are scored together; the split is real only on actual papers.
Every report is stamped with dataset + extractor + match rule + sample size + seed.
"""

from __future__ import annotations

import argparse
import csv
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import biored, cellxgene, fetch, jnlpba, measeval
from .extractors import EXTRACTORS
from .fields import (CELL_TYPES, FREQUENCIES, GATING_THRESHOLDS, GENES, SPECIES,
                     matchable)
from .schema import Document
from .score import MATCHERS, count, prf

RESULTS = Path(__file__).resolve().parents[3] / "results"  # repo-root /results (matches other eval tools)

# A scoring unit = a reported row: a label, which record field(s) feed predictions,
# which gold key holds the answers, and which matcher family grades it.
DATASETS = {
    "biored": {"fetch": fetch.ensure_biored, "load": biored.load, "units": [
        {"label": GENES, "fields": [GENES], "gold": GENES, "match": "entity"},
        {"label": SPECIES, "fields": [SPECIES], "gold": SPECIES, "match": "entity"}]},
    "jnlpba": {"fetch": fetch.ensure_jnlpba, "load": jnlpba.load, "units": [
        {"label": CELL_TYPES, "fields": [CELL_TYPES], "gold": CELL_TYPES, "match": "entity"}]},
    # in-domain cell-type benchmark from real single-cell studies (CL-labeled gold).
    # 'cellxgene' = study description text; 'cellxgene-abstract' = the real paper abstract
    # (richer, realistic prose).
    "cellxgene": {"fetch": fetch.ensure_cellxgene, "load": cellxgene.load, "units": [
        {"label": CELL_TYPES, "fields": [CELL_TYPES], "gold": CELL_TYPES, "match": "entity"}]},
    "cellxgene-abstract": {"fetch": fetch.ensure_cellxgene_abstracts, "load": cellxgene.load,
        "units": [{"label": CELL_TYPES, "fields": [CELL_TYPES], "gold": CELL_TYPES,
                   "match": "entity"}]},
    # MeasEval: one pooled quantity gold stands in for BOTH number fields
    "measeval": {"fetch": fetch.ensure_measeval, "load": measeval.load, "units": [
        {"label": "frequencies+gating_thresholds", "fields": [FREQUENCIES, GATING_THRESHOLDS],
         "gold": FREQUENCIES, "match": "number"}]},
}


def n_arg(v):
    """--n is 'all' or a non-negative integer; anything else is a clean usage error."""
    if v == "all":
        return v
    try:
        iv = int(v)
    except ValueError:
        iv = -1
    if iv < 0:
        raise argparse.ArgumentTypeError(f"--n must be a non-negative integer or 'all', got {v!r}")
    return iv


def take_sample(docs: list[Document], n, mode: str, seed: int) -> list[Document]:
    if n == "all" or int(n) >= len(docs):
        return docs
    n = int(n)
    if mode == "first":
        return docs[:n]
    return random.Random(seed).sample(docs, n)  # fixed seed -> same sample every run


def extract_all(docs: list[Document], extract, at_once: int) -> list[dict]:
    if at_once <= 1:
        recs = [extract(d.text) for d in docs]
    else:
        with ThreadPoolExecutor(max_workers=at_once) as ex:
            recs = list(ex.map(lambda d: extract(d.text), docs))  # map preserves order
    for d, r in zip(docs, recs):
        r["paper_id"] = d.id
    return recs


def run_one(name: str, args) -> None:
    meta = DATASETS[name]
    all_docs = meta["load"](meta["fetch"]())
    docs = take_sample(all_docs, args.n, args.sample, args.seed)
    raw = EXTRACTORS[args.extractor]
    extract = (lambda t: raw(t, model=args.model)) if args.extractor == "mine" else raw
    preds = extract_all(docs, extract, args.at_once)
    units = meta["units"]
    modes = ["strict", "relaxed"] if args.match == "both" else [args.match]

    who = f"{args.extractor} ({args.model})" if args.extractor == "mine" else args.extractor
    gold_n = {u["label"]: sum(len(d.gold.get(u["gold"], set())) for d in docs) for u in units}
    lines = [f"# {name} extraction benchmark", "",
             f"method (extractor): **{who}** · "
             f"sample: **{len(docs)}/{len(all_docs)}** docs "
             f"({args.sample}, seed {args.seed}) · matching one-to-one", "",
             "| field | match | precision | recall | F1 | TP | FP | FN | gold |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for mode in modes:
        for u in units:
            matcher = MATCHERS[(u["match"], mode)]
            agg = {"tp": 0, "fp": 0, "fn": 0}
            for doc, rec in zip(docs, preds):
                gold = doc.gold.get(u["gold"], set())
                pred: set[str] = set()
                for f in u["fields"]:
                    pred |= matchable(rec, f)
                b = count(gold, pred, matcher)
                for k in agg:
                    agg[k] += b[k]
            p, r, f = prf(agg)
            lines.append(f"| {u['label']} | {mode} | {p:.3f} | {r:.3f} | {f:.3f} "
                         f"| {agg['tp']} | {agg['fp']} | {agg['fn']} | {gold_n[u['label']]} |")
            rows.append({"dataset": name, "extractor": args.extractor,
                         "model": args.model if args.extractor == "mine" else "",
                         "field": u["label"],
                         "match": mode, "n_docs": len(docs), "seed": args.seed,
                         "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                         **agg})

    note = ("token-overlap-keyed-on-value" if units[0]["match"] == "number"
            else "substring (CD8~CD8A)")
    lines += ["",
              "> precision = of what was pulled, how much was a real gold item; "
              "recall = of the gold items, how many were found (one-to-one matched). "
              f"strict = exact string; relaxed = {note}."]
    if name == "measeval":
        lines.append("> note: MeasEval pools frequencies + gating_thresholds into one "
                     "quantity gold — they only separate on real papers.")

    tag = f"mine-{args.model}" if args.extractor == "mine" else args.extractor
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}_{tag}_benchmark_SUMMARY.md").write_text("\n".join(lines) + "\n")
    with open(RESULTS / f"{name}_{tag}_benchmark.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\n".join(lines))
    print(f"\nwrote results/{name}_{tag}_benchmark_SUMMARY.md + .csv\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score MINE-style extraction on public benchmarks")
    ap.add_argument("--dataset", default="biored", choices=[*DATASETS, "all"])
    ap.add_argument("--extractor", default="baseline", choices=list(EXTRACTORS))
    ap.add_argument("--model", default="deepseek",
                    help="model name from task_models.yaml for --extractor mine (e.g. deepseek, claude-haiku)")
    ap.add_argument("--n", type=n_arg, default=20, help="how many docs to score (int or 'all')")
    ap.add_argument("--sample", default="random", choices=["random", "first"])
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--match", default="both", choices=["strict", "relaxed", "both"])
    ap.add_argument("--at-once", type=int, default=8, dest="at_once",
                    help="parallel extractions (used by the model extractor)")
    args = ap.parse_args()

    for name in (list(DATASETS) if args.dataset == "all" else [args.dataset]):
        try:
            run_one(name, args)
        except NotImplementedError as exc:
            print(f"[{name}] skipped: {exc}\n")


if __name__ == "__main__":
    main()
