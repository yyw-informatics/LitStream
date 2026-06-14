"""Labeled-set evaluator for the triage task — turns the model bake-off into a
decision by measuring ACCURACY, not just cost.

For each enabled backend in config/task_models.yaml, run every abstract in
triage_set.jsonl through the same triage prompt, parse the predicted label, and
score it against the gold label. Two views:

  - 3-class accuracy: exact match on RELEVANT / BORDERLINE / NOT_RELEVANT.
  - KEEP precision/recall (the budget-relevant view): triage's real job is to
    decide which papers proceed to expensive deep-read. KEEP = (not NOT_RELEVANT).
    * recall  = of papers truly worth keeping, how many did we keep?  (misses = lost science)
    * precision = of papers we kept, how many were worth it?          (junk = wasted spend)

Every call is recorded in the cost ledger, so the final table shows accuracy AND
the real-money cost + latency side by side.

    mamba run -n litstream python -m litstream.eval.triage_eval
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from litstream.config.env import load_env
from litstream.ledger.cost import CostLedger
from litstream.tasks.models import build_model

LABELS = ("RELEVANT", "BORDERLINE", "NOT_RELEVANT")
EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parents[1]


def parse_label(text: str) -> str:
    """Extract a label from a model response. The prompt forces the label first,
    so inspect the leading tokens — robust to a justification that later contains
    a stray 'not' (e.g. 'RELEVANT. The paper does not ...')."""
    up = re.sub(r"[^A-Z_ ]", " ", text.strip().upper())
    tokens = up.split()
    for i, tok in enumerate(tokens[:4]):  # label is at the very start
        if tok in ("NOT", "NOT_RELEVANT"):
            return "NOT_RELEVANT"
        if tok == "BORDERLINE":
            return "BORDERLINE"
        if tok == "RELEVANT":
            return "RELEVANT"
    # fallback: whole-text scan (rare — only if no label up front)
    if "NOT RELEVANT" in up or "NOT_RELEVANT" in up:
        return "NOT_RELEVANT"
    if "BORDERLINE" in up:
        return "BORDERLINE"
    if "RELEVANT" in up:
        return "RELEVANT"
    return "UNKNOWN"


def build_prompt(context: str, item: dict) -> str:
    return (
        f"PROJECT CONTEXT:\n{context}\n\n"
        "Decide whether the following paper is worth deep-reading for THIS project.\n"
        "Respond with EXACTLY one label on the first line — RELEVANT, BORDERLINE, or "
        "NOT_RELEVANT — then one sentence of justification.\n\n"
        f"TITLE: {item['title']}\nABSTRACT: {item['abstract']}"
    )


SYSTEM = ("You are a precise literature-triage filter for a bioinformatics project. "
          "Follow the output format exactly: the first line is the single label.")


@dataclass
class Score:
    backend: str
    model: str
    n: int = 0
    correct3: int = 0                 # exact 3-class matches
    tp: int = 0; fp: int = 0; fn: int = 0; tn: int = 0   # on the KEEP class
    unknown: int = 0
    cost_cents: float = 0.0
    latency_ms: int = 0
    mistakes: list = field(default_factory=list)

    @property
    def acc3(self) -> float: return self.correct3 / self.n if self.n else 0.0
    @property
    def keep_prec(self) -> float: return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
    @property
    def keep_rec(self) -> float: return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
    @property
    def keep_f1(self) -> float:
        p, r = self.keep_prec, self.keep_rec
        return 2 * p * r / (p + r) if (p + r) else 0.0
    @property
    def avg_ms(self) -> int: return self.latency_ms // self.n if self.n else 0


def _keep(label: str) -> bool:
    return label != "NOT_RELEVANT"  # RELEVANT and BORDERLINE both proceed to deep-read


def _score_item(model, sc: Score, context: str, item: dict, spec_name: str,
                ledger: CostLedger, run_id: str) -> None:
    try:
        t0 = time.monotonic()
        res = model.complete(build_prompt(context, item), system=SYSTEM, max_tokens=120)
        sc.latency_ms += int((time.monotonic() - t0) * 1000)
    except Exception as exc:
        sc.mistakes.append((item["id"], item["gold"], f"ERROR:{type(exc).__name__}"))
        sc.n += 1; sc.unknown += 1; sc.fn += _keep(item["gold"]); return

    sc.cost_cents += ledger.record(
        run_id, res.model, phase="triage_eval", role=spec_name,
        input_tokens=res.input_tokens, output_tokens=res.output_tokens,
        cached_input_tokens=res.cached_input_tokens)

    pred, gold = parse_label(res.text), item["gold"]
    sc.n += 1
    if pred == "UNKNOWN": sc.unknown += 1
    if pred == gold: sc.correct3 += 1
    gk, pk = _keep(gold), (pred not in ("NOT_RELEVANT", "UNKNOWN"))
    if gk and pk: sc.tp += 1
    elif not gk and pk: sc.fp += 1
    elif gk and not pk: sc.fn += 1
    else: sc.tn += 1
    if pred != gold: sc.mistakes.append((item["id"], gold, pred))


def evaluate_groups(groups: list[tuple[str, list[dict]]], *, config_path: Path,
                    ledger: CostLedger, run_id: str, only: list[str] | None = None) -> list[Score]:
    """Score every backend over a list of (context, items) groups, pooling results
    per backend. One group = one review/project with its own context. `only` selects
    backends by name regardless of their enabled flag (for benchmarks)."""
    all_specs = yaml.safe_load(config_path.read_text())["task_models"]
    if only:
        by = {s["name"]: s for s in all_specs}
        specs = [by[n] for n in only if n in by]
    else:
        specs = [s for s in all_specs if s.get("enabled", True)]
    scores: list[Score] = []
    for spec in specs:
        if spec.get("price_usd_per_mtok"):
            p = spec["price_usd_per_mtok"]
            ledger.seed_pricing(spec["model"], p["input"], p["output"], p["cached_input"])
        try:
            model = build_model(spec, os.environ.get)
        except Exception as exc:
            print(f"  ! {spec['name']}: could not build ({exc})")
            continue
        sc = Score(spec["name"], spec["model"])
        for context, items in groups:
            for item in items:
                _score_item(model, sc, context, item, spec["name"], ledger, run_id)
        scores.append(sc)
    return scores


def evaluate(*, config_path: Path, set_path: Path, context_path: Path,
             ledger: CostLedger, run_id: str) -> list[Score]:
    """Single hand-labeled set (one shared context)."""
    context = context_path.read_text()
    items = [json.loads(l) for l in set_path.read_text().splitlines() if l.strip()]
    return evaluate_groups([(context, items)], config_path=config_path,
                           ledger=ledger, run_id=run_id)


def _print(scores: list[Score], n_items: int) -> None:
    print(f"\n  Triage evaluation · {n_items} labeled abstracts · KEEP = proceeds to deep-read")
    print("  " + "─" * 78)
    print(f"  {'backend':<14}{'3-class acc':>12}{'keep-prec':>11}{'keep-rec':>10}"
          f"{'keep-F1':>9}{'cost':>10}{'avg ms':>8}")
    for s in scores:
        print(f"  {s.backend:<14}{s.acc3:>11.0%}{s.keep_prec:>11.0%}{s.keep_rec:>10.0%}"
              f"{s.keep_f1:>9.2f}{'$'+format(s.cost_cents/100,'.4f'):>10}{s.avg_ms:>8}")
    print("\n  Mistakes (count; gold → predicted, first few):")
    for s in scores:
        if s.mistakes:
            shown = ", ".join(f"{i}[{g[:3]}→{p[:3]}]" for i, g, p in s.mistakes[:8])
            more = f"  (+{len(s.mistakes) - 8} more)" if len(s.mistakes) > 8 else ""
            print(f"    {s.backend} ({len(s.mistakes)}): {shown}{more}")
        else:
            print(f"    {s.backend}: none ✓")
    print("\n  Reading it: keep-recall protects science (missed papers); keep-precision")
    print("  protects budget (junk sent to expensive deep-read). Borderline labels are")
    print("  judgment calls — edit triage_set.jsonl to match your own.\n")


if __name__ == "__main__":
    load_env()
    config = ROOT / "litstream" / "config" / "task_models.yaml"
    db = ROOT / "litstream.db"
    ledger = CostLedger(str(db))
    ledger.set_policy(cap_usd=50.0)
    run_id = ledger.start_run(project="eval", routine=None, invocation="manual")
    items_n = sum(1 for l in (EVAL_DIR / "triage_set.jsonl").read_text().splitlines() if l.strip())
    scores = evaluate(config_path=config, set_path=EVAL_DIR / "triage_set.jsonl",
                      context_path=EVAL_DIR / "triage_context.txt", ledger=ledger, run_id=run_id)
    ledger.finish_run(run_id, status="completed")
    _print(scores, items_n)
    print(f"  ledger: {db}\n")
