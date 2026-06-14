"""MSLR synthesis-quality eval: two reference-based metrics for multi-document
medical-study summarization, scored against the MSLR gold review summaries.

Two metrics, not ROUGE (n-gram overlap rewards copying the reference's words and is
blind to a flipped conclusion):

  A. Nugget recall / precision / F1 (coverage).
     Split the gold summary into sentence-level nuggets; for each, ask the entailment
     verifier whether the candidate summary supports it -> recall. The reverse
     (candidate nuggets supported by the gold) -> precision. Catches missed findings.

  B. Conclusion-direction agreement.
     An LLM classifies each summary's bottom line into the canonical Cochrane set
     {BENEFIT, HARM, NO_DIFFERENCE, INSUFFICIENT, MIXED}; gold vs candidate -> agreement
     accuracy + Cohen's kappa + confusion. Catches flipped conclusions.

The verifier is reused from litstream_evidence (make_verifier('minicheck'|'overlap'),
.verify(claim, passage) -> (bool, float)). Candidate summaries come from a throwaway
summarizer (any task_models.yaml backend) because LitStream's real SYNTHESIZE is
single-cell-specific and can't summarize Cochrane RCT reviews; MSLR validates the
metric plus a baseline, and the metric code then reuses on real synthesis in
reference-free mode.

Validation runs first and needs no API key: the verifier is scored on the 41-case
claim-entailment battery, and the recall metric is sanity-checked on gold perturbations
(gold-vs-gold should be ~1.0; cross-review and empty candidates should be low).

    # validate the metric machinery only (no API key, no generation):
    python -m litstream.eval.mslr_eval --metric-only --verifier overlap
    # full run: generate cheap candidates + score both metrics on 30 dev reviews:
    python -m litstream.eval.mslr_eval --n 30 --summarizer claude-haiku --verifier minicheck
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from litstream.config.env import load_env
from litstream.eval.mslr_loader import ROOT, load_reviews, ReviewGroup
from litstream.ledger.cost import CostLedger
from litstream.tasks.models import build_model, record_to_ledger
from litstream_evidence.claim_battery import battery_meta, score as battery_score
from litstream_evidence.ground_retrieval import Verifier, make_verifier

RESULTS = ROOT / "results"
CONFIG = ROOT / "litstream" / "config" / "task_models.yaml"


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


# ── Metric A: nugget coverage ───────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def nuggets(text: str, min_len: int = 15) -> list[str]:
    """Sentence-level nuggets, the atomic claims a summary makes. Cochrane targets are
    short bottom-line conclusions, so sentence granularity is the right unit; tiny
    fragments (headers, '.', enumerations) are dropped."""
    return [s.strip() for s in _SENT_SPLIT.split((text or "").strip()) if len(s.strip()) >= min_len]


def _supported_fraction(verifier: Verifier, claims: list[str], passage: str) -> float | None:
    if not claims:
        return None
    # An empty passage supports nothing, and MiniCheck crashes on "", so short-circuit
    # rather than calling the verifier.
    if not (passage or "").strip():
        return 0.0
    return sum(verifier.verify(c, passage)[0] for c in claims) / len(claims)


def coverage(verifier: Verifier, gold: str, candidate: str, source: str | None = None) -> dict:
    """The two synthesis axes, scored jointly:

      recall       gold nuggets the candidate covers (missed findings).
      faithfulness candidate nuggets grounded in the source study abstracts
                   (invented/over-claimed). Uses the inputs as ground truth, so it
                   needs no reference and transfers to real synthesis. Requires `source`.
      F1           harmonic mean of recall and the precision axis (see f1_basis).
      f1_basis     which axis F1 paired recall with: "faithfulness" when a source was given,
                   else "precision_gold". Recorded so a reader of the dict knows what F1 means.

    `precision_gold` (candidate nuggets the terse gold entails) is also reported, but it is
    near-degenerate because Cochrane targets are so short; kept as a secondary signal.
    """
    gn, cn = nuggets(gold), nuggets(candidate)
    rec = _supported_fraction(verifier, gn, candidate)
    faith = _supported_fraction(verifier, cn, source) if source is not None else None
    prec_gold = _supported_fraction(verifier, cn, gold)
    # F1 pairs recall with faithfulness when a source is given, else with precision_gold.
    base = faith if faith is not None else prec_gold
    f1_basis = "faithfulness" if faith is not None else "precision_gold"
    if rec is None or base is None:
        f1 = None                                # an axis is unmeasured: F1 is undefined, not 0
    elif rec + base == 0:
        f1 = 0.0                                 # both zero: harmonic mean is 0 (avoid div-by-zero)
    else:
        f1 = 2 * rec * base / (rec + base)
    return {"recall": rec, "faithfulness": faith, "precision_gold": prec_gold,
            "f1": f1, "f1_basis": f1_basis, "n_gold": len(gn), "n_cand": len(cn)}


# ── Metric B: conclusion-direction agreement ────────────────────────────────────

DIRECTIONS = ("BENEFIT", "HARM", "NO_DIFFERENCE", "INSUFFICIENT", "MIXED")

_DIR_SYSTEM = ("You classify the overall bottom-line conclusion of a systematic-review "
               "summary. Output EXACTLY one label on the first line, nothing else.")

_DIR_PROMPT = (
    "Classify the OVERALL bottom-line conclusion of the systematic-review summary below "
    "as exactly one of:\n"
    "- BENEFIT: the intervention helped / was favoured overall.\n"
    "- HARM: the intervention was worse or caused net harm.\n"
    "- NO_DIFFERENCE: no meaningful difference between arms.\n"
    "- INSUFFICIENT: evidence too weak/sparse/uncertain to draw a conclusion.\n"
    "- MIXED: benefit on some outcomes, harm/no-effect on others — genuinely mixed.\n\n"
    "Answer with the single label on the first line.\n\nSUMMARY:\n{text}"
)


def parse_direction(text: str) -> str:
    up = re.sub(r"[^A-Z_ ]", " ", (text or "").upper())
    tokens = up.split()
    for tok in tokens[:4]:
        if tok in DIRECTIONS:
            return tok
        if tok == "NO" and "DIFFERENCE" in tokens[:5]:
            return "NO_DIFFERENCE"
    for d in DIRECTIONS:                       # fallback: anywhere in the text
        if d in up or d.replace("_", " ") in up:
            return d
    return "UNKNOWN"


def classify_direction(model, text: str, ledger: CostLedger, run_id: str) -> str:
    res = model.complete(_DIR_PROMPT.format(text=text[:6000]), system=_DIR_SYSTEM, max_tokens=16)
    record_to_ledger(ledger, run_id, res, phase="mslr_eval", role="direction")
    return parse_direction(res.text)


def cohen_kappa(a: list[str], b: list[str], labels) -> float:
    n = len(a)
    if not n:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ── candidate generation (system-under-test) ────────────────────────────────────

_GEN_SYSTEM = ("You write the bottom-line conclusions section of a systematic review. "
               "Be faithful to the studies; do not invent findings.")

_GEN_PROMPT = (
    "Below are the abstracts of the studies included in a systematic review. Write a "
    "concise (3-6 sentence) evidence summary: what the included studies found overall, "
    "the direction of effect, and the strength/certainty of the evidence. Do not add "
    "findings that are not supported by these abstracts.\n\n{studies}"
)


def study_blocks(group: ReviewGroup, *, max_studies: int, max_abstract_chars: int) -> str:
    """The included studies as one text block: the model's input and the grounding source
    that faithfulness is scored against, so the candidate is only judged on text it saw."""
    blocks = []
    for s in group.inputs[:max_studies]:
        ab = s["abstract"][:max_abstract_chars]
        blocks.append(f"STUDY (PMID {s['pmid']}): {s['title']}\n{ab}")
    return "\n\n".join(blocks)


def generate_candidate(model, group: ReviewGroup, *, max_studies: int, max_abstract_chars: int,
                       ledger: CostLedger, run_id: str):
    studies = study_blocks(group, max_studies=max_studies, max_abstract_chars=max_abstract_chars)
    res = model.complete(_GEN_PROMPT.format(studies=studies), system=_GEN_SYSTEM, max_tokens=400)
    record_to_ledger(ledger, run_id, res, phase="mslr_eval", role="generate")
    return res.text.strip()


def load_or_generate_candidates(groups: list[ReviewGroup], model, cache: Path, *,
                                max_studies: int, max_abstract_chars: int,
                                ledger: CostLedger, run_id: str) -> dict[str, str]:
    """Generate a candidate summary per review, cached and resumable: existing review_ids
    in the cache file are reused, only new ones hit the model."""
    cand: dict[str, str] = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cand[row["review_id"]] = row["candidate"]
        log(f"resumed {len(cand)} cached candidates from {cache.name}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("a") as fh:
        for i, g in enumerate(groups, 1):
            if g.review_id in cand:
                continue
            text = generate_candidate(model, g, max_studies=max_studies,
                                      max_abstract_chars=max_abstract_chars,
                                      ledger=ledger, run_id=run_id)
            cand[g.review_id] = text
            fh.write(json.dumps({"review_id": g.review_id, "candidate": text}) + "\n")
            fh.flush()
            if i % 5 == 0:
                log(f"generated {i}/{len(groups)} candidates")
    return cand


# ── validation (no API key needed) ──────────────────────────────────────────────

def validate_metric(verifier: Verifier, golds: list[str]) -> dict:
    """Check the recall metric discriminates: gold-vs-gold should be ~1.0, a different
    review's gold (cross) should be low, and an empty candidate should be ~0. If self
    isn't comfortably above cross, the metric isn't measuring coverage."""
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0
    # Cross pairs each gold with a different review's gold (rotate by one); needs >=2 reviews,
    # otherwise the rotation would pair the lone gold with itself and inflate the cross score.
    cross_pairs = list(zip(golds, golds[1:] + golds[:1])) if len(golds) > 1 else []
    return {
        "self": mean(coverage(verifier, g, g)["recall"] for g in golds),
        "cross": mean(coverage(verifier, g, c)["recall"] for g, c in cross_pairs),
        "empty": mean(coverage(verifier, g, "")["recall"] for g in golds),
        "n": len(golds),
    }


# ── IO ───────────────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["review_id", "n_studies", "n_gold_nuggets", "n_cand_nuggets",
            "recall", "faithfulness", "precision_gold", "f1", "gold_dir", "cand_dir", "dir_match"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {path.name} ({len(rows)} reviews)")


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.1%}"


def _macro(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


# ── per-model bake-off ──────────────────────────────────────────────────────────

def precompute_gold_directions(classifier, groups: list[ReviewGroup], ledger: CostLedger,
                               run_id: str) -> dict[str, str]:
    """Classify each gold summary's bottom line once with the fixed classifier, shared
    across every candidate model so direction agreement is comparable between them."""
    return {g.review_id: classify_direction(classifier, g.target, ledger, run_id) for g in groups}


def run_model(summarizer: str, groups: list[ReviewGroup], verifier: Verifier, classifier,
              gold_dirs: dict[str, str], *, args, specs: dict, db_path: Path) -> dict:
    """Generate candidates with one summarizer, score both metrics. Own ledger run, own
    candidate cache, own per-model CSV. The direction classifier is the shared fixed model,
    not the summarizer: the summarizer is the system under test, the classifier is the ruler.
    Returns one aggregate row for the comparison table."""
    ledger = CostLedger(str(db_path))
    ledger.set_policy(cap_usd=50.0)
    run_id = ledger.start_run(project=f"mslr_eval:{summarizer}", routine=None, invocation="manual")
    spec = specs[summarizer]
    if spec.get("price_usd_per_mtok"):
        p = spec["price_usd_per_mtok"]
        ledger.seed_pricing(spec["model"], p["input"], p["output"], p["cached_input"])
    model = build_model(spec, os.environ.get)

    cache = RESULTS / f"mslr_candidates_{summarizer}_{args.split}.jsonl"
    cand = load_or_generate_candidates(groups, model, cache, max_studies=args.max_docs,
                                       max_abstract_chars=args.max_abstract_chars,
                                       ledger=ledger, run_id=run_id)
    log(f"[{summarizer}] scoring coverage + direction over {len(groups)} reviews…")
    rows, gd_list, cd_list, confusion = [], [], [], defaultdict(int)
    for g in groups:
        c = cand[g.review_id]
        src = study_blocks(g, max_studies=args.max_docs, max_abstract_chars=args.max_abstract_chars)
        cov = coverage(verifier, g.target, c, source=src)
        gd, cd = gold_dirs[g.review_id], classify_direction(classifier, c, ledger, run_id)
        gd_list.append(gd); cd_list.append(cd); confusion[(gd, cd)] += 1
        rows.append({"review_id": g.review_id, "n_studies": len(g.inputs),
                     "n_gold_nuggets": cov["n_gold"], "n_cand_nuggets": cov["n_cand"],
                     "recall": cov["recall"], "faithfulness": cov["faithfulness"],
                     "precision_gold": cov["precision_gold"], "f1": cov["f1"],
                     "gold_dir": gd, "cand_dir": cd, "dir_match": int(gd == cd)})
    ledger.finish_run(run_id, status="completed")
    write_csv(RESULTS / f"mslr_eval_{summarizer}.csv", rows)
    acc = sum(a == b for a, b in zip(gd_list, cd_list)) / len(gd_list) if gd_list else 0.0
    return {"summarizer": summarizer, "model": spec["model"], "n": len(rows),
            "recall": _macro(rows, "recall"), "faithfulness": _macro(rows, "faithfulness"),
            "f1": _macro(rows, "f1"), "precision_gold": _macro(rows, "precision_gold"),
            "dir_acc": acc, "kappa": cohen_kappa(gd_list, cd_list, DIRECTIONS),
            "confusion": dict(confusion), "cost": ledger.run_cost_cents(run_id) / 100}


def summarize_bakeoff(cfg: dict, results: list[dict], battery: dict, valid: dict,
                      gold_dist: Counter | None) -> str:
    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    lines = [
        f"# MSLR synthesis bake-off — {cfg['subset']}/{cfg['split']}", "",
        f"reviews: **{cfg['n']}** · verifier: **{cfg['verifier']}** · "
        f"direction classifier: **{cfg['classifier']}**", "",
    ]
    if ok:
        lines += [
            "## Model comparison",
            "Metric A = nugget coverage/faithfulness · Metric B = conclusion-direction agreement.", "",
            "| Model | Recall | Faithfulness | F1 | Dir-agree | κ | Cost |",
            "|---|--:|--:|--:|--:|--:|--:|",
        ] + [
            f"| `{r['summarizer']}` ({r['model']}) | {_fmt(r['recall'])} | {_fmt(r['faithfulness'])} "
            f"| {_fmt(r['f1'])} | {r['dir_acc']:.0%} | {r['kappa']:.2f} | ${r['cost']:.4f} |"
            for r in ok
        ] + [""]
        lines += ["Recall = gold points covered · Faithfulness = candidate claims grounded in the "
                  "source studies · κ = chance-corrected direction agreement.", ""]
    if gold_dist:
        lines += ["**Gold direction mix** (the reference labels every candidate is compared against): "
                  + ", ".join(f"{d} {n}" for d, n in gold_dist.most_common()), ""]
    if errs:
        lines += ["**Models that failed to run:** "
                  + ", ".join(f"`{r['summarizer']}` ({r['error']})" for r in errs), ""]

    pm = battery["by_mode"]
    key_modes = [m for m in ("direction_reversal", "negation", "contradiction", "distribution_swap")
                 if m in pm]
    lines += [
        "## Validation (verifier + metric, model-independent)", "",
        f"**Verifier on the 41-case claim battery:** {battery['correct']}/{battery['n']} = "
        f"**{battery['accuracy']:.0%}**"
        + (" — key modes: " + ", ".join(f"{m} {pm[m]['correct']}/{pm[m]['n']}" for m in key_modes)
           if key_modes else ""), "",
        "**Recall-metric discrimination (gold perturbations):** "
        f"self **{valid['self']:.0%}** / cross **{valid['cross']:.0%}** / empty **{valid['empty']:.0%}** "
        + ("✅ discriminates" if valid['self'] - valid['cross'] > 0.2 else "⚠️ weak separation"), "",
        "> **Caveats — read before trusting these numbers.**",
        "> - Cochrane targets are terse, so recall has a low ceiling; `precision_gold` (in the CSVs) is "
        "near-degenerate — faithfulness-vs-source replaces it as the precision axis.",
        "> - Whether either metric correlates with EXPERT quality judgment is unvalidated (open gap).",
        "> - Direction agreement depends on the classifier; it is itself unaudited here. The "
        "entailment verifier is weakest on numeric claims (guarded by a number-presence rule).",
        "> - General biomedical RCT domain, not single-cell — validates the metric + baselines, not "
        "topic fit to LitStream's own projects.",
    ]
    return "\n".join(lines) + "\n"


# ── runner ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="MSLR synthesis bake-off — coverage/faithfulness + direction")
    ap.add_argument("--subset", default="cochrane", choices=["cochrane", "ms2"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=30, help="sample size (reviews); 0 = all")
    ap.add_argument("--min-docs", type=int, default=2, help="require >= this many studies/review")
    ap.add_argument("--max-docs", type=int, default=8, help="cap studies fed per review (cost)")
    ap.add_argument("--max-abstract-chars", type=int, default=2000)
    ap.add_argument("--summarizer", default="claude-haiku",
                    help="comma-separated task_models.yaml backends to bake off as candidate summarizers")
    ap.add_argument("--classifier", default="claude-haiku",
                    help="fixed model that labels conclusion direction for gold AND all candidates")
    ap.add_argument("--verifier", default="minicheck", choices=["overlap", "minicheck"])
    ap.add_argument("--minicheck-model", default="flan-t5-large")
    ap.add_argument("--metric-only", action="store_true",
                    help="validation only — no generation, no API key needed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    load_env()
    verifier = make_verifier(args.verifier, args.minicheck_model)
    summarizers = [s.strip() for s in args.summarizer.split(",") if s.strip()]

    groups = load_reviews(subset=args.subset, split=args.split, min_docs=args.min_docs,
                          max_docs_per_review=args.max_docs)
    random.Random(args.seed).shuffle(groups)
    if args.n:
        groups = groups[:args.n]
    log(f"{len(groups)} reviews (subset={args.subset} split={args.split} min_docs={args.min_docs})")

    meta = battery_meta()
    log(f"scoring verifier '{args.verifier}' on claim battery ({meta.get('n', '?')} cases)…")
    battery = battery_score(verifier)
    log(f"  battery: {battery['correct']}/{battery['n']} = {battery['accuracy']:.0%}")

    valid = validate_metric(verifier, [g.target for g in groups])
    log(f"  perturbations: self={valid['self']:.0%} cross={valid['cross']:.0%} empty={valid['empty']:.0%}")

    results: list[dict] = []
    gold_dist: Counter | None = None
    if not args.metric_only:
        import yaml
        db = ROOT / "litstream.db"
        specs = {s["name"]: s for s in yaml.safe_load(CONFIG.read_text())["task_models"]}
        classifier = build_model(specs[args.classifier], os.environ.get)
        gledger = CostLedger(str(db))
        grun = gledger.start_run(project="mslr_eval:gold", routine=None, invocation="manual")
        gold_dirs = precompute_gold_directions(classifier, groups, gledger, grun)
        gledger.finish_run(grun, status="completed")
        gold_dist = Counter(gold_dirs.values())
        log(f"gold directions: {dict(gold_dist)}")

        for s in summarizers:
            try:
                r = run_model(s, groups, verifier, classifier, gold_dirs,
                              args=args, specs=specs, db_path=db)
                log(f"[{s}] recall {_fmt(r['recall'])} · faithful {_fmt(r['faithfulness'])} · "
                    f"dir {r['dir_acc']:.0%} · ${r['cost']:.4f}")
            except Exception as exc:  # noqa: BLE001 — a bad model id/key shouldn't kill the others
                log(f"[{s}] FAILED: {type(exc).__name__}: {exc}")
                r = {"summarizer": s, "error": f"{type(exc).__name__}: {exc}"}
            results.append(r)

    cfg = {"subset": args.subset, "split": args.split, "n": len(groups),
           "verifier": args.verifier, "classifier": args.classifier}
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = summarize_bakeoff(cfg, results, battery, valid, gold_dist)
    (RESULTS / "mslr_bakeoff_SUMMARY.md").write_text(summary)
    log("wrote mslr_bakeoff_SUMMARY.md")
    print("\n" + summary)


if __name__ == "__main__":
    main()
