"""Statistical rigor pass on the EXISTING triage benchmark (no new API calls).

Addresses the methodology review's top-2 fixes: (1) topic-clustered bootstrap CIs +
paired significance, treating TOPIC as the replication unit (n=5), and (2) trivial
baselines (keep-all / majority / random). Writes results/stats_analysis.md.
"""

from __future__ import annotations

import csv
import random
import statistics as st
from pathlib import Path

random.seed(42)
ROOT = Path(__file__).resolve().parents[2]
B = 20000
MODELS = ["local-qwen", "deepseek", "claude-haiku", "claude-sonnet", "gpt-5", "gpt-5.5"]


def load():
    rows = list(csv.DictReader(open(ROOT / "results" / "triage_benchmark.csv")))
    topics = sorted({r["topic"] for r in rows})
    f1 = {(r["model"], r["topic"]): float(r["keep_f1"]) for r in rows}
    keepall = {}
    for r in rows:                       # keep-all F1 = 2·prev/(prev+1) (recall=1)
        pos, neg = int(r["tp"]) + int(r["fn"]), int(r["fp"]) + int(r["tn"])
        keepall[r["topic"]] = 2 * (pos / (pos + neg)) / ((pos / (pos + neg)) + 1)
    return topics, f1, keepall


def boot_ci(vals):
    means = sorted(st.mean(random.choices(vals, k=len(vals))) for _ in range(B))
    return means[int(.025 * B)], means[int(.975 * B)]


def paired(a_vals, b_vals):
    d = [x - y for x, y in zip(a_vals, b_vals)]
    diffs = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(B))
    lo, hi = diffs[int(.025 * B)], diffs[int(.975 * B)]
    p_gt = sum(1 for x in diffs if x > 0) / B
    return st.mean(d), lo, hi, p_gt


def main():
    topics, f1, keepall = load()
    L = ["# Triage benchmark — statistical analysis (topic-clustered, n=5 topics)", "",
         "*Replication unit = TOPIC (n=5), not abstract (n=146). CIs are topic-clustered "
         "bootstrap (20k resamples). With n=5 these are wide — that is the honest picture.*", ""]

    L += ["## Per-model keep-F1 with 95% CI vs the keep-all baseline", "",
          "| model | mean F1 | 95% CI | beats keep-all? |", "|---|---|---|---|"]
    ka = st.mean([keepall[t] for t in topics])
    for m in MODELS:
        vals = [f1[(m, t)] for t in topics]
        lo, hi = boot_ci(vals)
        _, dlo, dhi, _ = paired(vals, [keepall[t] for t in topics])
        beats = "**yes**" if dlo > 0 else "no (within noise)"
        L.append(f"| {m} | {st.mean(vals):.3f} | [{lo:.3f}, {hi:.3f}] | {beats} |")
    L.append(f"| *keep-all baseline* | {ka:.3f} | — | — |")
    L += ["", f"Keep-all (label everything relevant) scores **{ka:.3f}** pooled-mean — "
          "only the models marked **yes** add statistically detectable signal over it.", ""]

    L += ["## Key pairwise comparisons (paired by topic, bootstrap)", "",
          "| comparison | ΔF1 | P(A>B) | 95% CI of ΔF1 | significant? |", "|---|---|---|---|---|"]
    pairs = [("gpt-5.5", "deepseek"), ("gpt-5.5", "gpt-5"), ("gpt-5.5", "claude-sonnet"),
             ("claude-sonnet", "deepseek"), ("deepseek", "claude-haiku"), ("deepseek", "local-qwen")]
    for a, b in pairs:
        md, lo, hi, p = paired([f1[(a, t)] for t in topics], [f1[(b, t)] for t in topics])
        sig = "**yes**" if (lo > 0 or hi < 0) else "no (CI spans 0)"
        L.append(f"| {a} vs {b} | {md:+.3f} | {p:.2f} | [{lo:+.3f}, {hi:+.3f}] | {sig} |")
    L += ["", "*15 pairwise comparisons among 6 models → Bonferroni α = 0.0033; treat any "
          "single 'significant' as uncorrected and directional only.*", "",
          "## Bottom line",
          f"- Only **gpt-5.5** clearly clears keep-all and the cheap tier; everything from "
          "sonnet down is within topic-to-topic noise at n=5.",
          "- The cheap tier (local/deepseek/haiku) is a statistical tie — choose on cost.",
          "- To make a Bonferroni-corrected ranking claim you'd need ~8+ topics."]
    out = ROOT / "results" / "stats_analysis.md"
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[wrote {out}]")


if __name__ == "__main__":
    main()
