"""Provider bake-off for the agentic MINE task — single-shot, cross-provider.

Inject the mine-paper SKILL.md as the (cacheable) system prompt + a paper's text
as the user message, ask for the evidence-report markdown, and run it across every
enabled provider in task_models.yaml. Records cache-aware cost to the ledger and
writes each provider's output for side-by-side quality comparison.

Single-shot (not the full multi-turn agentic loop) so it is fair + tractable across
providers and reflects how you'd actually run a non-Claude model. Running ≥2 papers
shows the cache discount: the big skill system prefix is cached after the first call.

    mamba run -n litstream python -m litstream.eval.provider_bakeoff [n_papers]
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from litstream.config.env import load_env
from litstream.ledger.cost import CostLedger
from litstream.tasks.models import build_model, record_to_ledger
from litstream.skills import load_skill_body
from litstream_evidence.pdf_text import extract_text

ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = ROOT / "kb-skills-bioinformatics" / "skills"
PROJECT = ROOT / "kb-skills-bioinformatics" / "projects" / "citeseq_methods"
OUT_DIR = Path("/tmp/litstream_bakeoff")

USER_TMPL = (
    "PROJECT CONTEXT:\n{context}\n\n"
    "PAPER TEXT (extracted):\n{paper}\n\n"
    "Produce the evidence-report markdown for THIS paper, following the procedure in "
    "the system prompt. Output ONLY the markdown content (frontmatter + report)."
)


@dataclass
class Tally:
    provider: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    creation_tokens: int = 0
    output_tokens: int = 0
    cost_cents: float = 0.0
    latency_ms: int = 0


def main() -> None:
    load_env()
    n_papers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    system = load_skill_body("mine-paper", SKILLS_DIR)
    context = (PROJECT / "context.md").read_text()
    papers = sorted((PROJECT / "papers").glob("*.pdf"))[:n_papers]
    if not papers:
        print("no staged papers found"); return

    specs = [s for s in yaml.safe_load((ROOT / "litstream" / "config" / "task_models.yaml").read_text())
             ["task_models"] if s.get("enabled", True)]

    ledger = CostLedger(str(ROOT / "litstream.db"))
    ledger.set_policy(cap_usd=50.0)
    run_id = ledger.start_run(project="provider_bakeoff", routine=None, invocation="manual")

    print(f"\n  Provider bake-off · MINE task · {len(papers)} papers × {len(specs)} providers")
    print(f"  skill system prompt: {len(system):,} chars (cacheable)\n")

    tallies: list[Tally] = []
    for spec in specs:
        p = spec["price_usd_per_mtok"]
        ledger.seed_pricing(spec["model"], p["input"], p["output"], p["cached_input"])
        try:
            model = build_model(spec, os.environ.get)
        except Exception as exc:
            print(f"  ! {spec['name']}: build failed ({exc})"); continue
        t = Tally(spec["name"], spec["model"])
        for paper in papers:
            user = USER_TMPL.format(context=context, paper=extract_text(paper))
            try:
                res = model.complete(user, system=system, max_tokens=4000)
            except Exception as exc:
                print(f"    {spec['name']} / {paper.name}: ERROR {type(exc).__name__}: {str(exc)[:60]}")
                continue
            t.calls += 1
            t.input_tokens += res.input_tokens
            t.cached_tokens += res.cached_input_tokens
            t.creation_tokens += res.cache_creation_tokens
            t.output_tokens += res.output_tokens
            t.latency_ms += res.latency_ms
            t.cost_cents += record_to_ledger(ledger, run_id, res, phase="mine", role=spec["name"])
            (OUT_DIR / f"{spec['name']}__{paper.stem[:30]}.md").write_text(res.text)
        tallies.append(t)
    ledger.finish_run(run_id)

    print(f"  {'provider':<14}{'in':>8}{'cached':>8}{'creat':>7}{'out':>7}{'cost':>10}{'avg ms':>8}")
    for t in tallies:
        if not t.calls:
            print(f"  {t.provider:<14} — no successful calls —"); continue
        print(f"  {t.provider:<14}{t.input_tokens:>8}{t.cached_tokens:>8}{t.creation_tokens:>7}"
              f"{t.output_tokens:>7}{'$'+format(t.cost_cents/100,'.4f'):>10}{t.latency_ms//t.calls:>8}")
    print(f"\n  outputs for quality comparison: {OUT_DIR}/")
    print("  (cached>0 on paper 2+ = the skill system prefix was reused at the discount rate)\n")


if __name__ == "__main__":
    main()
