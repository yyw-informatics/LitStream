# Evaluation

LitStream evaluates the system at several levels: paper triage, structured evidence extraction,
summary faithfulness, citation support, model routing, and hypothesis generation. The point is not to
produce one flattering score. The point is to make scientific automation inspectable enough that a
human can see what worked, what failed, and what still needs judgment.

## Evaluation Principles

- **Use deterministic checks where possible.** Numbers, entity mentions, citation structure, file
  outputs, and regression behavior should not depend on another model call.
- **Use LLM-as-judge only for narrow semantic tasks.** Model judges are reserved for questions such as
  entailment, citation support, conclusion direction, or finding a supporting source span.
- **Keep generation and review separate.** Review tools produce reports and labels. They do not silently
  rewrite the synthesis or hide unsupported claims.
- **Benchmark routing choices.** Model selection is treated as an empirical question, especially for
  high-volume tasks such as triage and per-paper mining.
- **Report limitations directly.** Local novelty is not global novelty, LLM judges are not oracles, and
  small benchmark sets should not be oversold.

## Summarization Evaluation

LitStream handles two related but different summarization problems.

### Method-Paper Synthesis

For CITE-seq and other computational-biology method reviews, the pipeline mines papers into structured
evidence and then synthesizes across papers. Evaluation focuses on whether the synthesis is grounded in
the extracted evidence, whether cited claims resolve to the right evidence files, and whether important
numeric or entity details survive intact.

| Check | Tooling | What It Catches |
|---|---|---|
| Structured evidence extraction | `litstream_evidence.evidence_models`, `structure_evidence.py` | Missing or malformed per-paper evidence fields |
| Source grounding | `litstream_evidence.ground_retrieval` | Claims, numbers, or entities not supported by source text |
| Deterministic synthesis faithfulness | `litstream.eval.synthesis_faithfulness` | Fabricated numbers, entity drift, weak lexical support, citation gaps |
| Citation attribution | `litstream.eval.citation_check` | Supportive, partial, contradictory, or irrelevant cited evidence |
| Live synthesis audit | `litstream_lg.synthesis_audit` | A non-destructive audit overlay after synthesis |

This is the evaluation layer most directly tied to the bioinformatics use case: it checks whether
method claims remain connected to paper-level evidence instead of becoming a polished but unsupported
narrative.

### Experimental-Study Summarization

For experimental or medical-study summaries, LitStream includes an MSLR-based evaluation harness. This
is separate from the CITE-seq method-paper workflow. It is useful because MSLR-style gold summaries
let the project test summarization behavior against expert-written conclusions.

The evaluator checks:

- **nugget coverage:** whether gold summary claims are covered by the candidate summary;
- **faithfulness:** whether candidate claims are supported by the source abstracts;
- **conclusion direction:** whether the bottom-line conclusion agrees with the gold label.

This catches failures that ROUGE-style overlap can miss, especially missed findings and flipped
conclusions.

Relevant code:

```text
litstream.eval.mslr_loader
litstream.eval.mslr_eval
```

## Model Routing Benchmarks

LitStream can mix models by task, so evaluation also asks which model should do which job. The goal is
not to rank every model globally. The goal is to decide whether a cheaper or local model is good enough
for a specific phase, and whether stronger models should be reserved for synthesis or design.

| Tool | What It Measures |
|---|---|
| `litstream.eval.triage_eval` | Labeled triage accuracy and KEEP precision/recall |
| `litstream.eval.cost_vs_performance` | Cost per 1,000 screened papers versus triage quality |
| `litstream.eval.provider_bakeoff` | Side-by-side mining outputs across enabled providers |
| `litstream.eval.run_variance` | Run-to-run variance for model outputs |

See [MODEL_ROUTING.md](MODEL_ROUTING.md) for the routing surfaces and model mix-and-match mechanism.

## Hypothesis Candidate Validation

`litstream.hypotheses` has its own evaluation surface because generated hypotheses are higher risk than
summaries. The package is tested as a deterministic, context-bound generator: it should only compose
from already-grounded evidence, keep support frames visible, emit warnings for RNA/protein bridges, and
block incompatible species or cell-lineage merges.

The hypothesis checks cover:

- frame extraction from evidence records;
- graph construction and motif generation;
- compatibility filtering for species and lineage;
- ranking stability and deterministic output;
- smoke output in JSONL, CSV, Markdown, diagnostics JSON, and GraphML;
- null and hidden-edge evaluations.

See [HYPOTHESIS_GENERATION.md](HYPOTHESIS_GENERATION.md) for the example candidate, motif templates,
acceptance criteria, and targeted commands.

## Verification Commands

```bash
mamba run -n litstream-lg python -m pytest -q

mamba run -n litstream-lg python -m pytest tests/test_hypotheses.py -q

mamba run -n litstream-lg python -m litstream.hypotheses run \
  --evidence-dir tests/fixtures/hypotheses/evidence \
  --out-dir /tmp/litstream_hypotheses_smoke \
  --grounder overlap
```

## What This Does Not Claim

- It does not claim that generated hypotheses are globally novel. Novelty is local to the input corpus.
- It does not claim that LLM-as-judge labels are final truth. They are constrained review signals.
- It does not claim that model benchmark rankings are universal. They are task-specific and should be
  rerun when the task, corpus, or model pool changes.
- It does not claim that summarization quality is solved by one metric. LitStream uses complementary
  checks because each metric catches a different failure mode.
