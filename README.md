# LitStream

<p align="center">
  <a href="https://github.com/yyw-informatics/LitStream/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/yyw-informatics/LitStream/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="LangGraph agent runtime" src="https://img.shields.io/badge/LangGraph-agent_runtime-1C3C3C">
  <img alt="Bioinformatics literature review" src="https://img.shields.io/badge/domain-bioinformatics-0E7C7B">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

LitStream is a LangGraph-based literature research system for bioinformatics. It acquires papers,
reads them through a controlled workflow, extracts structured evidence, checks claims against source
text, synthesizes findings, tracks model calls, and can generate testable hypothesis candidates. The
project is built around one idea: scientific automation should stay auditable, with source provenance,
model routing, scheduled runs, review reports, and generated hypotheses treated as leads to test, not
discoveries.

## What Matters

- **Two summarization tracks.** LitStream supports project-specific method-paper synthesis, such as
  CITE-seq computational-method reviews, and evaluation tooling for experimental/medical-study
  summarization using MSLR-style gold summaries.
- **Grounded evidence before synthesis.** Per-paper notes are structured into evidence records with
  source quotes, then checked before they are reused.
- **Deterministic checks plus constrained LLM-as-judge.** String/number/entity checks provide a
  reproducible floor; optional model-based judges handle narrower semantic questions such as
  entailment, citation support, or conclusion direction.
- **Review layers instead of blind trust.** Synthesis and citations are audited by reports that
  identify claims for inspection, and hypothesis candidates carry evidence paths, warnings, and
  proposed tests.
- **Model mix-and-match.** Triage, extraction, synthesis, evaluation, and design can use different
  backends: local models, DeepSeek, OpenAI-compatible endpoints, or Claude models.
- **Scheduled operation.** Routine files define cron schedules, per-phase models, and run settings;
  every model call is recorded in a SQLite ledger so scheduled jobs remain inspectable.

## Core Workflow

```text
  -> acquire papers
  -> triage for relevance
  -> fetch/read PDFs
  -> mine per-paper evidence
  -> structure and ground extracted claims
  -> synthesize across papers
  -> audit the synthesis
  -> optionally generate hypothesis candidates
  -> deliver digest/report
```

## Summarization And Synthesis

LitStream wraps the workflow from
[yyw-informatics/kb-skills-bioinformatics](https://github.com/yyw-informatics/kb-skills-bioinformatics),
which turns method papers, documentation, biology literature, and a project `context.md` into a
reusable knowledge base and a project-specific analysis plan.

```text
Layer 1: reusable method knowledge
  dry-lab method papers + docs
    -> method knowledge base
       - method goal and intended task
       - input and output data types
       - statistical assumptions and model objective
       - preprocessing, normalization, integration, or denoising choices
       - benchmark datasets, metrics, and baselines
       - limitations, failure modes, and implementation notes

Layer 2: project-specific synthesis, driven by context.md
  wet-lab biology papers + context.md
    -> project evidence
       - genes, proteins, markers, and pathways
       - cell types, tissues, disease states, and species
       - perturbations, treatments, conditions, and comparisons
       - assays, cohorts, sample sizes, and experimental design
       - measured outcomes, effect direction, and statistical support
       - limitations, conflicts, and hypotheses raised by the paper
    -> literature synthesis

  method knowledge base + context.md
    -> method-fit summary

  literature synthesis + method-fit summary + context.md
    -> analysis plan
```

Key outputs: **`concept.md`**, **`theory.md`**, **`code.md`**, optional **`figures.md`**,
**`*_evidence.md`**, **`0_synthesis_literature.md`**, **`fitness_summary.md`**, and
**`analysis_plan.md`**.

## Evaluation

Evaluation is treated as its own workflow, not just a final score. LitStream checks different failure
modes separately so a reviewer can tell whether a problem came from retrieval, extraction, synthesis,
model choice, or hypothesis generation.

```text
Evaluation layers
  deterministic checks
    -> numbers, entities, citations, output files, regression behavior

  constrained LLM-as-judge
    -> entailment, citation support, source-span finding, conclusion direction

  summarization metrics
    -> method-paper synthesis checks
    -> MSLR-style nugget coverage, faithfulness, and conclusion-direction agreement

  model benchmarks
    -> triage accuracy, provider bakeoffs, cost/quality comparison, run variance

  hypothesis validation
    -> smoke candidate, null tests, hidden-edge checks, species and lineage guards
```

See [EVALUATION.md](EVALUATION.md) for the metrics, tools, limitations, and verification commands.

## Model Routing And Benchmarks

LitStream can mix models by task instead of treating one model as the system default. Routine config
and `task_models.yaml` let triage, mining, synthesis, evaluation, and design use different local,
DeepSeek, OpenAI-compatible, or Claude backends. Benchmark scripts compare quality and run-to-run
variance before changing a routing choice.

See [MODEL_ROUTING.md](MODEL_ROUTING.md) for the routing surfaces, benchmark tools, and supporting
implementation details.

## Hypothesis Candidate Generation

`litstream.hypotheses` is an optional package that composes already-grounded evidence into local,
testable hypothesis candidates. The smoke fixture produces a candidate like this:

```text
Evidence:
  IFN-beta increased FOXP3 in regulatory T cells.
  FOXP3 and CD25 marked regulatory T cells.

Candidate:
  In human PBMC, IFN-beta is predicted to increase a FOXP3-associated readout
  in regulatory T cell.

Proposed test:
  CITE-seq in human PBMC regulatory T cells, with FOXP3 RNA and CD25 ADT.

Caveat:
  CD25 ADT is inferred through marker-gene mapping; protein change is not directly supported.
```

See [HYPOTHESIS_GENERATION.md](HYPOTHESIS_GENERATION.md) for the detailed pipeline, motif templates,
example output, safety boundaries, and commands.

## Repository Layout

| Path | Purpose |
|---|---|
| `litstream_lg/` | LangGraph app: state, graph, nodes, routing, tools, model callbacks, CLI |
| `litstream_evidence/` | Framework-neutral evidence schema, structuring, PDF text, grounding, claim battery |
| `litstream/` | Shared engine: acquisition, paper library, ledger, scheduler, delivery, eval, hypotheses |
| `litstream/config/` | Routine, source, and model configuration |
| `tests/` | Offline regression tests and fixtures |

## Quickstart

```bash
# 1. Clone this repo and the research-skills repo it wraps.
git clone https://github.com/yyw-informatics/LitStream-langgraph
cd LitStream-langgraph
git clone https://github.com/yyw-informatics/kb-skills-bioinformatics

# 2. Create the environment.
mamba create -n litstream-lg python=3.12 -y
mamba run -n litstream-lg pip install -e ".[langgraph,dev,hypothesis]"

# 3. Add API keys.
cp .env.example .env
chmod 600 .env

# 4. Run a configured routine.
mamba run -n litstream-lg litstream-lg \
  --routine litstream/config/routines/weekly-citeseq.yaml \
  --project-dir ./kb-skills-bioinformatics \
  --cap-usd 50
```

## Tests

```bash
mamba run -n litstream-lg python -m pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
