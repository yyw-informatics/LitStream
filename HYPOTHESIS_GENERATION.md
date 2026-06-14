# Context-Bound Hypothesis Generation

`litstream.hypotheses` turns grounded biomedical evidence records into ranked, testable
hypothesis candidates. The module is designed for scientific AI workflows where the goal is not
free-form ideation, but disciplined generation of plausible next experiments from auditable
literature evidence.

The system separates three responsibilities that are often conflated in LLM-based discovery tools:

- extracting structured claims from source-grounded evidence;
- composing those claims through typed, inspectable motifs;
- ranking outputs by evidence quality, biological context compatibility, measurability, and risk.

The result is a candidate-generation layer for AI-assisted science: every proposed hypothesis
preserves its evidence path, exposes its assumptions, and resolves to an experimentally measurable
readout.

## Research Contribution

This component implements a conservative hypothesis-generation architecture for literature-driven
single-cell biology:

- **Typed evidence representation.** Findings are normalized into perturbation, disease, gene,
  signature, marker, cell-type, species, tissue, and evidence-mode frames.
- **Source-grounded support.** Candidate construction only uses support frames that pass quote-level
  grounding checks from the configured verifier.
- **Graph-based composition.** Evidence is assembled into a typed graph, enabling local multi-hop
  reasoning without asking a model to invent unsupported mechanisms.
- **Biology-aware constraints.** Cross-species joins, incompatible cell-lineage compositions,
  generic hub mediators, negated effects, and ambiguous multi-anchor findings are filtered before
  ranking.
- **Experiment-oriented output.** Candidates are phrased as predictions with proposed CITE-seq or
  RNA readouts, not as validated discoveries.
- **Auditability.** Reports include JSONL, CSV, Markdown, GraphML, Mermaid traces, diagnostics, and
  skipped-finding records.

## Pipeline

```text
evidence records
  -> deterministic frame extraction
  -> source grounding
  -> typed evidence graph
  -> motif-based candidate generation
  -> biological and provenance filters
  -> transparent multi-factor ranking
  -> JSONL, CSV, Markdown, GraphML, Mermaid traces, diagnostics
```

## Candidate Motifs

| Motif | Evidence shape | Candidate produced |
|---|---|---|
| `perturbation_to_marker_state_completion` | `perturbation -> marker`, `marker -> cell state/type/signature/phenotype` | Predicts that the perturbation changes a marker-associated readout in the matched cell context. |
| `disease_signature_reversal` | `disease -> signature`, `perturbation -> signature` with opposite directions | Predicts perturbations that reverse a disease-associated transcriptional signature. |
| `signature_consolidation` | `perturbation -> gene1...geneN`, where genes belong to a known signature | Consolidates gene-level perturbation evidence into a signature-level prediction. |
| `cite_seq_marker_bridge` | `perturbation -> RNA`, where RNA maps to an ADT marker | Proposes paired RNA/protein measurement while preserving an explicit bridge warning. |

## Ranking

Candidates are scored with an interpretable weighted model:

| Score component | Purpose |
|---|---|
| `grounding_score` | Uses the lowest-confidence supporting frame, preventing one strong citation from masking a fragile link. |
| `context_match_score` | Rewards matched species, tissue, and cell-type context; penalizes broad or partial matches. |
| `measurability_score` | Favors hypotheses with direct RNA, ADT, or signature readouts. |
| `evidence_design_score` | Distinguishes interventional evidence from lower-evidence observational support. |
| `local_nonredundancy_score` | Downranks candidates that closely restate input corpus edges. |
| `specificity_score` | Rewards concrete perturbation, readout, and cell-context specificity. |
| `risk_penalty` | Penalizes bridge assumptions, context mismatch, generic mediators, and other caveats. |

This makes ranking inspectable rather than opaque: a reviewer can see why a candidate rose or fell.

## Representative Candidate

Input evidence:

```json
{
  "paper_id": "paper_A",
  "findings": [
    {
      "statement": "IFN-beta increased FOXP3 in regulatory T cells",
      "source_quote": "IFN-beta increased FOXP3 expression in regulatory T cells."
    }
  ],
  "perturbations": [{"name": "IFN-beta"}],
  "genes": [{"symbol": "FOXP3", "species": "human"}],
  "cell_types": [{"name": "regulatory T cell"}],
  "species": ["human"],
  "tissue": ["PBMC"],
  "relevance": "HIGH"
}
```

When combined with grounded marker evidence that FOXP3 and CD25 mark regulatory T cells, the system
generates a ranked candidate:

```text
HYP-00001
Motif: perturbation_to_marker_state_completion
Score: 0.753

Claim:
In human PBMC, IFN-beta is predicted to increase a FOXP3-associated readout
in regulatory T cell.

Evidence path:
IFN-beta --INCREASES_READOUT--> FOXP3
FOXP3 --DEFINES_OR_MARKS--> regulatory T cell

Proposed test:
CITE-seq in human PBMC regulatory T cells.
Readouts: FOXP3 RNA; CD25 ADT.

Warning:
CD25 ADT is inferred through marker-gene mapping; protein change is not directly supported.
```

The candidate is intentionally phrased as a measurable prediction. The system does not claim global
biological novelty; it identifies a locally novel, evidence-connected experiment to prioritize.

## Scientific Validity Controls

- The generated hypothesis is never treated as quote-entailable evidence; only its support path is
  grounded against source text.
- Novelty is local to the input corpus, not a claim that the hypothesis has never appeared in the
  broader literature.
- Observational support is represented as association, not causation.
- Cross-species and incompatible cell-lineage compositions are blocked by default.
- RNA and surface-protein claims remain distinct; RNA-to-ADT bridges require explicit warnings.
- Negated effects such as "did not increase" are handled as no-change frames.
- Findings with multiple plausible anchors abstain rather than attaching a readout to the wrong
  perturbation or disease.
- Zero-candidate runs still produce diagnostics, making abstention an auditable outcome.

## Evaluation Surface

The repository includes offline checks for the core scientific-AI behaviors:

- frame extraction against hand-labeled gold frames;
- hidden-edge recovery with Recall@k and MRR;
- seeded null-model comparisons using sign, context, and evidence-mode shuffles;
- regression tests for motif coverage, provenance filters, context blocking, bridge warnings,
  negation handling, deterministic ranking, and artifact generation.

The evaluation design emphasizes failure modes that matter for AI scientist work: unsupported
claims, context leakage, biologically invalid joins, non-measurable predictions, and brittle ranking.

## Run

```bash
mamba run -n litstream-lg python -m litstream.hypotheses run \
  --evidence-dir tests/fixtures/hypotheses/evidence \
  --out-dir /tmp/litstream_hypotheses_example \
  --grounder overlap
```

Installed console script:

```bash
mamba run -n litstream-lg litstream-hypothesize \
  --evidence-dir tests/fixtures/hypotheses/evidence \
  --out-dir /tmp/litstream_hypotheses_example
```

Outputs:

```text
hypotheses.jsonl
hypotheses.csv
hypotheses.md
diagnostics.json
skipped_findings.csv
hypothesis_graph.graphml
figures/
```

## Evaluate

```bash
mamba run -n litstream-lg python -m litstream.hypotheses eval frames \
  --gold tests/fixtures/hypotheses/gold_frames.jsonl \
  --evidence-dir tests/fixtures/hypotheses/evidence

mamba run -n litstream-lg python -m litstream.hypotheses eval hidden-edge \
  --evidence-dir tests/fixtures/hypotheses/evidence

mamba run -n litstream-lg python -m litstream.hypotheses eval null \
  --evidence-dir tests/fixtures/hypotheses/evidence \
  --seed 0
```

## Tests

```bash
mamba run -n litstream-lg python -m pytest tests/test_hypotheses.py -q
```

## Library Use

```python
from litstream.hypotheses import ContextBoundHypothesisGenerator, HypothesisConfig, run_to_dir

config = HypothesisConfig()
result = ContextBoundHypothesisGenerator(config).run(evidence_records)
summary = run_to_dir("tests/fixtures/hypotheses/evidence", "/tmp/litstream_hypotheses_example", config)
```
