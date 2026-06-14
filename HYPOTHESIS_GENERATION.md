# Context-Bound Hypothesis Generation

## Approach

- **Objective.** Generate ranked, testable hypothesis candidates from grounded biomedical evidence records,
  with emphasis on literature-driven single-cell biology.
- **Structured evidence.** Findings are normalized into biomedical frames and
  composed through typed graph motifs instead of model-invented mechanisms.
- **Grounded constraints.** Candidate construction uses quote-grounded support frames, while
  cross-species joins, incompatible cell lineages, generic mediators, negated effects, and ambiguous
  anchors are blocked before ranking.
- **Interpretable ranking.** Candidates are scored by grounding, context compatibility, measurability,
  evidence design, nonredundancy, specificity, and risk.
- **Experimental traceability.** Outputs are measurable predictions, not validated discoveries,
  with evidence paths, assumptions, warnings, diagnostics, JSONL, CSV, Markdown, GraphML, and Mermaid
  traces preserved.

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

### Example Score Calculation

The score is a prioritization rank, not a probability that the hypothesis is true. The ranker combines
weighted evidence, context, measurability, design, novelty, and specificity terms, then subtracts
explicit risk penalties:

```text
rank_score =
  0.30 * grounding_score
+ 0.20 * context_match_score
+ 0.20 * measurability_score
+ 0.15 * evidence_design_score
+ 0.10 * local_nonredundancy_score
+ 0.05 * specificity_score
- risk_penalty
```

For this candidate:

| Component | Value | Weight | Contribution | Rationale |
|---|---:|---:|---:|---|
| `grounding_score` | 1.000 | 0.30 | 0.300 | Supporting frames are grounded to source quotes. |
| `context_match_score` | 1.000 | 0.20 | 0.200 | Species, tissue, cell type, and perturbation context are specified. |
| `measurability_score` | 1.000 | 0.20 | 0.200 | FOXP3 RNA and paired CD25 ADT are measurable by CITE-seq. |
| `evidence_design_score` | 1.000 | 0.15 | 0.150 | The IFN-beta -> FOXP3 support edge is interventional. |
| `local_nonredundancy_score` | 0.700 | 0.10 | 0.070 | The candidate is not a direct restatement, but reuses corpus entities. |
| `specificity_score` | 0.667 | 0.05 | 0.033 | Cell type, perturbation, readout, and tissue are named; comparator and signature are absent. |
| `risk_penalty` | 0.200 | - | -0.200 | CD25 ADT is an RNA-to-surface-protein bridge and requires a warning. |

```text
subtotal = 0.300 + 0.200 + 0.200 + 0.150 + 0.070 + 0.033 = 0.953
rank_score = 0.953 - 0.200 = 0.753
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
