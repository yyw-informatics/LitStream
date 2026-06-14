# Model Routing And Benchmarks

LitStream is designed so each task can use the model that fits its cost, latency, and accuracy needs.
The important feature is not a single preferred model; it is the ability to benchmark alternatives and
mix them by phase.

## Routing Surfaces

- Agentic LangGraph phases use `litstream_lg.models.make_chat_model`. Routine YAML files can assign
  different models to `mine`, `synthesize`, `evaluate`, and `design`.
- Single-shot tasks use `litstream.tasks.models`, a small provider interface for local Ollama/vLLM,
  DeepSeek, OpenAI-compatible APIs, and Anthropic.
- `litstream/config/task_models.yaml` enables or disables providers, stores model IDs, records API key
  environment variables, and seeds token pricing for the ledger.
- Routine files, such as `litstream/config/routines/weekly-citeseq.yaml`, combine schedule, project
  scope, source queries, phases, model assignments, and run limits.

## Benchmark Tools

| Tool | What It Checks |
|---|---|
| `litstream.eval.triage_eval` | Labeled triage accuracy and KEEP precision/recall |
| `litstream.eval.cost_vs_performance` | Cost per 1,000 screened papers versus accuracy |
| `litstream.eval.provider_bakeoff` | Side-by-side provider comparison for the mining task |
| `litstream.eval.run_variance` | Run-to-run variance for model outputs |
| `litstream.eval.mslr_eval` | Experimental-study summarization quality against expert gold summaries |

## How The Mix-And-Match Pattern Is Used

High-volume tasks, such as triage and per-paper mining, can run on cheaper or local models. Integrative
tasks, such as synthesis and experimental design, can use stronger hosted models. The benchmark tools
make those choices empirical: compare candidate backends on the same inputs, inspect quality and
variance, then update the routine or task-model config.

Every model call returns usage metadata when the provider exposes it. LitStream records those calls in
the SQLite ledger, including regular input tokens, cached input tokens, cache creation tokens, output
tokens, model name, phase, and role. That makes scheduled runs easier to audit after the fact.
