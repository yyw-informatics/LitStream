"""litstream_lg — LitStream's autonomous research pipeline on LangGraph.

The orchestration and LLM layer is built on LangGraph + LangChain:
  - the phase pipeline            → a LangGraph StateGraph with conditional budget edges
  - the agentic tool-use executor → langchain.agents.create_agent (prebuilt ReAct)
  - the file tools                → LangChain StructuredTools
  - per-phase model routing       → ChatAnthropic / ChatOpenAI via the shared resolver
  - cost capture                  → a LangChain BaseCallbackHandler
  - phase resume                  → a LangGraph SqliteSaver checkpointer

Framework-neutral plumbing is reused from the `litstream` package: the SQLite cost
ledger, the content-addressed paper library + dedup, the HTTP source clients + triage +
source-mute policy, PDF fetch, the digest/notify delivery, the cron/launchd scheduler,
the kb-skills prompt rendering, and the eval harness.
"""
