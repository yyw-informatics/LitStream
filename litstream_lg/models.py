"""Model routing and cost capture for pipeline model calls.

  - Routing resolves a phase's model name into a LangChain `ChatAnthropic` /
    `ChatOpenAI` that plugs straight into `create_agent`.
  - Cost capture writes token usage to the shared SQLite ledger after each LLM call.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from .routing import resolve, price_for  # noqa: F401  (price_for re-exported)

# Large evidence-file tool calls need enough output room to complete in one response.
_MAX_TOKENS = 8192


class BudgetExceeded(Exception):
    """Raised when the run's accrued cost crosses the configured per-run cap."""


def make_chat_model(model_name: str, *, callbacks: list | None = None,
                    max_tokens: int = _MAX_TOKENS):
    """Resolve a phase's model name to a LangChain chat model (Anthropic or any
    OpenAI-compatible endpoint)."""
    spec = resolve(model_name)
    if spec.kind == "anthropic":
        return ChatAnthropic(model=spec.model, api_key=spec.api_key or "missing",
                             max_tokens=max_tokens, callbacks=callbacks)
    kwargs: dict = {"model": spec.model, "api_key": spec.api_key or "missing",
                    "callbacks": callbacks}
    if spec.base_url:
        kwargs["base_url"] = spec.base_url
    if spec.reasoning:
        # Reasoning models need extra output headroom for hidden reasoning tokens.
        kwargs["max_tokens"] = max(max_tokens, 16000)
    else:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def caching_middleware(model_name: str) -> list:
    """Prompt-caching middleware for a model, so re-sent context isn't re-billed every
    agent turn (the static skill system prompt, tool schemas, and the paper dominate
    each call).

    Anthropic needs explicit cache breakpoints, so the middleware tags the system
    message, the tool definitions, and a rolling message prefix with `cache_control`.
    OpenAI-compatible providers cache automatically server-side and need no middleware;
    their cached tokens still flow through LedgerCallbackHandler via usage_metadata.
    ttl='5m' matches the ledger's 1.25x cache-creation pricing (a 1h cache would write
    at 2x and under-bill).
    """
    if resolve(model_name).kind == "anthropic":
        return [AnthropicPromptCachingMiddleware(ttl="5m", unsupported_model_behavior="ignore")]
    return []


class LedgerCallbackHandler(BaseCallbackHandler):
    """Records every LLM call's token usage into the cost ledger.

    `on_llm_end` reads LangChain's normalized `usage_metadata` and debits the ledger,
    pricing regular input separately from cache-read and cache-creation."""

    # Required so budget exceptions propagate out of the model call.
    raise_error = True

    def __init__(self, ledger, run_id: str, *, model: str, phase: str | None = None,
                 role: str = "langgraph", cap_cents: float | None = None):
        self.ledger = ledger
        self.run_id = run_id
        self.model = model
        self.phase = phase
        self.role = role
        self.cap_cents = cap_cents
        self.calls = 0
        self.cost_cents = 0.0

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
        try:
            for gens in getattr(response, "generations", []) or []:
                for gen in gens:
                    msg = getattr(gen, "message", None)
                    um = getattr(msg, "usage_metadata", None) if msg else None
                    if not um:
                        continue
                    inp = int(um.get("input_tokens", 0) or 0)
                    out = int(um.get("output_tokens", 0) or 0)
                    details = um.get("input_token_details", {}) or {}
                    cache_read = int(details.get("cache_read", 0) or 0)
                    cache_creation = int(details.get("cache_creation", 0) or 0)
                    regular = max(inp - cache_read - cache_creation, 0)
                    self.calls += 1
                    self.cost_cents += self.ledger.record(
                        self.run_id, self.model, phase=self.phase, role=self.role,
                        input_tokens=regular, output_tokens=out,
                        cached_input_tokens=cache_read, cache_creation_tokens=cache_creation)
        except Exception as exc:   # cost capture is best-effort — a ledger write must not abort a run
            print(f"[ledger] usage record failed: {type(exc).__name__}: {exc}", flush=True)
        if self.cap_cents is not None and self.ledger.run_cost_cents(self.run_id) >= self.cap_cents:
            raise BudgetExceeded(
                f"run cost ${self.ledger.run_cost_cents(self.run_id)/100:.2f} reached cap "
                f"${self.cap_cents/100:.2f} during phase '{self.phase}'")
