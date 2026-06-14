"""routing.resolve / price_for and models.make_chat_model.

Provider resolution maps a model name to a Spec (anthropic vs openai_compat, base_url,
price); make_chat_model turns that Spec into a LangChain chat model. We assert
type/attributes only — the model is never invoked, so no network or real key is needed.
"""

from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from litstream_lg.models import make_chat_model
from litstream_lg.routing import KNOWN_PRICING, price_for, resolve


# ---------------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------------

def test_resolve_claude_is_anthropic_with_known_pricing():
    spec = resolve("claude-sonnet-4-6")
    assert spec.kind == "anthropic"
    assert spec.model == "claude-sonnet-4-6"
    assert spec.base_url == "https://api.anthropic.com/v1"
    assert spec.price == KNOWN_PRICING["claude-sonnet-4-6"] == (3.00, 15.00, 0.30)
    assert spec.reasoning is False


def test_resolve_claude_unknown_pricing_is_none_but_still_anthropic():
    # A claude-* name not in KNOWN_PRICING still resolves to anthropic, price None.
    spec = resolve("claude-something-unlisted")
    assert spec.kind == "anthropic"
    assert spec.price is None


def test_resolve_task_models_deepseek_is_openai_compat():
    spec = resolve("deepseek")               # matches by task_models.yaml `name`
    assert spec.kind == "openai_compat"
    assert spec.model == "deepseek-chat"
    assert spec.base_url == "https://api.deepseek.com/v1"
    assert spec.price == (0.27, 1.10, 0.07)  # from price_usd_per_mtok in the yaml
    assert spec.reasoning is False


def test_resolve_matches_by_model_id_too():
    # task_models entries match either `name` or `model`.
    spec = resolve("deepseek-chat")
    assert spec.kind == "openai_compat"
    assert spec.model == "deepseek-chat"


def test_resolve_reasoning_model_flag():
    spec = resolve("gpt-5")
    assert spec.kind == "openai_compat"
    assert spec.reasoning is True


def test_resolve_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="unknown model/provider"):
        resolve("totally-made-up-model-xyz")


# ---------------------------------------------------------------------------
# price_for()
# ---------------------------------------------------------------------------

def test_price_for_claude_returns_known_tuple():
    assert price_for("claude-haiku-4-5") == (1.00, 5.00, 0.10)


def test_price_for_deepseek_returns_yaml_tuple():
    assert price_for("deepseek") == (0.27, 1.10, 0.07)


def test_price_for_local_model_is_zero():
    assert price_for("local-llama") == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# make_chat_model()  — construct only, never invoke
# ---------------------------------------------------------------------------

def test_make_chat_model_claude_returns_chat_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    model = make_chat_model("claude-sonnet-4-6")
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-4-6"
    assert model.max_tokens == 8192      # agentic default: room for a full evidence write


def test_make_chat_model_claude_missing_key_still_constructs(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    model = make_chat_model("claude-haiku-4-5")        # falls back to "missing" sentinel
    assert isinstance(model, ChatAnthropic)


def test_make_chat_model_deepseek_returns_chat_openai_with_base_url(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    model = make_chat_model("deepseek")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "deepseek-chat"
    # base_url is exposed as openai_api_base on ChatOpenAI.
    assert str(model.openai_api_base).rstrip("/") == "https://api.deepseek.com/v1"
    assert model.max_tokens == 8192


def test_make_chat_model_reasoning_model_gets_larger_max_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # default max_tokens 8192 < 16000 -> reasoning path bumps it to 16000.
    model = make_chat_model("gpt-5")
    assert isinstance(model, ChatOpenAI)
    assert model.max_tokens == 16000


def test_make_chat_model_passes_callbacks_through(monkeypatch):
    from langchain_core.callbacks import BaseCallbackHandler

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    sentinel = BaseCallbackHandler()       # callbacks are pydantic-validated to this type
    model = make_chat_model("claude-haiku-4-5", callbacks=[sentinel])
    assert sentinel in (model.callbacks or [])
