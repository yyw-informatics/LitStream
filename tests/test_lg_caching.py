"""Prompt-caching wiring — Anthropic gets explicit cache breakpoints, OpenAI-style
providers get none (they cache automatically server-side)."""

from __future__ import annotations

from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

from litstream_lg.models import caching_middleware


def test_anthropic_model_gets_caching_middleware(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mw = caching_middleware("claude-sonnet-4-6")
    assert len(mw) == 1
    assert isinstance(mw[0], AnthropicPromptCachingMiddleware)


def test_openai_compatible_model_gets_no_middleware(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    assert caching_middleware("deepseek") == []        # DeepSeek caches automatically


def test_local_model_gets_no_middleware():
    assert caching_middleware("qwen2.5:7b") == []
