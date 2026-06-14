"""Task-model router for non-agentic, single-shot pipeline tasks.

These are the cheap front-of-pipeline jobs — relevance triage, metadata
extraction, reranking — that don't need the agentic skill harness. Each is just
"prompt -> text + token usage", so one thin interface covers every backend and
lets them be compared on cost and quality:

    1. local / free   — Ollama or vLLM (OpenAI-compatible) → open HF models
    2. DeepSeek       — OpenAI-compatible API, very cheap
    3. cheaper hosted — OpenAI mini model, or Claude Haiku

Stdlib only (urllib). Every call returns token usage so it can be recorded in
the cost ledger; local models price at $0, the rest compute from model_pricing.
Kept separate from the agentic pipeline driver, which needs the full Claude
Agent SDK.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass
class TaskResult:
    text: str
    model: str
    input_tokens: int = 0           # REGULAR (non-cached) input
    output_tokens: int = 0
    cached_input_tokens: int = 0     # cache read (discounted)
    cache_creation_tokens: int = 0   # cache write (Claude bills at 1.25x input)
    latency_ms: int = 0


def _parse_wait(h) -> float | None:
    """Seconds to wait from a 429 response's headers (Retry-After or reset-tokens)."""
    ra = h.get("Retry-After")
    if ra and ra.replace(".", "", 1).isdigit():
        return float(ra)
    reset = h.get("x-ratelimit-reset-tokens") or h.get("x-ratelimit-reset-requests")
    if reset:
        m = re.fullmatch(r"(?:(\d+)m)?(?:([\d.]+)s)?(?:(\d+)ms)?", reset.strip())
        if m:
            mins, secs, ms = m.groups()
            return (int(mins or 0) * 60) + float(secs or 0) + (int(ms or 0) / 1000)
    return None


def _http_post(url: str, headers: dict, payload: dict, timeout: float = 120.0,
               retries: int = 4) -> dict:
    """POST JSON, return parsed JSON. Retries 429s with header-aware backoff — TPM
    limits reset over ~60s, so we honor Retry-After / x-ratelimit-reset-* (clamped
    2-65s) rather than a fixed short sleep. Isolated so tests can monkeypatch it."""
    data = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={**headers, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                wait = _parse_wait(e.headers) or (15 * (attempt + 1))
                time.sleep(min(max(wait, 2.0), 65.0))
                continue
            raise


class TaskModel(Protocol):
    name: str
    model: str
    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> TaskResult: ...


class OpenAICompatModel:
    """Covers DeepSeek, OpenAI, and local Ollama/vLLM — they share the schema.

    base_url is the API root including /v1 (e.g. https://api.deepseek.com/v1,
    https://api.openai.com/v1, http://localhost:11434/v1). api_key may be a
    placeholder for local servers that ignore it.
    """

    def __init__(self, name: str, base_url: str, model: str, api_key: str = "",
                 reasoning: bool = False):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.reasoning = reasoning

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> TaskResult:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        payload: dict = {"model": self.model, "messages": messages}
        if self.reasoning:
            payload["max_completion_tokens"] = max(max_tokens, 4000)
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        t0 = time.monotonic()
        r = _http_post(f"{self.base_url}/chat/completions", headers, payload)
        dt = int((time.monotonic() - t0) * 1000)
        usage = r.get("usage", {}) or {}
        # DeepSeek surfaces cache hits; OpenAI nests them; both optional.
        cached = usage.get("prompt_cache_hit_tokens") \
            or (usage.get("prompt_tokens_details", {}) or {}).get("cached_tokens", 0) or 0
        prompt = usage.get("prompt_tokens", 0)
        return TaskResult(
            text=r["choices"][0]["message"]["content"],
            model=self.model,
            input_tokens=max(prompt - cached, 0),   # regular (non-cached) portion
            output_tokens=usage.get("completion_tokens", 0),
            cached_input_tokens=cached,              # DeepSeek/OpenAI: cache read only
            latency_ms=dt,
        )


class AnthropicModel:
    """Claude Haiku (or any Claude model) via the Messages API."""

    def __init__(self, name: str, model: str, api_key: str,
                 base_url: str = "https://api.anthropic.com/v1",
                 version: str = "2023-06-01"):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.version = version

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> TaskResult:
        payload = {"model": self.model, "max_tokens": max_tokens,
                   "temperature": temperature,
                   "messages": [{"role": "user", "content": prompt}]}
        if system:
            # Mark the system block cacheable — a big reused skill prefix then gets
            # billed at the discounted cache-read rate on subsequent calls.
            payload["system"] = [{"type": "text", "text": system,
                                  "cache_control": {"type": "ephemeral"}}]
        headers = {"x-api-key": self.api_key, "anthropic-version": self.version}
        t0 = time.monotonic()
        r = _http_post(f"{self.base_url}/messages", headers, payload)
        dt = int((time.monotonic() - t0) * 1000)
        usage = r.get("usage", {}) or {}
        text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
        return TaskResult(
            text=text,
            model=self.model,
            input_tokens=usage.get("input_tokens", 0),           # Claude: already non-cached
            output_tokens=usage.get("output_tokens", 0),
            cached_input_tokens=usage.get("cache_read_input_tokens", 0),
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
            latency_ms=dt,
        )


def build_model(spec: dict, resolve_key) -> TaskModel:
    """Construct a TaskModel from a config dict (see config/task_models.yaml).

    resolve_key(env_name) -> api key string (e.g. os.environ.get). Kept injectable
    so the caller controls how secrets are read.
    """
    kind = spec["kind"]
    key = resolve_key(spec["api_key_env"]) if spec.get("api_key_env") else spec.get("api_key", "")
    if kind == "openai_compat":
        return OpenAICompatModel(spec["name"], spec["base_url"], spec["model"], key or "",
                                 reasoning=spec.get("reasoning", False))
    if kind == "anthropic":
        return AnthropicModel(spec["name"], spec["model"], key or "")
    raise ValueError(f"unknown task-model kind: {kind!r}")


def record_to_ledger(ledger, run_id: str, result: TaskResult, *,
                     phase: str | None = None, role: str | None = None) -> int:
    """Bridge a TaskResult into the cost ledger. Returns cost_cents."""
    return ledger.record(
        run_id, result.model, phase=phase, role=role,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        cached_input_tokens=result.cached_input_tokens,
        cache_creation_tokens=result.cache_creation_tokens,
    )
