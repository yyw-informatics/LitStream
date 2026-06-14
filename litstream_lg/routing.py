"""Provider resolution and pricing.

A self-contained resolver: it reads task_models.yaml and produces a plain `Spec` that
`models.py` turns into a LangChain chat model. Keeping it in this package keeps the
dependency footprint free of any agent SDK.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import litstream
import yaml

_CONFIG = Path(litstream.__file__).resolve().parent / "config"

KNOWN_PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00, 0.10),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
    "claude-sonnet-4-6": (3.00, 15.00, 0.30),
    "claude-fable-5": (5.00, 25.00, 0.50),
    "claude-opus-4-8": (15.00, 75.00, 1.50),
    "deepseek-chat": (0.27, 1.10, 0.07),
    "llama3.1:8b": (0.0, 0.0, 0.0),
    "qwen2.5:7b": (0.0, 0.0, 0.0),
}


@dataclass
class Spec:
    kind: str               # "anthropic" | "openai_compat"
    base_url: str
    model: str
    api_key: str
    price: tuple | None = None      # (input, output, cached) USD/Mtok
    reasoning: bool = False


def resolve(model_name: str) -> Spec:
    """Map a phase's model name to a provider spec (Claude → Anthropic; anything in
    task_models.yaml → its OpenAI-compatible endpoint)."""
    if model_name.startswith("claude"):
        return Spec("anthropic", "https://api.anthropic.com/v1", model_name,
                    os.environ.get("ANTHROPIC_API_KEY", ""), price=KNOWN_PRICING.get(model_name))
    cfg = yaml.safe_load((_CONFIG / "task_models.yaml").read_text())
    for s in cfg["task_models"]:
        if model_name in (s["model"], s["name"]):
            key = os.environ.get(s["api_key_env"], "") if s.get("api_key_env") else s.get("api_key", "")
            p = s.get("price_usd_per_mtok")
            price = (p["input"], p["output"], p["cached_input"]) if p else None
            return Spec(s["kind"], s.get("base_url", ""), s["model"], key,
                        price=price, reasoning=s.get("reasoning", False))
    raise ValueError(f"unknown model/provider: {model_name!r}")


def price_for(model_name: str) -> tuple | None:
    """USD/Mtok (input, output, cached) for a model, to seed the ledger."""
    spec = resolve(model_name)
    return spec.price or KNOWN_PRICING.get(spec.model)
