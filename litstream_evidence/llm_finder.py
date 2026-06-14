"""LLM finder — the escalation tier of grounding.

Once the routed cheap stack (presence for entities, MiniCheck for numbers) leaves a FLAGGED set, a
mid-tier LLM gets one more pass — but as a FINDER, never a judge. For each flagged item it reads the
paper and returns the shortest VERBATIM span that states the claim, or UNSUPPORTED. That span is then
(1) checked to be a real substring of the paper (catching an LLM-invented quote) and (2) re-judged by
the SAME routed verifier (presence for entities, MiniCheck for numbers). The item is rescued only if
both pass. So the LLM can surface a passage the embedding retriever missed, but it cannot bless a
fabrication: MiniCheck stays the numeric authority, so invented frequencies the cheap stack already
caught cannot be resurrected. This targets the common cause of flags — retrieval miss or paraphrase —
without re-opening the door to hallucination.

`make_finder('llm')` builds the default mid-tier chat model lazily.
"""

from __future__ import annotations

import re
from typing import Protocol

from litstream_evidence.ground_retrieval import Verifier, _item_text, _needs_strict


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().casefold()


def quote_in_source(quote: str, source_text: str) -> bool:
    """Is the LLM's proposed span a real substring of the paper (whitespace/case-insensitive)?
    The guard that stops the LLM rescuing anything by inventing a quote."""
    q = _norm(quote)
    return bool(q) and q in _norm(source_text)


class Finder(Protocol):
    def find(self, claim: str, source_text: str) -> str: ...   # verbatim span, or '' for UNSUPPORTED


def _support_model():
    """Lazy Pydantic model: the LLM returns a verdict + the verbatim supporting span."""
    from typing import Literal

    from pydantic import BaseModel, Field

    class Support(BaseModel):
        verdict: Literal["SUPPORTED", "UNSUPPORTED"] = Field(
            description="SUPPORTED only if the PAPER TEXT explicitly states the claim; else UNSUPPORTED.")
        quote: str = Field(
            default="",
            description="If SUPPORTED, the shortest span copied VERBATIM (character-for-character) from "
                        "the PAPER TEXT that states the claim. Empty if UNSUPPORTED.")

    return Support


_FIND_PROMPT = (
    "You are a strict grounding checker for a scientific paper.\n"
    "Decide whether the PAPER TEXT EXPLICITLY states the CLAIM.\n"
    "Judge ONLY what the paper says. Do NOT use outside knowledge. Do NOT judge whether the claim is "
    "true in general — only whether THIS paper states it.\n"
    "If it does, copy the SHORTEST span from the PAPER TEXT, VERBATIM (character-for-character), that "
    "states it. If it does not, answer UNSUPPORTED with an empty quote.\n\n"
    "CLAIM: {claim}\n\n=== PAPER TEXT ===\n{paper}\n"
)


class LLMFinder:
    """LLM finder: a LangChain chat model with `.with_structured_output(Support)`. Returns the
    model's verbatim span when it says SUPPORTED, else '' — the downstream substring and verifier
    checks (in `apply_llm_finder`) decide whether that span actually rescues the item."""

    def __init__(self, model, max_chars: int = 45_000):
        self.structured = model.with_structured_output(_support_model())
        self.max_chars = max_chars

    def find(self, claim: str, source_text: str) -> str:
        res = self.structured.invoke(
            _FIND_PROMPT.format(claim=claim, paper=(source_text or "")[:self.max_chars]))
        if isinstance(res, dict):
            verdict, quote = res.get("verdict", ""), res.get("quote", "")
        else:
            verdict, quote = getattr(res, "verdict", ""), getattr(res, "quote", "")
        return (quote or "").strip() if str(verdict).upper() == "SUPPORTED" else ""


def _default_finder_model(model_name: str = "claude-sonnet-4-6"):
    try:
        from litstream_lg.models import make_chat_model
    except Exception as exc:
        raise RuntimeError("the LLM finder needs the LangChain stack (litstream_lg + "
                           "langchain-anthropic). Install it, or pass your own model.") from exc
    return make_chat_model(model_name)


def make_finder(name: str = "llm", model=None, model_name: str = "claude-sonnet-4-6") -> Finder:
    if name == "llm":
        return LLMFinder(model if model is not None else _default_finder_model(model_name))
    raise ValueError(f"unknown finder {name!r} (use 'llm')")


def apply_llm_finder(record: dict, report: dict, source_text: str, finder: Finder,
                     verifier: Verifier, value_verifier: Verifier | None = None) -> dict:
    """Second chance for each FLAGGED item: ask `finder` for a verbatim supporting span, then rescue
    the item only if the span is a real substring of the paper AND the routed verifier (presence for
    entities, `value_verifier` for numeric claims) judges the claim supported by that span. Mutates
    `record` (rescued items get the real span as `source_quote`) and `report` (counts move from
    flagged to grounded; adds a `rescued` list). Returns the updated report."""
    value_verifier = value_verifier or verifier
    still_flagged: list[dict] = []
    rescued: list[dict] = []
    for entry in report.get("flagged_items", []):
        field, item = entry["field"], entry["item"]
        claim = _item_text(item)
        span = finder.find(claim, source_text)
        v = value_verifier if _needs_strict(field, claim) else verifier
        if span and quote_in_source(span, source_text) and v.verify(claim, span)[0]:
            item["source_quote"] = re.sub(r"\s+", " ", span).strip()[:240]
            bf = report["by_field"][field]
            bf["grounded"] += 1
            bf["flagged"] -= 1
            report["grounded"] += 1
            report["flagged"] -= 1
            rescued.append(entry)
        else:
            still_flagged.append(entry)
    report["flagged_items"] = still_flagged
    report["rescued"] = rescued
    return report
