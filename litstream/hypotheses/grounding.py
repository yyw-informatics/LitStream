"""Frame grounding — verify each finding frame against its own source quote.

We do NOT verify the novel hypothesis (a hypothesis is, by construction, not entailed by any quote);
we verify the *support* frames only. Premise = ``frame.source_quote``; claim = ``frame.atomic_claim``.

Two backends:

* ``overlap`` (default, offline) — :class:`LexicalFrameVerifier`, a stemming-tolerant content-word
  overlap that drops report-scaffold words ("the paper reports that") and still *requires every number
  in the claim to appear in the passage* (the number check is what catches invented quantities — the
  same contract as the shared :class:`litstream_evidence.ground_retrieval.OverlapVerifier`). Tolerant
  of morphology (marks/marked, cell/cells) because the deterministic atomic claim re-words the quote.
* ``minicheck`` — wraps the shared find-then-verify entailment cascade.

Score convention: *supported -> 1.0* (so a real entailment clears ``min_grounding_score`` regardless
of backend); a failed frame keeps its low raw overlap so the gate can drop it and surface why.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .config import HypothesisConfig
from .schema import FindingFrame

_STOP = {"the", "and", "for", "with", "were", "was", "are", "this", "that", "from", "into",
         "which", "not", "have", "has", "had", "used", "using", "our", "your", "per", "paper",
         "reports", "report", "reported", "study", "shows", "showed", "show", "found", "observed",
         "was", "compared", "relative", "than", "between", "within"}


def _stem(tok: str) -> str:
    """Crude suffix stripper so marks/marked/marking and cell/cells collapse."""
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def _content_stems(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {_stem(t) for t in toks if len(t) >= 3 and t not in _STOP}


def _numbers(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])", text or "")


class LexicalFrameVerifier:
    """Offline, model-free: supported iff enough of the claim's (stemmed) content words appear in the
    passage AND every standalone number in the claim appears in it."""

    def __init__(self, min_overlap: float = 0.5):
        self.min_overlap = min_overlap

    def verify(self, claim: str, passage: str) -> tuple[bool, float]:
        cw = _content_stems(claim)
        if not cw:
            return (False, 0.0)
        overlap = len(cw & _content_stems(passage)) / len(cw)
        plow = (passage or "").lower()
        numbers_ok = all(n in plow for n in _numbers(claim))
        return (overlap >= self.min_overlap and numbers_ok, round(overlap, 3))


class StubGrounder:
    """Deterministic grounder: every frame is entailed with score 1.0."""

    name = "stub"

    def verify(self, frame: FindingFrame) -> FindingFrame:
        return replace(frame, grounding_label="entailed", grounding_score=1.0)


class FrameGrounder:
    """Adapter over any ``verify(claim, passage) -> (bool, float)`` verifier."""

    def __init__(self, verifier: Any, name: str = "overlap"):
        self.verifier = verifier
        self.name = name

    def verify(self, frame: FindingFrame) -> FindingFrame:
        claim = frame.atomic_claim or frame.raw_statement
        premise = frame.source_quote or ""
        if not premise.strip():
            return replace(frame, grounding_label="unknown", grounding_score=0.0,
                           warnings=frame.warnings + ("no_source_quote",))
        supported, raw = self.verifier.verify(claim, premise)
        if supported:
            return replace(frame, grounding_label="entailed", grounding_score=1.0)
        return replace(frame, grounding_label="contradicted", grounding_score=round(float(raw), 3),
                       warnings=frame.warnings + (f"grounding_overlap={round(float(raw), 3)}",))


def make_grounder(name: str = "overlap", minicheck_model: str = "flan-t5-large"):
    """``stub`` (always-entailed), ``overlap``/``lexical`` (offline stemming check, default), or
    ``minicheck`` (entailment model via the shared cascade — heavy, lazy-loaded)."""
    if name == "stub":
        return StubGrounder()
    if name in ("overlap", "lexical"):
        return FrameGrounder(LexicalFrameVerifier(), name="overlap")
    if name == "minicheck":
        from litstream_evidence.ground_retrieval import make_verifier
        return FrameGrounder(make_verifier("minicheck", minicheck_model), name="minicheck")
    raise ValueError(f"unknown grounder {name!r} (use 'stub', 'overlap', or 'minicheck')")


def verify_frames(
    frames: list[FindingFrame], config: HypothesisConfig, grounder: Any | None = None
) -> tuple[list[FindingFrame], dict]:
    """Ground every frame, then drop the ones that fail the grounding gate. Returns (kept, diag)."""
    grounder = grounder or make_grounder(config.grounder)
    kept: list[FindingFrame] = []
    skipped: list[dict] = []
    n_entailed = n_contradicted = n_unknown = 0
    for fr in frames:
        g = grounder.verify(fr)
        if g.grounding_label == "entailed":
            n_entailed += 1
        elif g.grounding_label == "contradicted":
            n_contradicted += 1
        else:
            n_unknown += 1
        if config.require_grounded_frames and g.grounding_label != "entailed":
            skipped.append({"paper_id": g.paper_id, "frame_id": g.frame_id,
                            "finding_text": g.raw_statement, "reason": "grounding_failed",
                            "source_quote": g.source_quote})
            continue
        if g.grounding_score < config.min_grounding_score:
            skipped.append({"paper_id": g.paper_id, "frame_id": g.frame_id,
                            "finding_text": g.raw_statement, "reason": "grounding_below_threshold",
                            "source_quote": g.source_quote})
            continue
        kept.append(g)
    diag = {
        "grounder": getattr(grounder, "name", str(grounder)),
        "frames_in": len(frames),
        "frames_grounded": len(kept),
        "entailed": n_entailed,
        "contradicted": n_contradicted,
        "unknown": n_unknown,
        "skipped": skipped,
    }
    return kept, diag
