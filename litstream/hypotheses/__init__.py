"""LitStream hypothesis-candidate generation (ContextBoundHypothesisGenerator v0.1).

Opt-in, offline, conservative: turns already-grounded per-paper evidence records into ranked, locally
non-redundant, source-grounded, testable hypothesis *candidates* — never a claim of biological
discovery. Importable with no GPU / network (networkx for the graph, optional matplotlib for figures).

    from litstream.hypotheses import ContextBoundHypothesisGenerator, HypothesisConfig, run_to_dir
"""

from __future__ import annotations

from .config import HypothesisConfig
from .pipeline import ContextBoundHypothesisGenerator, filter_records_by_relevance, run_to_dir
from .schema import (
    BioContext, Entity, EvidenceEdge, FindingFrame, HypothesisCandidate, HypothesisRunResult,
)

__all__ = [
    "HypothesisConfig",
    "ContextBoundHypothesisGenerator",
    "run_to_dir",
    "filter_records_by_relevance",
    "Entity",
    "BioContext",
    "FindingFrame",
    "EvidenceEdge",
    "HypothesisCandidate",
    "HypothesisRunResult",
]
