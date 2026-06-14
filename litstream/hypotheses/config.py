"""Configuration for the hypothesis-candidate generator. Conservative defaults: grounded frames
required, cross-species / context-transfer / observational-causal-language all off, block-rather-
than-penalize on incompatible context. Override via a YAML file or CLI flags."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

# Generic "hub" mediators a hypothesis must not lean on (a bridge through 'inflammation' or 'T cell'
# is uninformative). Configurable defaults.
DEFAULT_GENERIC_MEDIATORS: tuple[str, ...] = (
    "inflammation",
    "immune response",
    "immune activation",
    "activation",
    "t cell",
    "t cells",
    "pbmc",
    "cytokine response",
    "cellular response",
    "immune cell",
    "immune cells",
    "lymphocyte",
)

_RELEVANCE_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2}


@dataclass
class HypothesisConfig:
    min_paper_relevance: Literal["HIGH", "MODERATE", "LOW"] = "MODERATE"

    require_grounded_frames: bool = True
    min_grounding_score: float = 0.80

    allow_cross_species: bool = False
    allow_context_transfer: bool = False
    allow_low_relevance: bool = False
    allow_observational_causal_language: bool = False

    max_path_length: int = 2
    max_candidates: int = 100
    max_candidates_per_mediator: int = 5
    max_candidates_per_anchor: int = 10

    require_named_cell_type: bool = True
    require_named_readout: bool = True
    require_comparator_for_interventional_claims: bool = False

    # near-duplicate thresholds
    token_jaccard_dup: float = 0.85
    embedding_cosine_dup: float = 0.90

    # signature-consolidation knobs
    signature_min_genes: int = 3
    signature_majority_frac: float = 0.70

    novelty_scope: str = "local_corpus_only"

    write_markdown: bool = True
    write_csv: bool = True
    write_jsonl: bool = True
    write_graphml: bool = True
    write_figures: bool = True

    llm_verbalizer: bool = False
    visualization_backend: Literal["mermaid", "matplotlib", "none"] = "mermaid"

    grounder: str = "overlap"

    generic_mediators: tuple[str, ...] = DEFAULT_GENERIC_MEDIATORS

    @property
    def min_relevance_rank(self) -> int:
        return _RELEVANCE_ORDER.get(self.min_paper_relevance, 1)

    def relevance_ok(self, relevance: str) -> bool:
        if self.allow_low_relevance:
            return relevance in _RELEVANCE_ORDER
        return _RELEVANCE_ORDER.get(relevance, -1) >= self.min_relevance_rank

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "HypothesisConfig":
        if not path:
            return cls()
        import yaml
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypothesisConfig":
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in (data or {}).items() if k in known}
        if "generic_mediators" in kw and kw["generic_mediators"] is not None:
            kw["generic_mediators"] = tuple(kw["generic_mediators"])
        return cls(**kw)

    def with_overrides(self, **kw: Any) -> "HypothesisConfig":
        """Return a copy with non-None overrides applied (CLI flags that were actually passed)."""
        from dataclasses import replace
        clean = {k: v for k, v in kw.items() if v is not None and k in {f.name for f in fields(self)}}
        return replace(self, **clean)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        d = asdict(self)
        d["generic_mediators"] = list(self.generic_mediators)
        return d
