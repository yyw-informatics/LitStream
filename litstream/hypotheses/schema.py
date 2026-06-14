"""Core data model for the context-bound hypothesis-candidate generator.

Plain ``dataclasses`` only — the module imports with no GPU, network, or third-party
dependency (``networkx`` is needed only by ``graph_builder``, ``yaml`` only by ``normalize``).
The ``Entity`` / ``BioContext`` / ``FindingFrame`` / ``EvidenceEdge`` records are *frozen* so they
can serve as stable, hashable values; ``HypothesisCandidate`` stays mutable because scores, warnings,
and the test design are filled in after generation.

Graph identity is the string ``Entity.entity_id`` — we key the networkx graph by id, never by the
Entity object — so ``attrs`` (a dict) is excluded from hashing/equality and the records stay hashable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

# Kept as Literal aliases for documentation + as plain tuples for runtime validation.

EntityType = Literal[
    "GENE_RNA",
    "SURFACE_PROTEIN",
    "GENE_OR_PROTEIN_UNSPECIFIED",
    "CELL_TYPE",
    "CELL_STATE",
    "DISEASE",
    "PERTURBATION",
    "SIGNATURE",
    "CELL_FREQUENCY",
    "PHENOTYPE",
    "UNKNOWN",
]

Direction = Literal[
    "increase",
    "decrease",
    "no_change",
    "association_positive",
    "association_negative",
    "unknown",
]

ReadoutModality = Literal[
    "rna",
    "surface_protein",
    "signature",
    "cell_frequency",
    "cell_state",
    "cytokine",
    "phenotype",
    "unknown",
]

EvidenceMode = Literal[
    "interventional",
    "longitudinal",
    "case_control",
    "cross_sectional_association",
    "descriptive_marker",
    "descriptive_expression",
    "unknown",
]

RelationType = Literal[
    "INCREASES_READOUT",
    "DECREASES_READOUT",
    "ASSOCIATED_WITH_HIGHER_READOUT",
    "ASSOCIATED_WITH_LOWER_READOUT",
    "DEFINES_OR_MARKS",
    "DESCRIPTIVE_EXPRESSION",
]

HypothesisMotif = Literal[
    "perturbation_to_marker_state_completion",
    "disease_signature_reversal",
    "signature_consolidation",
    "cite_seq_marker_bridge",
]

GroundingLabel = Literal["entailed", "contradicted", "unknown"]
Relevance = Literal["HIGH", "MODERATE", "LOW"]

# Directions that carry a real effect (an edge whose sign matters for composition).
EFFECT_DIRECTIONS: frozenset[str] = frozenset(
    {"increase", "decrease", "association_positive", "association_negative"}
)
# Directions safe to read as causal-language ("predicted to increase") vs association-only.
INTERVENTIONAL_DIRECTIONS: frozenset[str] = frozenset({"increase", "decrease"})
ASSOCIATION_DIRECTIONS: frozenset[str] = frozenset(
    {"association_positive", "association_negative"}
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    """Lowercase identifier-slug: 'IFN-β stimulation' -> 'ifn_b_stimulation' (after Greek folding
    upstream it becomes 'ifn_beta_stimulation'). Collapses runs of non-alphanumerics to one '_'."""
    return _SLUG_RE.sub("_", str(text or "").strip().lower()).strip("_") or "unknown"


def make_entity_id(etype: str, species: str | None, canonical_name: str) -> str:
    """Stable graph key. ``GENE_RNA:human:foxp3`` / ``DISEASE:na:covid_19``."""
    sp = slug(species) if species else "na"
    return f"{etype}:{sp}:{slug(canonical_name)}"


@dataclass(frozen=True)
class Entity:
    """A normalized biological entity. ``entity_id`` is the canonical identity used as the graph key;
    ``attrs`` (e.g. the marker->gene bridge) is excluded from hashing/equality so the record stays
    hashable while still carrying metadata."""

    entity_id: str
    type: EntityType
    canonical_name: str
    raw_names: tuple[str, ...] = ()
    species: str | None = None
    normalizer: str = "rule"
    normalization_confidence: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def is_named(self) -> bool:
        """True when this is a concrete, non-empty, non-UNKNOWN entity."""
        return bool(self.canonical_name) and self.type != "UNKNOWN"

    def label(self) -> str:
        """Human-readable node label for reports / diagrams."""
        return self.canonical_name


@dataclass(frozen=True)
class BioContext:
    species: tuple[str, ...] = ()
    tissue: tuple[str, ...] = ()
    disease: tuple[str, ...] = ()
    cell_type: tuple[str, ...] = ()
    cell_state: tuple[str, ...] = ()
    perturbation: tuple[str, ...] = ()
    condition: str | None = None
    comparator: str | None = None
    timepoint: str | None = None

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.species, self.tissue, self.disease, self.cell_type, self.cell_state,
             self.perturbation, self.condition, self.comparator, self.timepoint)
        )

    def describe(self) -> str:
        """Compact one-line description for claims / diagrams."""
        bits: list[str] = []
        if self.species:
            bits.append(" ".join(self.species))
        if self.tissue:
            bits.append(" ".join(self.tissue))
        if self.cell_type:
            bits.append(" ".join(self.cell_type))
        if self.disease:
            bits.append(" ".join(self.disease))
        return " ".join(bits) or "unspecified context"


@dataclass(frozen=True)
class FindingFrame:
    """One atomic, normalized finding extracted from a paper's ``findings`` list. Graph edges are
    derived from frames; the frame keeps full provenance (raw statement + source quote)."""

    frame_id: str
    paper_id: str
    raw_statement: str
    source_quote: str
    relevance: Relevance

    readout: Entity
    readout_modality: ReadoutModality
    direction: Direction

    context: BioContext
    evidence_mode: EvidenceMode

    parser_confidence: float
    grounding_label: GroundingLabel = "unknown"
    grounding_score: float = 0.0

    atomic_claim: str | None = None
    # entities other than the readout that the finding names (driver perturbation/disease, cell type)
    anchor: Entity | None = None
    cell_type: Entity | None = None
    warnings: tuple[str, ...] = ()

    @property
    def is_grounded(self) -> bool:
        return self.grounding_label == "entailed"


@dataclass(frozen=True)
class EvidenceEdge:
    """A directed, signed, context-typed edge derived from one frame. Stored as a networkx edge
    attribute; the graph is keyed by ``source_entity.entity_id`` / ``target_entity.entity_id``."""

    edge_id: str
    source_entity: Entity
    relation: RelationType
    target_entity: Entity
    frame_id: str
    paper_id: str
    context: BioContext
    evidence_mode: EvidenceMode
    grounding_score: float
    parser_confidence: float
    source_quote: str
    raw_statement: str
    direction: Direction = "unknown"
    warnings: tuple[str, ...] = ()


@dataclass
class HypothesisCandidate:
    hypothesis_id: str
    claim: str
    motif: HypothesisMotif
    predicted_direction: Direction
    context: BioContext

    anchor: Entity
    mediator: Entity | None
    readouts: list[Entity]

    support_frame_ids: list[str]
    support_edge_ids: list[str]
    support_paper_ids: list[str]

    evidence_mode_summary: list[EvidenceMode]
    novelty_scope: str = "local_corpus_only"

    scores: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    test_design: dict[str, Any] = field(default_factory=dict)

    embedding_text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def rank_score(self) -> float:
        return float(self.scores.get("rank_score", 0.0))

    @property
    def cell_type_name(self) -> str:
        return " ".join(self.context.cell_type)


@dataclass
class HypothesisRunResult:
    frames: list[FindingFrame]
    graph: Any
    candidates: list[HypothesisCandidate]
    diagnostics: dict[str, Any] = field(default_factory=dict)
