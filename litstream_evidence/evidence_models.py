"""Pydantic models for the evidence record — the typed mirror of `evidence_schema`.

The target for `model.with_structured_output(EvidenceRecord)`: the model fills these fields
directly and LangChain returns a validated object, so field descriptions double as instructions to
the model. Keep field names in sync with evidence_schema.py. Imported lazily by the LLM converter to
keep the dependency-free path clear of pydantic.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Disease(BaseModel):
    name: str = Field(description="disease/condition studied, e.g. COVID-19, melanoma, type 1 diabetes")
    source_quote: str = Field(description="a short phrase copied verbatim from the paper text")


class Perturbation(BaseModel):
    name: str = Field(description="what was applied/changed, e.g. 'anti-CD3/CD28', 'anti-PD-1', 'LPS', 'IFN-β'")
    type: str = Field(default="", description="stimulation | drug | genetic | vaccine | timepoint")
    source_quote: str = Field(description="a short phrase copied verbatim from the paper text")


class Gene(BaseModel):
    symbol: str = Field(description="gene symbol, e.g. FOXP3")
    species: str = Field(default="", description="human or mouse, if the paper says")
    source_quote: str = Field(description="a short phrase copied verbatim from the paper text")


class CellType(BaseModel):
    name: str = Field(description="cell type, e.g. regulatory T cell")
    source_quote: str = Field(description="a short phrase copied verbatim from the paper text")


class SurfaceMarker(BaseModel):
    marker: str = Field(description="antibody / surface (ADT) marker, e.g. CD25")
    maps_to_gene: str = Field(default="", description="the gene/protein it corresponds to, if known")
    source_quote: str = Field(description="a short phrase copied verbatim from the paper text")


class Frequency(BaseModel):
    cell_type: str = Field(description="the cell type the number refers to")
    value: str = Field(description="the number itself, e.g. '5.2'")
    unit: str = Field(default="", description="e.g. '%'")
    of_population: str = Field(default="", description="the denominator, e.g. 'CD4 T cells'")
    condition: str = Field(default="", description="e.g. 'healthy', 'stimulated'")
    source_quote: str = Field(description="a short phrase copied verbatim from the paper text")


class Cohort(BaseModel):
    group: str = Field(description="what the count refers to, e.g. 'healthy donors', 'COVID patients', 'cells'")
    n: str = Field(description="the count itself, e.g. '120'")
    unit: str = Field(default="", description="e.g. 'donors', 'cells', 'samples'")
    source_quote: str = Field(description="a short phrase copied verbatim from the paper text")


class GatingThreshold(BaseModel):
    marker: str = Field(description="the marker being gated, e.g. CD25")
    operator: str = Field(default="", description="e.g. '>', 'high', 'dim', 'bright'")
    value: str = Field(default="", description="the cutoff, e.g. '500' or 'high'")
    source_quote: str = Field(description="a short phrase copied verbatim from the paper text")


class Signature(BaseModel):
    name: str = Field(description="name of the gene signature/set")
    genes: list[str] = Field(default_factory=list, description="the gene symbols in the set")
    species: str = Field(default="")
    source_quote: str = Field(description="a short phrase copied verbatim from the paper text")


class Statement(BaseModel):
    statement: str = Field(description="a single checkable proposition stated by the paper")
    source_quote: str = Field(description="a short phrase copied verbatim from the paper text")


class EvidenceRecord(BaseModel):
    """Structured extraction of one paper, against the project's interests. Fill only facts
    actually stated in the paper; leave a list empty if it has nothing."""
    paper_id: str = Field(default="")
    relevance: Literal["HIGH", "MODERATE", "LOW", "NOT_USEFUL"] = Field(
        default="NOT_USEFUL", description="how useful this paper is for the project's interests")
    species: list[str] = Field(default_factory=list, description="organisms studied, e.g. ['human']")
    tissue: list[str] = Field(default_factory=list, description="tissues/samples, e.g. ['PBMC']")
    diseases: list[Disease] = Field(default_factory=list)
    perturbations: list[Perturbation] = Field(default_factory=list)
    genes: list[Gene] = Field(default_factory=list)
    cell_types: list[CellType] = Field(default_factory=list)
    surface_markers: list[SurfaceMarker] = Field(default_factory=list)
    frequencies: list[Frequency] = Field(default_factory=list)
    cohort: list[Cohort] = Field(default_factory=list)
    gating_thresholds: list[GatingThreshold] = Field(default_factory=list)
    signatures: list[Signature] = Field(default_factory=list)
    study_aim: list[Statement] = Field(
        default_factory=list,
        description="the study's EXPLICITLY stated aim/question/hypothesis (e.g. 'We hypothesized "
                    "that…', 'To test whether…'). Leave empty if the paper doesn't state one — do NOT infer.")
    findings: list[Statement] = Field(
        default_factory=list,
        description="key results/conclusions the paper states, each a single checkable proposition")
