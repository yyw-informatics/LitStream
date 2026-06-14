"""Tests for the evidence structuring step (litstream/eval/structure_evidence.py)
and the shared field list (litstream/eval/evidence_schema.py). Deterministic:
StubStructurer over hand-written evidence and source text, no model/network/PDF.
"""

from __future__ import annotations

import pytest

from litstream_evidence.evidence_schema import RELEVANCE, empty_record, validate
from litstream_evidence.structure_evidence import (
    LLMStructurer, StubStructurer, _map_relevance, ground_record, make_structurer,
    structure_evidence,
)

SOURCE = ("We studied FOXP3 and CD25 in regulatory T cell populations from human PBMC. "
          "CD19 was used for B cell gating.")

EVIDENCE = """---
relevance: "HIGH RELEVANCE"
species: "human"
tissue: "PBMC, bone marrow"
cell_types: ["regulatory T cell", "B cell"]
---
# Evidence
FOXP3 and CD25 mark Treg cells. CD19 marks B cells.
"""


@pytest.fixture
def record():
    rec, _ = structure_evidence("p1", EVIDENCE, SOURCE, StubStructurer())
    return rec


def test_stub_produces_a_schema_valid_record(record):
    assert validate(record) == []
    assert record["paper_id"] == "p1"
    assert record["relevance"] == "HIGH"
    assert record["species"] == ["human"]
    assert "PBMC" in record["tissue"]


def test_stub_pulls_fields_from_frontmatter_and_body(record):
    names = {c["name"] for c in record["cell_types"]}
    assert {"regulatory T cell", "B cell"} <= names
    markers = {m["marker"] for m in record["surface_markers"]}
    assert "CD25" in markers and "CD19" in markers
    genes = {g["symbol"] for g in record["genes"]}
    assert "FOXP3" in genes        # in the body + lexicon
    assert "CD25" not in genes     # CD25 is a marker here, not in the gene lexicon


def test_grounding_flags_unsupported_quotes():
    rec = empty_record("p")
    rec["relevance"] = "HIGH"
    rec["genes"] = [
        {"symbol": "CD25", "species": "human", "source_quote": "CD25"},               # in source
        {"symbol": "FOXP3", "species": "human", "source_quote": "FOXP3 cured cancer"},  # NOT in source
    ]
    report = ground_record(rec, SOURCE)
    assert report["by_field"]["genes"]["grounded"] == 1
    assert report["by_field"]["genes"]["ungrounded"] == 1
    assert any(u["item"]["symbol"] == "FOXP3" for u in report["ungrounded_items"])


def test_stub_quotes_are_grounded_in_source(record):
    report = ground_record(record, SOURCE)
    assert report["ungrounded"] == 0
    assert report["grounded"] > 0


def test_validate_catches_bad_records():
    bad = {"paper_id": "x", "relevance": "BANANA", "species": "human",
           "genes": [{"source_quote": "q"}]}
    errs = validate(bad)
    assert any("relevance" in e for e in errs)
    assert any("species" in e for e in errs)
    assert any("symbol" in e for e in errs)


@pytest.mark.parametrize("raw,expected", [
    ("HIGH RELEVANCE", "HIGH"), ("Moderate", "MODERATE"), ("low relevance", "LOW"),
    ("NOT RELEVANT", "NOT_USEFUL"), ("NOT-USEFUL", "NOT_USEFUL"), ("", "NOT_USEFUL"),
    ("gibberish", "NOT_USEFUL"),
])
def test_map_relevance(raw, expected):
    assert _map_relevance(raw) == expected


def test_empty_record_is_valid():
    assert validate(empty_record("x")) == []
    assert empty_record("x")["relevance"] in RELEVANCE


def test_make_structurer():
    assert isinstance(make_structurer("stub"), StubStructurer)
    with pytest.raises(ValueError):
        make_structurer("nope")


class _FakeStructured:
    def __init__(self, result):
        self.result, self.calls = result, []

    def invoke(self, prompt, *a, **k):
        self.calls.append(prompt)
        return self.result


class _FakeModel:
    """Stands in for a LangChain chat model: `.with_structured_output(schema)` returns a
    runnable that yields a canned EvidenceRecord, so we test the wiring without an API call."""
    def __init__(self, result):
        self.result, self.schema = result, None

    def with_structured_output(self, schema, **kw):
        self.schema = schema
        return _FakeStructured(self.result)


def test_llm_structurer_wiring_returns_a_valid_record():
    from litstream_evidence.evidence_models import EvidenceRecord, Gene, CellType
    canned = EvidenceRecord(
        paper_id="x", relevance="HIGH", species=["human"],
        genes=[Gene(symbol="FOXP3", species="human", source_quote="FOXP3+ Tregs")],
        cell_types=[CellType(name="regulatory T cell", source_quote="Tregs were...")],
    )
    model = _FakeModel(canned)
    s = LLMStructurer(model)
    assert model.schema is EvidenceRecord            # handed our Pydantic model to with_structured_output
    rec = s.structure("the analyst note", "paper text mentioning FOXP3 and Tregs")
    assert isinstance(rec, dict)                      # model_dump'd to a plain dict
    assert rec["relevance"] == "HIGH"
    assert rec["genes"][0]["symbol"] == "FOXP3"
    assert validate(rec) == []                        # the produced dict matches our field list
    assert "PAPER TEXT" in s.structured.calls[0]      # the paper text went into the prompt


def test_make_structurer_llm_accepts_an_injected_model():
    from litstream_evidence.evidence_models import EvidenceRecord
    s = make_structurer("llm", model=_FakeModel(EvidenceRecord()))
    assert isinstance(s, LLMStructurer)
