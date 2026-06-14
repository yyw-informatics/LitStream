"""Tests for the LLM-finder escalation tier (litstream/eval/llm_finder.py). No API, no model:
a fake finder returns canned spans, and the rescue logic + guards are checked directly. The point of
the tier is that the LLM can only SURFACE a real passage — it can never bless a fabrication, and it
can never override MiniCheck's authority over numbers."""

from __future__ import annotations

import pytest

from litstream_evidence.evidence_schema import empty_record
from litstream_evidence.ground_retrieval import MiniCheckVerifier, OverlapVerifier, reground_record
from litstream_evidence.llm_finder import LLMFinder, apply_llm_finder, make_finder, quote_in_source

SOURCE = ("Methods. We profiled PBMC with CITE-seq. Cytotoxic CD8 T lymphocytes expressed GZMB. "
          "CD4 T cells were 25 to 40 percent of T cells in healthy donors. CD19 marked B cells.")


class _Finder:
    """Fake finder: returns a canned span when a key appears in the claim, else '' (UNSUPPORTED)."""

    def __init__(self, table: dict):
        self.table = table

    def find(self, claim: str, source_text: str) -> str:
        for key, span in self.table.items():
            if key in claim:
                return span
        return ""


def _flag_one(rec: dict, verifier=None, value_verifier=None):
    # run the cheap pass with a retriever that surfaces nothing useful, so the item is flagged
    return reground_record(rec, lambda q: ["irrelevant text about microscopes"],
                           verifier or OverlapVerifier(), value_verifier=value_verifier)[1]


# ---- the substring guard --------------------------------------------------------

def test_quote_in_source_normalizes_and_rejects_invented():
    assert quote_in_source("cytotoxic   CD8 t LYMPHOCYTES", SOURCE) is True   # whitespace/case-insensitive
    assert quote_in_source("a totally invented supporting span", SOURCE) is False
    assert quote_in_source("", SOURCE) is False


# ---- rescue a real (paraphrase/retrieval-miss) flag -----------------------------

def test_llm_finder_rescues_entity_with_real_span():
    rec = empty_record("p")
    rec["cell_types"] = [{"name": "cytotoxic CD8 T lymphocytes", "source_quote": "made up"}]
    report = _flag_one(rec)
    assert report["by_field"]["cell_types"]["flagged"] == 1

    finder = _Finder({"cytotoxic CD8 T lymphocytes": "Cytotoxic CD8 T lymphocytes expressed GZMB"})
    apply_llm_finder(rec, report, SOURCE, finder, OverlapVerifier())

    assert report["by_field"]["cell_types"]["grounded"] == 1
    assert report["by_field"]["cell_types"]["flagged"] == 0
    assert len(report["rescued"]) == 1
    assert "GZMB" in rec["cell_types"][0]["source_quote"]      # quote replaced with the REAL span


# ---- the LLM cannot bless a fabrication -----------------------------------------

def test_llm_finder_rejects_invented_quote():
    rec = empty_record("p")
    rec["cell_types"] = [{"name": "plasmacytoid dendritic cells", "source_quote": "x"}]
    report = _flag_one(rec)
    assert report["by_field"]["cell_types"]["flagged"] == 1

    # the finder hallucinates a supporting quote that is NOT in the paper
    finder = _Finder({"plasmacytoid dendritic cells":
                      "The paper reports plasmacytoid dendritic cells in all samples"})
    apply_llm_finder(rec, report, SOURCE, finder, OverlapVerifier())

    assert report["by_field"]["cell_types"]["flagged"] == 1    # not a substring -> stays flagged
    assert report["rescued"] == []


# ---- the LLM cannot override MiniCheck's authority over numbers ------------------

def test_llm_finder_preserves_number_teeth_for_values():
    rec = empty_record("p")
    rec["frequencies"] = [{"cell_type": "B cells", "value": "5-15", "unit": "%", "source_quote": "x"}]
    yes = MiniCheckVerifier(predict=lambda c, p: True)            # MiniCheck says "supported" for everything
    report = _flag_one(rec, value_verifier=yes)
    assert report["by_field"]["frequencies"]["flagged"] == 1

    finder = _Finder({"B cells": "CD19 marked B cells"})
    apply_llm_finder(rec, report, SOURCE, finder, OverlapVerifier(), value_verifier=yes)

    assert report["rescued"] == []
    assert report["by_field"]["frequencies"]["flagged"] == 1


def test_llmfinder_returns_quote_only_when_supported():
    class _Res:
        def __init__(self, verdict, quote):
            self.verdict, self.quote = verdict, quote

    class _Structured:
        def __init__(self, res):
            self.res = res

        def invoke(self, prompt):
            return self.res

    class _FakeModel:
        def __init__(self, res):
            self.res = res

        def with_structured_output(self, schema):
            return _Structured(self.res)

    assert LLMFinder(_FakeModel(_Res("SUPPORTED", "CD19 marked B cells"))).find("B cells", SOURCE) \
        == "CD19 marked B cells"
    assert LLMFinder(_FakeModel(_Res("UNSUPPORTED", ""))).find("X", SOURCE) == ""


def test_make_finder_unknown_raises():
    with pytest.raises(ValueError):
        make_finder("nope")
