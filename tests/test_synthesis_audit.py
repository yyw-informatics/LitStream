"""Tests for the post-synthesize citation audit overlay (litstream_lg/synthesis_audit.py + the
nodes.agentic_phase_node hook). Fully offline: a tiny synthesis on disk + an injected fake verifier
(the keyword labeler from test_citation_check). No network, no real LLM, no MiniCheck download.

The two invariants under test: (1) the `0_synthesis_audit.md` overlay IS produced and lists the
failing claims, and (2) the synthesis markdown is left byte-for-byte UNCHANGED (flag-only).
"""

from __future__ import annotations

import pytest

import litstream_lg.nodes as nodes
from litstream_lg.synthesis_audit import audit_synthesis_output

from test_citation_check import SYNTHESIS, EVIDENCE, _Keyword
from test_lg_node_integration import _base_state, _start_run
from test_lg_helpers import FakeToolCallingModel, make_usage


def _write_project(tmp_path, project="demo"):
    """Lay out projects/<project>/literature/ with the synthesis + its evidence files."""
    lit = tmp_path / f"projects/{project}/literature"
    lit.mkdir(parents=True)
    (lit / "0_synthesis_literature.md").write_text(SYNTHESIS)
    for name, text in EVIDENCE.items():
        (lit / name).write_text(text)
    return lit


# ---------------------------------------------------------------------------
# The wrapper: overlay produced, failing claims listed, synthesis untouched
# ---------------------------------------------------------------------------

def test_audit_writes_overlay_and_leaves_synthesis_untouched(tmp_path):
    lit = _write_project(tmp_path)
    synth = lit / "0_synthesis_literature.md"
    before = synth.read_text()

    summary = audit_synthesis_output(tmp_path, "demo", _verifier=_Keyword())

    # 1) the overlay was written next to the synthesis
    assert summary["audited"] is True
    overlay = lit / "0_synthesis_audit.md"
    assert overlay.is_file()
    assert summary["report"] == str(overlay)

    report = overlay.read_text()
    assert "# Synthesis citation audit — demo" in report
    # the contradicted and partial claims surface in the flag table with their verdicts
    assert "Contradictory" in report
    assert "Partial" in report
    assert summary["flagged"] >= 2

    # 2) the synthesis prose is NOT edited — flag-only, byte-for-byte identical
    assert synth.read_text() == before


def test_audit_no_synthesis_is_a_noop(tmp_path):
    (tmp_path / "projects/demo/literature").mkdir(parents=True)
    summary = audit_synthesis_output(tmp_path, "demo", _verifier=_Keyword())
    assert summary == {"audited": False, "note": "no synthesis found"}
    assert not (tmp_path / "projects/demo/literature/0_synthesis_audit.md").exists()


def _patch_synth_model(monkeypatch, tmp_path, project):
    """Model double whose tool write lands the synthesis at the expected path."""
    out_rel = f"projects/{project}/literature/0_synthesis_literature.md"

    def fake_factory(model_name, *, callbacks=None, max_tokens=4096):
        return FakeToolCallingModel(out_path=out_rel, out_content=SYNTHESIS,
                                    call1_usage=make_usage(100, 20),
                                    call2_usage=make_usage(50, 5),
                                    callbacks=callbacks)
    monkeypatch.setattr(nodes, "make_chat_model", fake_factory)


def test_node_audits_synthesis_when_flagged(tmp_path, monkeypatch):
    state = _base_state(tmp_path, pending=("synthesize",))
    state["routine"]["audit_synthesis"] = True
    state["routine"]["synthesis_audit"] = {"verifier": "overlap"}
    _start_run(state)

    for name, text in EVIDENCE.items():
        (tmp_path / f"projects/demo/literature").mkdir(parents=True, exist_ok=True)
        (tmp_path / f"projects/demo/literature/{name}").write_text(text)
    _patch_synth_model(monkeypatch, tmp_path, "demo")
    monkeypatch.setattr("litstream.eval.citation_check.make_verifier",
                        lambda *a, **k: _Keyword())

    result = nodes.agentic_phase_node(state)

    assert result["status"] == "running"
    assert result["phases_done"] == ["synthesize"]
    assert result["audit_result"]["audited"] is True
    overlay = tmp_path / "projects/demo/literature/0_synthesis_audit.md"
    assert overlay.is_file()
    assert (tmp_path / "projects/demo/literature/0_synthesis_literature.md").read_text() == SYNTHESIS


def test_node_skips_audit_when_flag_off(tmp_path, monkeypatch):
    state = _base_state(tmp_path, pending=("synthesize",))
    _start_run(state)
    _patch_synth_model(monkeypatch, tmp_path, "demo")

    result = nodes.agentic_phase_node(state)

    assert result["status"] == "running"
    assert "audit_result" not in result
    assert not (tmp_path / "projects/demo/literature/0_synthesis_audit.md").exists()


def test_node_audit_failure_is_non_fatal(tmp_path, monkeypatch):
    """If the audit blows up (e.g. the grounding extra is missing), the phase still succeeds."""
    state = _base_state(tmp_path, pending=("synthesize",))
    state["routine"]["audit_synthesis"] = True
    _start_run(state)
    _patch_synth_model(monkeypatch, tmp_path, "demo")
    monkeypatch.setattr(nodes, "_audit_synthesis",
                        lambda cfg, pdir, project: {"error": "boom"})

    result = nodes.agentic_phase_node(state)

    assert result["status"] == "running"
    assert result["phases_done"] == ["synthesize"]
    assert result["audit_result"] == {"error": "boom"}
