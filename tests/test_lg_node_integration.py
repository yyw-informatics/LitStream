"""agentic_phase_node end-to-end with a FAKE tool-calling model — the integration test.

Proves create_react_agent + the LangChain file tools + the LedgerCallbackHandler + the
output-existence check all work together with NO network and NO real LLM. We monkeypatch
`litstream_lg.nodes.make_chat_model` to hand back a scripted fake model (callbacks passed
straight through, exactly as the real factory does), so everything else in the node — the
real ReAct agent, the real confined tools, the real callback debiting the real ledger,
and the real output check — runs unchanged.

Also covers the budget-gate early returns, which must short-circuit BEFORE any model
construction or call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import litstream_lg.nodes as nodes
from litstream.ledger.cost import CostLedger

from test_lg_helpers import FakeToolCallingModel, make_usage, write_skill


def _base_state(tmp_path, *, cap_usd=50.0, max_cost_per_run_usd=None,
                pending=("mine",), project="demo"):
    skills_dir = tmp_path / "skills"
    write_skill(skills_dir, "mine-paper")
    write_skill(skills_dir, "synthesize-literature")
    routine = {"name": "r", "default_model": "claude-haiku-4-5",
               "models": {}}
    if max_cost_per_run_usd is not None:
        routine["max_cost_per_run_usd"] = max_cost_per_run_usd
    (tmp_path / "context.md").write_text("study context")
    return {
        "routine": routine,
        "project": project,
        "project_dir": str(tmp_path),
        "skills_dir": str(skills_dir),
        "context_rel": "context.md",
        "db_path": str(tmp_path / "ledger.db"),
        "library_dir": str(tmp_path / "lib"),
        "cap_usd": cap_usd,
        "pending_phases": list(pending),
        "phases_done": [],
        "status": "running",
        "cost_cents": 0.0,
    }


def _start_run(state):
    led = CostLedger(state["db_path"])
    led.set_policy(cap_usd=state["cap_usd"])
    run_id = led.start_run(project=state["project"], routine="r", invocation="manual")
    state["run_id"] = run_id
    led.close()
    return run_id


def _patch_fake_model(monkeypatch, out_path, *, call1_usage=None, call2_usage=None,
                      recorder=None):
    """Replace make_chat_model so the node builds our fake model with its callbacks."""
    def fake_factory(model_name, *, callbacks=None, max_tokens=4096):
        model = FakeToolCallingModel(
            out_path=out_path,
            call1_usage=call1_usage or make_usage(100, 20, cache_read=10, cache_creation=5),
            call2_usage=call2_usage or make_usage(50, 5),
            callbacks=callbacks,
        )
        if recorder is not None:
            recorder["model"] = model
            recorder["callbacks"] = callbacks
        return model
    monkeypatch.setattr(nodes, "make_chat_model", fake_factory)


def test_agentic_phase_happy_path(tmp_path, monkeypatch):
    state = _base_state(tmp_path, pending=("mine", "synthesize"))
    run_id = _start_run(state)
    out_rel = f"projects/{state['project']}/literature/p1_evidence.md"
    rec = {}
    _patch_fake_model(monkeypatch, out_rel, recorder=rec)

    result = nodes.agentic_phase_node(state)

    out_file = tmp_path / out_rel
    assert out_file.is_file()
    assert out_file.stat().st_size > 200

    assert result["phases_done"] == ["mine"]
    assert result["pending_phases"] == ["synthesize"]
    assert result["status"] == "running"
    assert result["cost_cents"] > 0

    led = CostLedger(state["db_path"])
    events = led._conn.execute(
        "SELECT COUNT(*) AS n FROM cost_events WHERE run_id = ?", (run_id,)
    ).fetchone()["n"]
    led.close()
    assert events >= 1
    assert rec["callbacks"] and any(
        isinstance(cb, nodes.LedgerCallbackHandler) for cb in rec["callbacks"])


def test_agentic_phase_normalizes_mine_naming(tmp_path, monkeypatch):
    """mine phase writes a non-_evidence name; the node normalizes it and still passes."""
    state = _base_state(tmp_path, pending=("mine",))
    _start_run(state)
    out_rel = f"projects/{state['project']}/literature/smith2024.md"
    _patch_fake_model(monkeypatch, out_rel)

    result = nodes.agentic_phase_node(state)

    assert result.get("phases_done") == ["mine"]
    lit = tmp_path / f"projects/{state['project']}/literature"
    assert (lit / "smith2024_evidence.md").is_file()
    assert not (lit / "smith2024.md").exists()


def test_agentic_phase_missing_output_fails(tmp_path, monkeypatch):
    """If the model 'writes' to a path the output check doesn't see -> status failed."""
    state = _base_state(tmp_path, pending=("synthesize",))
    _start_run(state)
    out_rel = f"projects/{state['project']}/literature/wrong_name.md"
    _patch_fake_model(monkeypatch, out_rel)

    result = nodes.agentic_phase_node(state)
    assert result["status"] == "failed"
    assert result["pending_phases"] == []
    assert "no output" in result["note"]


def test_agentic_phase_per_run_cap_aborts_without_model_call(tmp_path, monkeypatch):
    state = _base_state(tmp_path, max_cost_per_run_usd=0.001)
    run_id = _start_run(state)

    led = CostLedger(state["db_path"])
    led.seed_pricing("claude-haiku-4-5", 1.0, 5.0, 0.1)
    led.record(run_id, model="claude-haiku-4-5", input_tokens=100_000)
    led.close()

    called = {"factory": False}
    def boom_factory(*a, **k):
        called["factory"] = True
        raise AssertionError("model must NOT be constructed past the budget gate")
    monkeypatch.setattr(nodes, "make_chat_model", boom_factory)

    result = nodes.agentic_phase_node(state)
    assert called["factory"] is False
    assert result["status"] == "aborted_budget"
    assert result["pending_phases"] == []
    assert "per-run cap" in result["note"]


def test_agentic_phase_month_cap_aborts_without_model_call(tmp_path, monkeypatch):
    state = _base_state(tmp_path, cap_usd=0.001)
    run_id = _start_run(state)

    led = CostLedger(state["db_path"])
    led.seed_pricing("claude-haiku-4-5", 1.0, 5.0, 0.1)
    led.record(run_id, model="claude-haiku-4-5", input_tokens=100_000)
    led.close()

    monkeypatch.setattr(nodes, "make_chat_model",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("model must NOT run past month cap")))

    result = nodes.agentic_phase_node(state)
    assert result["status"] == "aborted_budget"
    assert result["pending_phases"] == []
    assert "month cap" in result["note"]
