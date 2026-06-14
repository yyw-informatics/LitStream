"""Mid-phase budget hard-stop — the cost callback aborts a runaway agent loop.

The between-phase budget gate alone cannot stop a single non-converging agentic phase
from spending past the per-run cap before the next gate, so LedgerCallbackHandler binds
the cap inside the loop and aborts as soon as the run cost crosses it.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from litstream.ledger.cost import CostLedger
from litstream_lg.models import BudgetExceeded, LedgerCallbackHandler


def _usage_result(input_tokens: int, output_tokens: int) -> LLMResult:
    msg = AIMessage(content="x")
    msg.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                          "total_tokens": input_tokens + output_tokens,
                          "input_token_details": {}}
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


@pytest.fixture
def ledger(tmp_path):
    led = CostLedger(str(tmp_path / "b.db"))
    led.set_policy(cap_usd=50.0)
    led.seed_pricing("claude-sonnet-4-6", 3.0, 15.0, 0.3)   # $3 / $15 per Mtok
    yield led
    led.close()


def test_no_cap_never_aborts(ledger):
    rid = ledger.start_run(project="t", routine="t")
    cb = LedgerCallbackHandler(ledger, rid, model="claude-sonnet-4-6", phase="mine")
    for _ in range(20):                       # would be expensive, but no cap set
        cb.on_llm_end(_usage_result(100_000, 4096))
    assert cb.calls == 20                     # never raised


def test_aborts_once_run_cost_crosses_cap(ledger):
    rid = ledger.start_run(project="t", routine="t")
    # cap = 5 cents ($0.05). One call ≈ 10000*3/1e6*100 + 100*15/1e6*100 = 3.15c.
    cb = LedgerCallbackHandler(ledger, rid, model="claude-sonnet-4-6", phase="mine",
                               cap_cents=5.0)
    cb.on_llm_end(_usage_result(10_000, 100))         # 3.15c — under cap, fine
    assert cb.calls == 1
    with pytest.raises(BudgetExceeded):
        cb.on_llm_end(_usage_result(10_000, 100))     # 6.30c total — crosses 5c cap
    # the over-cap call was still recorded before aborting (cost is real)
    assert ledger.run_cost_cents(rid) >= 5.0


def test_cap_not_crossed_does_not_abort(ledger):
    rid = ledger.start_run(project="t", routine="t")
    cb = LedgerCallbackHandler(ledger, rid, model="claude-sonnet-4-6", phase="mine",
                               cap_cents=1000.0)              # $10 cap, tiny calls
    for _ in range(5):
        cb.on_llm_end(_usage_result(1000, 50))
    assert cb.calls == 5
    assert ledger.run_cost_cents(rid) < 1000.0


def test_budget_exceeded_propagates_through_model_invoke(ledger):
    """The cap must abort the real dispatch path, not just a direct on_llm_end() call:
    langchain-core re-raises a callback's exception only when the handler sets
    raise_error=True, so this drives a model.invoke() with the handler attached."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    rid = ledger.start_run(project="t", routine="t")
    LedgerCallbackHandler(ledger, rid, model="claude-sonnet-4-6").on_llm_end(
        _usage_result(50_000, 2_000))
    cb = LedgerCallbackHandler(ledger, rid, model="claude-sonnet-4-6", phase="mine", cap_cents=5.0)
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    with pytest.raises(BudgetExceeded):
        model.invoke("go", config={"callbacks": [cb]})
