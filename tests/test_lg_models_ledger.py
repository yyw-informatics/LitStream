"""LedgerCallbackHandler cost accounting.

The handler reads LangChain's normalized usage_metadata off an on_llm_end response
and debits the shared SQLite ledger: regular input is priced separately from
cache-read and cache-creation. These tests construct synthetic LLMResults (no
provider, no network) and assert the ledger row and run rollup.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from litstream_lg.models import LedgerCallbackHandler

from test_lg_helpers import ledger, make_usage  # noqa: F401  (ledger is a fixture)


def _result(*messages):
    """An LLMResult whose generations[i][0].message is each given AIMessage."""
    return LLMResult(generations=[[ChatGeneration(message=m)] for m in messages])


def _seed_run(led):
    # input $3 / output $15 / cached $0.30 per Mtok (cents/Mtok: 300 / 1500 / 30).
    led.seed_pricing("m", input_per_mtok=3.0, output_per_mtok=15.0,
                     cached_input_per_mtok=0.3)
    return led.start_run(project="p", routine="r", invocation="manual")


def _events(led, run_id):
    return led._conn.execute(
        "SELECT input_tokens, output_tokens, cached_input_tokens, cost_cents "
        "FROM cost_events WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()


def _run_row(led, run_id):
    return led._conn.execute(
        "SELECT input_tokens, output_tokens, cached_input_tokens, cost_cents "
        "FROM runs WHERE id = ?", (run_id,)
    ).fetchone()


def test_on_llm_end_debits_regular_input_minus_cache(ledger):
    run_id = _seed_run(ledger)
    handler = LedgerCallbackHandler(ledger, run_id, model="m", phase="mine")

    # input=100 with cache_read=10, cache_creation=5 -> REGULAR input = 85.
    msg = AIMessage(content="x", usage_metadata=make_usage(100, 20, cache_read=10, cache_creation=5))
    handler.on_llm_end(_result(msg))

    rows = _events(ledger, run_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["input_tokens"] == 85          # 100 - 10 - 5, the REGULAR slice
    assert row["output_tokens"] == 20
    assert row["cached_input_tokens"] == 10   # cache READ stored separately

    # Cost: regular@300 + creation@300*1.25 + cache_read@30 + output@1500, per Mtok.
    expected = (85 * 300 + 5 * 300 * 1.25 + 10 * 30 + 20 * 1500) / 1_000_000
    assert row["cost_cents"] == pytest.approx(expected)
    assert handler.calls == 1
    assert handler.cost_cents == pytest.approx(expected)
    # The run rollup mirrors the single event.
    assert _run_row(ledger, run_id)["cost_cents"] == pytest.approx(expected)


def test_cache_creation_charged_at_input_rate_times_1_25(ledger):
    """A pure cache-creation call: input == cache_creation, no regular, no cache read."""
    run_id = _seed_run(ledger)
    handler = LedgerCallbackHandler(ledger, run_id, model="m")
    msg = AIMessage(content="x", usage_metadata=make_usage(40, 0, cache_creation=40))
    handler.on_llm_end(_result(msg))

    row = _events(ledger, run_id)[0]
    assert row["input_tokens"] == 0           # regular = 40 - 0 - 40
    expected = (40 * 300 * 1.25) / 1_000_000  # creation @ 1.25x input rate
    assert row["cost_cents"] == pytest.approx(expected)


def test_no_usage_metadata_is_no_debit_no_crash(ledger):
    run_id = _seed_run(ledger)
    handler = LedgerCallbackHandler(ledger, run_id, model="m")
    msg = AIMessage(content="no usage here")   # usage_metadata is None
    handler.on_llm_end(_result(msg))           # must not raise
    assert _events(ledger, run_id) == []
    assert handler.calls == 0
    assert handler.cost_cents == 0.0
    assert _run_row(ledger, run_id)["cost_cents"] == 0.0


def test_empty_generations_no_crash(ledger):
    run_id = _seed_run(ledger)
    handler = LedgerCallbackHandler(ledger, run_id, model="m")
    handler.on_llm_end(LLMResult(generations=[]))   # nothing at all
    assert handler.calls == 0


def test_multiple_generations_each_debited(ledger):
    run_id = _seed_run(ledger)
    handler = LedgerCallbackHandler(ledger, run_id, model="m")
    m1 = AIMessage(content="a", usage_metadata=make_usage(100, 10))
    m2 = AIMessage(content="b", usage_metadata=make_usage(200, 20, cache_read=50))
    handler.on_llm_end(_result(m1, m2))

    rows = _events(ledger, run_id)
    assert len(rows) == 2
    assert handler.calls == 2
    assert rows[0]["input_tokens"] == 100      # no cache -> all regular
    assert rows[1]["input_tokens"] == 150      # 200 - 50 cache read
    assert rows[1]["cached_input_tokens"] == 50

    # Run rollup sums both events; handler.cost_cents accumulates across them.
    total = sum(r["cost_cents"] for r in rows)
    assert _run_row(ledger, run_id)["cost_cents"] == pytest.approx(total)
    assert handler.cost_cents == pytest.approx(total)


def test_handler_accumulates_across_multiple_on_llm_end_calls(ledger):
    run_id = _seed_run(ledger)
    handler = LedgerCallbackHandler(ledger, run_id, model="m")
    handler.on_llm_end(_result(AIMessage(content="1", usage_metadata=make_usage(100, 10))))
    handler.on_llm_end(_result(AIMessage(content="2", usage_metadata=make_usage(300, 30))))

    assert handler.calls == 2
    rows = _events(ledger, run_id)
    assert len(rows) == 2
    assert handler.cost_cents == pytest.approx(sum(r["cost_cents"] for r in rows))
    # run rollup == handler running total.
    assert _run_row(ledger, run_id)["cost_cents"] == pytest.approx(handler.cost_cents)


def test_missing_detail_keys_default_to_zero(ledger):
    """usage_metadata without input_token_details -> all input counts as regular."""
    run_id = _seed_run(ledger)
    handler = LedgerCallbackHandler(ledger, run_id, model="m")
    msg = AIMessage(content="x",
                    usage_metadata={"input_tokens": 70, "output_tokens": 8, "total_tokens": 78})
    handler.on_llm_end(_result(msg))
    row = _events(ledger, run_id)[0]
    assert row["input_tokens"] == 70
    assert row["cached_input_tokens"] == 0


def test_phase_and_role_recorded_on_event(ledger):
    run_id = _seed_run(ledger)
    handler = LedgerCallbackHandler(ledger, run_id, model="m", phase="synthesize", role="langgraph")
    handler.on_llm_end(_result(AIMessage(content="x", usage_metadata=make_usage(10, 1))))
    row = ledger._conn.execute(
        "SELECT phase, role FROM cost_events WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row["phase"] == "synthesize"
    assert row["role"] == "langgraph"
