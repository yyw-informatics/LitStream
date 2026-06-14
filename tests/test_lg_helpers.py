"""Shared hermetic fixtures and fakes for the litstream_lg test suite.

Everything here runs offline: no network, no real LLM. The fake chat model
implements just enough of the langchain BaseChatModel contract to drive
``create_react_agent`` with a scripted tool-call then final-answer sequence, so the
agentic node can be exercised end-to-end without a provider.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from litstream.ledger.cost import CostLedger


@pytest.fixture
def ledger(tmp_path):
    """A CostLedger on a throwaway SQLite file, closed on teardown."""
    led = CostLedger(str(tmp_path / "ledger.db"))
    try:
        yield led
    finally:
        led.close()


def make_usage(input_tokens, output_tokens, cache_read=0, cache_creation=0):
    """LangChain-normalized usage_metadata, the shape on_llm_end reads."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {"cache_read": cache_read, "cache_creation": cache_creation},
    }


class FakeToolCallingModel(BaseChatModel):
    """A scripted chat model for create_react_agent.

    Call 1 emits a write_file tool call (with usage) that creates the phase's output
    file; call 2 emits a plain final answer. ``bind_tools`` returns self (the agent
    binds tools to the model), and callbacks passed at construction fire on_llm_end
    on each ``_generate``, mirroring how the real model debits the ledger.
    """

    # pydantic-backed fields (BaseChatModel is a pydantic model).
    out_path: str = "out.md"
    out_content: str = "evidence " * 60        # > 200 bytes so output_exists passes
    call1_usage: dict = {}
    call2_usage: dict = {}
    n_calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools, **kwargs):       # noqa: ANN001
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        idx = self.n_calls
        object.__setattr__(self, "n_calls", idx + 1)
        if idx == 0:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": "write_file",
                             "args": {"path": self.out_path, "content": self.out_content},
                             "id": "call_1"}],
                usage_metadata=self.call1_usage or make_usage(100, 20, cache_read=10, cache_creation=5),
            )
        else:
            msg = AIMessage(content="done",
                            usage_metadata=self.call2_usage or make_usage(50, 5))
        return ChatResult(generations=[ChatGeneration(message=msg)])


def write_skill(skills_dir: Path, skill_name: str, body: str = "Do the thing.") -> None:
    """Drop a minimal kb-skills SKILL.md so render_phase_parts can render a prompt."""
    sd = skills_dir / skill_name
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: test skill\n---\n\n{body}\n"
    )
