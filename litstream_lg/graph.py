"""Assemble the pipeline as a LangGraph StateGraph.

    START → acquire ─(budget)→ pdf ─(has phases)→ agentic ⟲ ─(loop)→ deliver → END
                  └(over budget)──────────────────────────────────────┘

The control flow is a declarative graph with a built-in checkpointer for durable resume.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .state import PipelineState
from .nodes import (acquire_node, pdf_node, agentic_phase_node, deliver_node,
                    route_after_acquire, route_after_pdf, route_after_phase)


def build_graph(checkpointer=None):
    """Compile the pipeline graph. Pass a LangGraph checkpointer (e.g. SqliteSaver)
    to enable durable resume across runs/crashes."""
    g = StateGraph(PipelineState)
    g.add_node("acquire", acquire_node)
    g.add_node("pdf", pdf_node)
    g.add_node("agentic", agentic_phase_node)
    g.add_node("deliver", deliver_node)

    g.add_edge(START, "acquire")
    g.add_conditional_edges("acquire", route_after_acquire,
                            {"pdf": "pdf", "deliver": "deliver"})
    g.add_conditional_edges("pdf", route_after_pdf,
                            {"agentic": "agentic", "deliver": "deliver"})
    g.add_conditional_edges("agentic", route_after_phase,
                            {"agentic": "agentic", "deliver": "deliver"})
    g.add_edge("deliver", END)
    return g.compile(checkpointer=checkpointer)
