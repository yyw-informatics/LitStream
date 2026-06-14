"""graph wiring + the three routing functions.

build_graph() must compile to exactly {acquire, pdf, agentic, deliver} (+ START/END);
the routing functions encode the budget/loop control flow:
  - route_after_acquire: 'deliver' on aborted_budget, else 'pdf'
  - route_after_pdf:      'agentic' iff pending_phases, else 'deliver'
  - route_after_phase:    'deliver' on failed/aborted or empty pending, else 'agentic'
"""

from __future__ import annotations

from litstream_lg.graph import build_graph
from litstream_lg.nodes import (route_after_acquire, route_after_pdf,
                                route_after_phase)


def test_build_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_node_set_is_exactly_the_four_nodes():
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    assert {"acquire", "pdf", "agentic", "deliver"} <= nodes
    user_nodes = nodes - {"__start__", "__end__"}
    assert user_nodes == {"acquire", "pdf", "agentic", "deliver"}


def test_route_after_acquire_budget_abort_goes_to_deliver():
    assert route_after_acquire({"status": "aborted_budget"}) == "deliver"


def test_route_after_acquire_running_goes_to_pdf():
    assert route_after_acquire({"status": "running"}) == "pdf"


def test_route_after_acquire_default_goes_to_pdf():
    assert route_after_acquire({}) == "pdf"


def test_route_after_pdf_with_pending_goes_to_agentic():
    assert route_after_pdf({"pending_phases": ["mine"]}) == "agentic"


def test_route_after_pdf_no_pending_goes_to_deliver():
    assert route_after_pdf({"pending_phases": []}) == "deliver"
    assert route_after_pdf({}) == "deliver"


def test_route_after_phase_failed_goes_to_deliver():
    assert route_after_phase({"status": "failed", "pending_phases": ["synthesize"]}) == "deliver"


def test_route_after_phase_aborted_goes_to_deliver():
    assert route_after_phase({"status": "aborted_budget", "pending_phases": ["synthesize"]}) == "deliver"


def test_route_after_phase_more_phases_loops_to_agentic():
    assert route_after_phase({"status": "running", "pending_phases": ["synthesize"]}) == "agentic"


def test_route_after_phase_empty_pending_goes_to_deliver():
    assert route_after_phase({"status": "running", "pending_phases": []}) == "deliver"
