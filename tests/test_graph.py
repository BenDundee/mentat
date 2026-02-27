"""Tests for the LangGraph workflow."""

from unittest.mock import MagicMock, patch

from mentat.core.models import Intent, OrchestrationResult
from mentat.graph.state import GraphState
from mentat.graph.workflow import _route_after_orchestration, format_response


def _make_state(**overrides) -> GraphState:
    base: GraphState = {
        "messages": [],
        "user_message": "Hello",
        "orchestration_result": None,
        "search_results": None,
        "rag_results": None,
        "persona_context": None,
        "plan_context": None,
        "coaching_response": None,
        "quality_rating": None,
        "final_response": None,
    }
    return GraphState(**{**base, **overrides})


def test_format_response_with_result():
    """format_response should render intent and reasoning."""
    result = OrchestrationResult(
        intent=Intent.CHECK_IN,
        confidence=0.95,
        reasoning="User is sharing how they are doing.",
    )
    state = _make_state(orchestration_result=result)
    new_state = format_response(state)

    assert new_state["final_response"] is not None
    assert "check-in" in new_state["final_response"]
    assert "95%" in new_state["final_response"]


def test_format_response_without_result():
    """format_response should return a fallback message when result is None."""
    state = _make_state()
    new_state = format_response(state)
    assert new_state["final_response"] is not None
    assert len(new_state["final_response"]) > 0


def test_build_graph_has_expected_nodes():
    """build_graph() should include orchestration and format_response nodes."""
    from mentat.graph.workflow import build_graph

    with (
        patch("mentat.graph.workflow.OrchestrationAgent") as MockOrch,
        patch("mentat.graph.workflow.SearchAgent") as MockSearch,
    ):
        MockOrch.return_value.run = MagicMock()
        MockSearch.return_value.run = MagicMock()
        graph = build_graph()

    node_names = list(graph.nodes.keys())
    assert "orchestration" in node_names
    assert "format_response" in node_names


def test_compile_graph_succeeds():
    """compile_graph() should return a compiled graph without raising."""
    from mentat.graph.workflow import compile_graph

    with (
        patch("mentat.graph.workflow.OrchestrationAgent") as MockOrch,
        patch("mentat.graph.workflow.SearchAgent") as MockSearch,
    ):
        MockOrch.return_value.run = MagicMock()
        MockSearch.return_value.run = MagicMock()
        compiled = compile_graph()

    assert compiled is not None


def test_route_with_search():
    """Returns 'search' when suggested_agents contains 'search'."""
    result = OrchestrationResult(
        intent=Intent.QUESTION,
        confidence=0.9,
        reasoning="Needs current info.",
        suggested_agents=("search",),
    )
    state = _make_state(orchestration_result=result)
    assert _route_after_orchestration(state) == "search"


def test_route_without_search():
    """Returns 'format_response' when suggested_agents is empty."""
    result = OrchestrationResult(
        intent=Intent.CHECK_IN,
        confidence=0.95,
        reasoning="Simple check-in.",
        suggested_agents=(),
    )
    state = _make_state(orchestration_result=result)
    assert _route_after_orchestration(state) == "format_response"


def test_route_none_result():
    """Returns 'format_response' when orchestration_result is None."""
    state = _make_state(orchestration_result=None)
    assert _route_after_orchestration(state) == "format_response"


def test_build_graph_has_search_node():
    """build_graph() should include the 'search' node."""
    from mentat.graph.workflow import build_graph

    with (
        patch("mentat.graph.workflow.OrchestrationAgent") as MockOrch,
        patch("mentat.graph.workflow.SearchAgent") as MockSearch,
    ):
        MockOrch.return_value.run = MagicMock()
        MockSearch.return_value.run = MagicMock()
        graph = build_graph()

    node_names = list(graph.nodes.keys())
    assert "search" in node_names
