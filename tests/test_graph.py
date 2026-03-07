"""Tests for the LangGraph workflow."""

from unittest.mock import MagicMock, patch

from mentat.core.models import Intent, OrchestrationResult
from mentat.graph.state import GraphState
from mentat.graph.workflow import (
    _MAX_COACHING_ATTEMPTS,
    _route_after_orchestration,
    _route_after_quality,
    format_response,
)


def _make_mock_neo4j() -> MagicMock:
    """Return a minimal Neo4jService mock."""
    return MagicMock()


def _make_mock_embedding() -> MagicMock:
    """Return a minimal EmbeddingService mock."""
    return MagicMock()


def _make_state(**overrides) -> GraphState:
    base: GraphState = {
        "messages": [],
        "user_message": "Hello",
        "orchestration_result": None,
        "search_results": None,
        "rag_results": None,
        "context_management_result": None,
        "persona_context": None,
        "plan_context": None,
        "coaching_response": None,
        "quality_rating": None,
        "quality_feedback": None,
        "coaching_attempts": None,
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


def _patch_all_agents():
    """Context manager that patches all agents in the workflow module."""
    return (
        patch("mentat.graph.workflow.OrchestrationAgent"),
        patch("mentat.graph.workflow.SearchAgent"),
        patch("mentat.graph.workflow.RAGAgent"),
        patch("mentat.graph.workflow.ContextManagementAgent"),
        patch("mentat.graph.workflow.CoachingAgent"),
        patch("mentat.graph.workflow.QualityAgent"),
    )


def test_build_graph_has_expected_nodes():
    """build_graph() should include all pipeline nodes."""
    from mentat.graph.workflow import build_graph

    mock_neo4j = _make_mock_neo4j()
    mock_emb = _make_mock_embedding()
    with (
        patch("mentat.graph.workflow.OrchestrationAgent") as MockOrch,
        patch("mentat.graph.workflow.SearchAgent") as MockSearch,
        patch("mentat.graph.workflow.RAGAgent") as MockRAG,
        patch("mentat.graph.workflow.ContextManagementAgent") as MockCM,
        patch("mentat.graph.workflow.CoachingAgent") as MockCoach,
        patch("mentat.graph.workflow.QualityAgent") as MockQuality,
    ):
        MockOrch.return_value.run = MagicMock()
        MockSearch.return_value.run = MagicMock()
        MockRAG.return_value.run = MagicMock()
        MockCM.return_value.run = MagicMock()
        MockCoach.return_value.run = MagicMock()
        MockQuality.return_value.run = MagicMock()
        graph = build_graph(neo4j_service=mock_neo4j, embedding_service=mock_emb)

    node_names = list(graph.nodes.keys())
    assert "orchestration" in node_names
    assert "search" in node_names
    assert "rag" in node_names
    assert "context_management" in node_names
    assert "coaching" in node_names
    assert "quality" in node_names
    assert "format_response" in node_names


def test_compile_graph_succeeds():
    """compile_graph() should return a compiled graph without raising."""
    from mentat.graph.workflow import compile_graph

    mock_neo4j = _make_mock_neo4j()
    mock_emb = _make_mock_embedding()
    with (
        patch("mentat.graph.workflow.OrchestrationAgent") as MockOrch,
        patch("mentat.graph.workflow.SearchAgent") as MockSearch,
        patch("mentat.graph.workflow.RAGAgent") as MockRAG,
        patch("mentat.graph.workflow.ContextManagementAgent") as MockCM,
        patch("mentat.graph.workflow.CoachingAgent") as MockCoach,
        patch("mentat.graph.workflow.QualityAgent") as MockQuality,
    ):
        MockOrch.return_value.run = MagicMock()
        MockSearch.return_value.run = MagicMock()
        MockRAG.return_value.run = MagicMock()
        MockCM.return_value.run = MagicMock()
        MockCoach.return_value.run = MagicMock()
        MockQuality.return_value.run = MagicMock()
        compiled = compile_graph(neo4j_service=mock_neo4j, embedding_service=mock_emb)

    assert compiled is not None


def test_route_with_search():
    """Returns ['search'] when suggested_agents contains only 'search'."""
    result = OrchestrationResult(
        intent=Intent.QUESTION,
        confidence=0.9,
        reasoning="Needs current info.",
        suggested_agents=("search",),
    )
    state = _make_state(orchestration_result=result)
    assert _route_after_orchestration(state) == ["search"]


def test_route_with_rag():
    """Returns ['rag'] when suggested_agents contains only 'rag'."""
    result = OrchestrationResult(
        intent=Intent.COACHING_SESSION,
        confidence=0.9,
        reasoning="Needs past context.",
        suggested_agents=("rag",),
    )
    state = _make_state(orchestration_result=result)
    assert _route_after_orchestration(state) == ["rag"]


def test_route_with_both_agents_returns_parallel_targets():
    """Returns ['search', 'rag'] when both are suggested, enabling parallel fan-out."""
    result = OrchestrationResult(
        intent=Intent.QUESTION,
        confidence=0.9,
        reasoning="Needs both current info and past context.",
        suggested_agents=("search", "rag"),
    )
    state = _make_state(orchestration_result=result)
    assert _route_after_orchestration(state) == ["search", "rag"]


def test_route_without_agents():
    """Returns ['context_management'] when suggested_agents is empty."""
    result = OrchestrationResult(
        intent=Intent.CHECK_IN,
        confidence=0.95,
        reasoning="Simple check-in.",
        suggested_agents=(),
    )
    state = _make_state(orchestration_result=result)
    assert _route_after_orchestration(state) == ["context_management"]


def test_route_none_result():
    """Returns ['context_management'] when orchestration_result is None."""
    state = _make_state(orchestration_result=None)
    assert _route_after_orchestration(state) == ["context_management"]


# ---------------------------------------------------------------------------
# _route_after_quality
# ---------------------------------------------------------------------------


def test_route_after_quality_low_rating_routes_to_coaching():
    """Returns 'coaching' when rating ≤ 3 and attempts < max."""
    state = _make_state(quality_rating=2, coaching_attempts=1)
    assert _route_after_quality(state) == "coaching"


def test_route_after_quality_high_rating_routes_to_format_response():
    """Returns 'format_response' when rating > 3."""
    state = _make_state(quality_rating=4, coaching_attempts=1)
    assert _route_after_quality(state) == "format_response"


def test_route_after_quality_max_attempts_stops_loop():
    """Returns 'format_response' when max attempts reached, even with low rating."""
    state = _make_state(quality_rating=1, coaching_attempts=_MAX_COACHING_ATTEMPTS)
    assert _route_after_quality(state) == "format_response"


def test_route_after_quality_none_rating_routes_to_format_response():
    """Returns 'format_response' when quality_rating is None."""
    state = _make_state(quality_rating=None, coaching_attempts=1)
    assert _route_after_quality(state) == "format_response"


# ---------------------------------------------------------------------------
# format_response passthrough of new fields
# ---------------------------------------------------------------------------


def test_format_response_passes_through_quality_fields():
    """format_response passes through quality_rating, quality_feedback, attempts."""
    from mentat.core.models import ContextManagementResult

    cm = ContextManagementResult(
        coaching_brief="Use GROW model.",
        session_phase="exploration",
        tone_guidance="Warm.",
        key_information="",
        conversation_summary="",
    )
    state = _make_state(
        coaching_response="Good coaching response.",
        context_management_result=cm,
        quality_rating=4,
        quality_feedback=None,
        coaching_attempts=1,
    )
    new_state = format_response(state)

    assert new_state["quality_rating"] == 4
    assert new_state["quality_feedback"] is None
    assert new_state["coaching_attempts"] == 1
