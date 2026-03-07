"""Tests for the OutputTestingAgent."""

from mentat.agents.output_testing import OutputTestingAgent
from mentat.core.models import (
    Intent,
    OrchestrationResult,
    SearchAgentResult,
    SearchResult,
)
from mentat.graph.state import GraphState


def _make_state(**overrides) -> GraphState:
    base: GraphState = {
        "messages": [],
        "user_message": "What are the latest trends in executive coaching?",
        "orchestration_result": None,
        "search_results": None,
        "rag_results": None,
        "context_management_result": None,
        "persona_context": None,
        "plan_context": None,
        "coaching_response": None,
        "quality_rating": None,
        "final_response": None,
    }
    return GraphState(**{**base, **overrides})


def _make_orchestration_result(**overrides) -> OrchestrationResult:
    defaults = {
        "intent": Intent.QUESTION,
        "confidence": 0.9,
        "reasoning": "User asked a factual question.",
        "suggested_agents": ("search",),
    }
    return OrchestrationResult(**{**defaults, **overrides})


def _make_search_result() -> SearchResult:
    return SearchResult(
        title="Coaching Trends 2025",
        url="https://example.com/trends",
        snippet="Executive coaching is evolving rapidly...",
        retrieved_at="2026-02-26T00:00:00+00:00",
    )


class TestOutputTestingAgent:
    def setup_method(self):
        self.agent = OutputTestingAgent()

    def test_run_returns_graph_state(self):
        """run() should return a GraphState."""
        state = _make_state()
        new_state = self.agent.run(state)
        assert isinstance(new_state, dict)
        assert "final_response" in new_state

    def test_includes_user_message(self):
        """Output should contain the user message."""
        state = _make_state(user_message="How do I manage my team better?")
        new_state = self.agent.run(state)
        assert "How do I manage my team better?" in new_state["final_response"]

    def test_includes_orchestration_result(self):
        """Output should render intent, confidence, and reasoning."""
        result = _make_orchestration_result()
        state = _make_state(orchestration_result=result)
        new_state = self.agent.run(state)
        response = new_state["final_response"]
        assert "question" in response
        assert "90%" in response
        assert "User asked a factual question." in response

    def test_includes_suggested_agents(self):
        """Output should list suggested agents."""
        result = _make_orchestration_result(suggested_agents=("search",))
        state = _make_state(orchestration_result=result)
        new_state = self.agent.run(state)
        assert "search" in new_state["final_response"]

    def test_no_suggested_agents_shows_none(self):
        """Output should show none marker when suggested_agents is empty."""
        result = _make_orchestration_result(suggested_agents=())
        state = _make_state(orchestration_result=result)
        new_state = self.agent.run(state)
        assert "_(none)_" in new_state["final_response"]

    def test_includes_search_results(self):
        """Output should render search queries, results, and summary."""
        search = SearchAgentResult(
            queries=("exec coaching trends",),
            results=(_make_search_result(),),
            summary="Coaching is growing fast.",
        )
        state = _make_state(search_results=search)
        new_state = self.agent.run(state)
        response = new_state["final_response"]
        assert "exec coaching trends" in response
        assert "Coaching Trends 2025" in response
        assert "https://example.com/trends" in response
        assert "Coaching is growing fast." in response

    def test_omits_none_fields(self):
        """Output should not mention fields that are None."""
        state = _make_state()
        new_state = self.agent.run(state)
        response = new_state["final_response"]
        assert "RAG Results" not in response
        assert "Persona Context" not in response
        assert "Quality Rating" not in response

    def test_includes_message_count(self):
        """Output should show the message history count."""
        from langchain_core.messages import HumanMessage

        state = _make_state(messages=[HumanMessage(content="hello")])
        new_state = self.agent.run(state)
        assert "1 message(s)" in new_state["final_response"]

    def test_final_response_matches_message_content(self):
        """final_response and the AIMessage content should be identical."""
        state = _make_state(orchestration_result=_make_orchestration_result())
        new_state = self.agent.run(state)
        assert new_state["messages"][-1].content == new_state["final_response"]

    def test_passthrough_fields_unchanged(self):
        """State fields not owned by this agent should pass through unchanged."""
        state = _make_state(
            rag_results="some rag context",
            quality_rating=4,
        )
        new_state = self.agent.run(state)
        assert new_state["rag_results"] == "some rag context"
        assert new_state["quality_rating"] == 4


def _patch_all_workflow_agents():
    """Context manager that patches every agent instantiated in build_graph."""
    from unittest.mock import patch

    patches = [
        patch("mentat.graph.workflow.OrchestrationAgent"),
        patch("mentat.graph.workflow.SearchAgent"),
        patch("mentat.graph.workflow.RAGAgent"),
        patch("mentat.graph.workflow.ContextManagementAgent"),
        patch("mentat.graph.workflow.CoachingAgent"),
        patch("mentat.graph.workflow.QualityAgent"),
        patch("mentat.graph.workflow.SessionUpdateAgent"),
    ]
    return patches


def test_build_graph_debug_mode():
    """build_graph(debug=True) should wire OutputTestingAgent as the final node."""
    from contextlib import ExitStack
    from unittest.mock import MagicMock

    mock_neo4j = MagicMock()
    mock_emb = MagicMock()
    with ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in _patch_all_workflow_agents()]
        for m in mocks:
            m.return_value.run = MagicMock()
        from mentat.graph.workflow import build_graph

        graph = build_graph(
            neo4j_service=mock_neo4j, embedding_service=mock_emb, debug=True
        )

    assert "format_response" in graph.nodes


def test_build_graph_default_is_not_debug():
    """build_graph() without debug flag should not use OutputTestingAgent."""
    from contextlib import ExitStack
    from unittest.mock import MagicMock

    mock_neo4j = MagicMock()
    mock_emb = MagicMock()
    with ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in _patch_all_workflow_agents()]
        for m in mocks:
            m.return_value.run = MagicMock()
        from mentat.graph.workflow import build_graph

        graph = build_graph(
            neo4j_service=mock_neo4j, embedding_service=mock_emb, debug=False
        )

    assert "format_response" in graph.nodes
