"""Tests for the Context Management Agent."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from mentat.core.models import (
    ContextManagementResult,
    Intent,
    OrchestrationResult,
    RAGAgentResult,
    SearchAgentResult,
    SearchResult,
)
from mentat.graph.state import GraphState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> GraphState:
    base: GraphState = {
        "messages": [],
        "user_message": "I'm struggling to delegate work to my team.",
        "orchestration_result": OrchestrationResult(
            intent=Intent.COACHING_SESSION,
            confidence=0.9,
            reasoning="User wants structured coaching guidance.",
            suggested_agents=(),
        ),
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


def _make_cm_result(**overrides) -> ContextManagementResult:
    defaults = {
        "coaching_brief": "Acknowledge difficulty, then explore delegation blockers.",
        "session_phase": "exploration",
        "tone_guidance": "Warm and Socratic — ask before advising.",
        "key_information": "User leads a team of 5 engineers.",
        "conversation_summary": "User is working on leadership skills.",
    }
    return ContextManagementResult(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_context_management_result_frozen():
    """ContextManagementResult is frozen — mutation raises ValidationError."""
    result = _make_cm_result()
    with pytest.raises((ValidationError, TypeError)):
        result.coaching_brief = "mutated"  # type: ignore[misc]


def test_context_management_result_fields():
    """ContextManagementResult stores all required fields."""
    result = _make_cm_result()
    assert result.session_phase == "exploration"
    assert result.tone_guidance == "Warm and Socratic — ask before advising."
    assert "delegation" in result.coaching_brief
    assert len(result.key_information) > 0
    assert len(result.conversation_summary) > 0


# ---------------------------------------------------------------------------
# ContextManagementAgent._build_context
# ---------------------------------------------------------------------------


def _make_agent_instance():  # type: ignore[return]
    """Create a ContextManagementAgent bypassing __init__ for unit tests."""
    from mentat.agents.context_management import ContextManagementAgent

    with patch("mentat.agents.context_management.BaseAgent.__init__"):
        agent = object.__new__(ContextManagementAgent)
        agent._logger = MagicMock()
        agent._recent_message_count = 10
        agent.llm = MagicMock()
        agent.prompt_template = MagicMock()
    return agent


def test_build_context_includes_user_message():
    """_build_context should include the user message."""
    agent = _make_agent_instance()
    state = _make_state()
    ctx = agent._build_context(state)
    assert "struggling to delegate" in ctx


def test_build_context_includes_intent():
    """_build_context should include the orchestration intent."""
    agent = _make_agent_instance()
    state = _make_state()
    ctx = agent._build_context(state)
    assert "coaching-session" in ctx


def test_build_context_no_search_fallback():
    """_build_context should note absence of search results."""
    agent = _make_agent_instance()
    state = _make_state(search_results=None)
    ctx = agent._build_context(state)
    assert "Search Agent: no results" in ctx


def test_build_context_includes_search_summary():
    """_build_context should include search summary when available."""
    agent = _make_agent_instance()
    search = SearchAgentResult(
        queries=("delegation strategies",),
        results=(
            SearchResult(
                title="How to Delegate",
                url="https://example.com",
                snippet="Key tips for delegation...",
                retrieved_at="2026-02-28T00:00:00+00:00",
            ),
        ),
        summary="Effective delegation requires clear expectations and trust.",
    )
    state = _make_state(search_results=search)
    ctx = agent._build_context(state)
    assert "clear expectations and trust" in ctx


def test_build_context_includes_rag_summary():
    """_build_context should include RAG summary when available."""
    agent = _make_agent_instance()
    rag = RAGAgentResult(
        query="delegation team management",
        summary="User previously discussed wanting to empower their team.",
    )
    state = _make_state(rag_results=rag)
    ctx = agent._build_context(state)
    assert "empower their team" in ctx


def test_build_context_no_rag_fallback():
    """_build_context should note absence of RAG results."""
    agent = _make_agent_instance()
    state = _make_state(rag_results=None)
    ctx = agent._build_context(state)
    assert "RAG Agent: no results" in ctx


def test_build_context_truncates_messages():
    """_build_context should include at most recent_message_count messages."""
    from langchain_core.messages import HumanMessage

    agent = _make_agent_instance()
    agent._recent_message_count = 2
    messages = [HumanMessage(content=f"msg {i}") for i in range(10)]
    state = _make_state(messages=messages)
    ctx = agent._build_context(state)
    # Only the last 2 messages should appear
    assert "msg 8" in ctx
    assert "msg 9" in ctx
    assert "msg 0" not in ctx


# ---------------------------------------------------------------------------
# ContextManagementAgent.run
# ---------------------------------------------------------------------------


def test_run_populates_context_management_result():
    """run() should populate context_management_result in the returned state."""
    from mentat.agents.context_management import ContextManagementAgent, _ContextBrief

    fake_brief = _ContextBrief(
        session_phase="exploration",
        tone_guidance="Socratic",
        key_information="Team of 5 engineers.",
        conversation_summary="User working on leadership.",
        coaching_brief="Ask about specific delegation examples.",
    )

    with patch("mentat.agents.context_management.BaseAgent.__init__"):
        agent = object.__new__(ContextManagementAgent)
        agent._logger = MagicMock()
        agent._recent_message_count = 10
        agent.llm = MagicMock()
        agent.prompt_template = MagicMock()

        # Patch _call_llm to return fake_brief directly
        agent._call_llm = MagicMock(return_value=fake_brief)

        state = _make_state()
        new_state = agent.run(state)

    assert new_state["context_management_result"] is not None
    cm = new_state["context_management_result"]
    assert cm.session_phase == "exploration"
    assert cm.coaching_brief == "Ask about specific delegation examples."
    assert cm.tone_guidance == "Socratic"


def test_run_preserves_other_state_fields():
    """run() should pass through other state fields unchanged."""
    from mentat.agents.context_management import ContextManagementAgent, _ContextBrief

    fake_brief = _ContextBrief(
        session_phase="action-planning",
        tone_guidance="Direct",
        key_information="None",
        conversation_summary="Short session.",
        coaching_brief="Set a concrete next step.",
    )

    with patch("mentat.agents.context_management.BaseAgent.__init__"):
        agent = object.__new__(ContextManagementAgent)
        agent._logger = MagicMock()
        agent._recent_message_count = 10
        agent.llm = MagicMock()
        agent.prompt_template = MagicMock()
        agent._call_llm = MagicMock(return_value=fake_brief)

        orch = OrchestrationResult(
            intent=Intent.COACHING_SESSION,
            confidence=0.85,
            reasoning="Wants action plan.",
        )
        state = _make_state(orchestration_result=orch, final_response="previous")
        new_state = agent.run(state)

    assert new_state["orchestration_result"] is orch
    assert new_state["user_message"] == state["user_message"]
    assert new_state["final_response"] == "previous"


# ---------------------------------------------------------------------------
# format_response integration — uses coaching_brief when CM result present
# ---------------------------------------------------------------------------


def test_format_response_uses_coaching_brief():
    """format_response should use coaching_brief when cm result is present."""
    from mentat.graph.workflow import format_response

    cm = _make_cm_result(coaching_brief="Start with empathy, then ask open questions.")
    state = _make_state(context_management_result=cm)
    new_state = format_response(state)

    assert new_state["final_response"] == "Start with empathy, then ask open questions."
    assert "Start with empathy" in new_state["messages"][-1].content


def test_format_response_fallback_without_cm_result():
    """format_response falls back to orchestration result when CM result is absent."""
    from mentat.graph.workflow import format_response

    state = _make_state(context_management_result=None)
    new_state = format_response(state)

    assert new_state["final_response"] is not None
    assert "coaching-session" in new_state["final_response"]
    assert "90%" in new_state["final_response"]
