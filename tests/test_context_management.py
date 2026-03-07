"""Tests for the Context Management Agent."""

from unittest.mock import MagicMock

import pytest
from helpers import make_cm_result, make_state
from pydantic import ValidationError

from mentat.core.models import (
    Intent,
    OrchestrationResult,
    RAGAgentResult,
    SearchAgentResult,
    SearchResult,
)

_USER_MESSAGE = "I'm struggling to delegate work to my team."

# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_context_management_result_frozen():
    """ContextManagementResult is frozen — mutation raises ValidationError."""
    result = make_cm_result()
    with pytest.raises((ValidationError, TypeError)):
        result.coaching_brief = "mutated"  # type: ignore[misc]


def test_context_management_result_fields():
    """ContextManagementResult stores all required fields."""
    result = make_cm_result()
    assert result.session_phase == "exploration"
    assert result.tone_guidance == "Warm and Socratic — ask before advising."
    assert "delegation" in result.coaching_brief
    assert len(result.key_information) > 0
    assert len(result.conversation_summary) > 0


# ---------------------------------------------------------------------------
# ContextManagementAgent._build_context
# ---------------------------------------------------------------------------


def test_build_context_includes_user_message(make_agent):
    """_build_context should include the user message."""
    from mentat.agents.context_management import ContextManagementAgent

    agent = make_agent(ContextManagementAgent, _recent_message_count=10)
    state = make_state(user_message=_USER_MESSAGE)
    ctx = agent._build_context(state)
    assert "struggling to delegate" in ctx


def test_build_context_includes_intent(make_agent):
    """_build_context should include the orchestration intent."""
    from mentat.agents.context_management import ContextManagementAgent

    orch = OrchestrationResult(
        intent=Intent.COACHING_SESSION,
        confidence=0.9,
        reasoning="User wants structured coaching guidance.",
        suggested_agents=(),
    )
    agent = make_agent(ContextManagementAgent, _recent_message_count=10)
    state = make_state(orchestration_result=orch)
    ctx = agent._build_context(state)
    assert "coaching-session" in ctx


def test_build_context_no_search_fallback(make_agent):
    """_build_context should note absence of search results."""
    from mentat.agents.context_management import ContextManagementAgent

    agent = make_agent(ContextManagementAgent, _recent_message_count=10)
    state = make_state(search_results=None)
    ctx = agent._build_context(state)
    assert "Search Agent: no results" in ctx


def test_build_context_includes_search_summary(make_agent):
    """_build_context should include search summary when available."""
    from mentat.agents.context_management import ContextManagementAgent

    agent = make_agent(ContextManagementAgent, _recent_message_count=10)
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
    state = make_state(search_results=search)
    ctx = agent._build_context(state)
    assert "clear expectations and trust" in ctx


def test_build_context_includes_rag_summary(make_agent):
    """_build_context should include RAG summary when available."""
    from mentat.agents.context_management import ContextManagementAgent

    agent = make_agent(ContextManagementAgent, _recent_message_count=10)
    rag = RAGAgentResult(
        query="delegation team management",
        summary="User previously discussed wanting to empower their team.",
    )
    state = make_state(rag_results=rag)
    ctx = agent._build_context(state)
    assert "empower their team" in ctx


def test_build_context_no_rag_fallback(make_agent):
    """_build_context should note absence of RAG results."""
    from mentat.agents.context_management import ContextManagementAgent

    agent = make_agent(ContextManagementAgent, _recent_message_count=10)
    state = make_state(rag_results=None)
    ctx = agent._build_context(state)
    assert "RAG Agent: no results" in ctx


def test_build_context_truncates_messages(make_agent):
    """_build_context should include at most recent_message_count messages."""
    from langchain_core.messages import HumanMessage

    from mentat.agents.context_management import ContextManagementAgent

    agent = make_agent(ContextManagementAgent, _recent_message_count=2)
    messages = [HumanMessage(content=f"msg {i}") for i in range(10)]
    state = make_state(messages=messages)
    ctx = agent._build_context(state)
    # Only the last 2 messages should appear
    assert "msg 8" in ctx
    assert "msg 9" in ctx
    assert "msg 0" not in ctx


# ---------------------------------------------------------------------------
# ContextManagementAgent.run
# ---------------------------------------------------------------------------


def test_run_populates_context_management_result(make_agent):
    """run() should populate context_management_result in the returned state."""
    from mentat.agents.context_management import ContextManagementAgent, _ContextBrief

    fake_brief = _ContextBrief(
        session_phase="exploration",
        tone_guidance="Socratic",
        key_information="Team of 5 engineers.",
        conversation_summary="User working on leadership.",
        coaching_brief="Ask about specific delegation examples.",
    )

    agent = make_agent(
        ContextManagementAgent,
        _recent_message_count=10,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )
    agent._call_llm = MagicMock(return_value=fake_brief)

    state = make_state()
    new_state = agent.run(state)

    assert new_state["context_management_result"] is not None
    cm = new_state["context_management_result"]
    assert cm.session_phase == "exploration"
    assert cm.coaching_brief == "Ask about specific delegation examples."
    assert cm.tone_guidance == "Socratic"


def test_run_preserves_other_state_fields(make_agent):
    """run() should pass through other state fields unchanged."""
    from mentat.agents.context_management import ContextManagementAgent, _ContextBrief

    fake_brief = _ContextBrief(
        session_phase="action-planning",
        tone_guidance="Direct",
        key_information="None",
        conversation_summary="Short session.",
        coaching_brief="Set a concrete next step.",
    )

    agent = make_agent(
        ContextManagementAgent,
        _recent_message_count=10,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )
    agent._call_llm = MagicMock(return_value=fake_brief)

    orch = OrchestrationResult(
        intent=Intent.COACHING_SESSION,
        confidence=0.85,
        reasoning="Wants action plan.",
    )
    state = make_state(orchestration_result=orch, final_response="previous")
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

    cm = make_cm_result(coaching_brief="Start with empathy, then ask open questions.")
    state = make_state(context_management_result=cm)
    new_state = format_response(state)

    assert new_state["final_response"] == "Start with empathy, then ask open questions."
    assert "Start with empathy" in new_state["messages"][-1].content


def test_format_response_fallback_without_cm_result():
    """format_response falls back to orchestration result when CM result is absent."""
    from mentat.graph.workflow import format_response

    orch = OrchestrationResult(
        intent=Intent.COACHING_SESSION,
        confidence=0.9,
        reasoning="User wants structured coaching guidance.",
        suggested_agents=(),
    )
    state = make_state(orchestration_result=orch, context_management_result=None)
    new_state = format_response(state)

    assert new_state["final_response"] is not None
    assert "coaching-session" in new_state["final_response"]
    assert "90%" in new_state["final_response"]
