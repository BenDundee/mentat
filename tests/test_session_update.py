"""Tests for the Session Update Agent."""

import os
from unittest.mock import MagicMock, patch

import pytest

from mentat.core.models import Intent, OrchestrationResult
from mentat.graph.state import GraphState
from mentat.session.models import (
    ConversationSession,
    ConversationType,
    OnboardingPhase,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(**overrides) -> ConversationSession:
    defaults = {
        "session_id": "test-session",
        "conversation_type": ConversationType.ONBOARDING,
        "phase": OnboardingPhase.SET_EXPECTATIONS.value,
        "scratchpad": "",
        "collected_data": {},
        "turn_count": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    return ConversationSession(**{**defaults, **overrides})


def _make_state(**overrides) -> GraphState:
    base: GraphState = {
        "messages": [],
        "user_message": "Yes, that sounds great!",
        "orchestration_result": OrchestrationResult(
            intent=Intent.COACHING_SESSION,
            confidence=0.9,
            reasoning="Onboarding conversation.",
            suggested_agents=(),
        ),
        "search_results": None,
        "rag_results": None,
        "context_management_result": None,
        "persona_context": None,
        "plan_context": None,
        "coaching_response": "Welcome to our coaching journey.",
        "quality_rating": None,
        "quality_feedback": None,
        "coaching_attempts": None,
        "final_response": "Welcome to our coaching journey.",
        "session_state": _make_session(),
    }
    return GraphState(**{**base, **overrides})


def _make_agent_instance():  # type: ignore[return]
    """Create a SessionUpdateAgent bypassing __init__ for unit tests."""
    from mentat.agents.session_update import SessionUpdateAgent

    with patch("mentat.agents.session_update.BaseAgent.__init__"):
        agent = object.__new__(SessionUpdateAgent)
        agent._logger = MagicMock()
        agent.llm = MagicMock()
        agent.prompt_template = MagicMock()
    return agent


def _make_output(
    phase_complete: bool = False,
    updated_scratchpad: str = "Updated notes.",
    extracted_data: dict | None = None,
    reasoning: str = "No change.",
):
    from mentat.agents.session_update import _SessionUpdateOutput

    return _SessionUpdateOutput(
        phase_complete=phase_complete,
        updated_scratchpad=updated_scratchpad,
        extracted_data=extracted_data or {},
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# SessionUpdateAgent.run — basic operation
# ---------------------------------------------------------------------------


def test_run_with_no_session_state_skips_update():
    """run() should return state unchanged when session_state is None."""
    agent = _make_agent_instance()
    state = _make_state(session_state=None)

    # Should not call LLM
    agent._call_llm = MagicMock()
    result = agent.run(state)

    agent._call_llm.assert_not_called()
    assert result.get("session_state") is None


def test_run_advances_turn_count():
    """run() should increment turn_count in the updated session."""
    agent = _make_agent_instance()
    output = _make_output(phase_complete=False, updated_scratchpad="New notes.")
    agent._call_llm = MagicMock(return_value=output)

    state = _make_state(session_state=_make_session(turn_count=2))
    result = agent.run(state)

    assert result["session_state"] is not None
    assert result["session_state"].turn_count == 3


def test_run_phase_advances_when_complete():
    """run() should advance the phase when phase_complete is True."""
    agent = _make_agent_instance()
    output = _make_output(phase_complete=True, updated_scratchpad="Phase done.")
    agent._call_llm = MagicMock(return_value=output)

    session = _make_session(phase=OnboardingPhase.SET_EXPECTATIONS.value)
    state = _make_state(session_state=session)
    result = agent.run(state)

    assert result["session_state"] is not None
    assert result["session_state"].phase == OnboardingPhase.BACKGROUND_360.value


def test_run_phase_stays_when_not_complete():
    """run() should keep the phase when phase_complete is False."""
    agent = _make_agent_instance()
    output = _make_output(phase_complete=False)
    agent._call_llm = MagicMock(return_value=output)

    session = _make_session(phase=OnboardingPhase.BACKGROUND_360.value)
    state = _make_state(session_state=session)
    result = agent.run(state)

    assert result["session_state"] is not None
    assert result["session_state"].phase == OnboardingPhase.BACKGROUND_360.value


def test_run_merges_extracted_data():
    """run() should merge extracted_data into collected_data."""
    agent = _make_agent_instance()
    output = _make_output(
        phase_complete=False,
        extracted_data={"role": "VP Engineering"},
    )
    agent._call_llm = MagicMock(return_value=output)

    session = _make_session(collected_data={"existing": "value"})
    state = _make_state(session_state=session)
    result = agent.run(state)

    assert result["session_state"] is not None
    assert result["session_state"].collected_data["role"] == "VP Engineering"
    assert result["session_state"].collected_data["existing"] == "value"


def test_run_preserves_other_state_fields():
    """run() should pass through all other state fields unchanged."""
    agent = _make_agent_instance()
    output = _make_output()
    agent._call_llm = MagicMock(return_value=output)

    orch = OrchestrationResult(
        intent=Intent.COACHING_SESSION,
        confidence=0.85,
        reasoning="Onboarding.",
    )
    state = _make_state(orchestration_result=orch, user_message="Hello!")
    result = agent.run(state)

    assert result["orchestration_result"] is orch
    assert result["user_message"] == "Hello!"
    assert result["final_response"] == state["final_response"]


# ---------------------------------------------------------------------------
# SessionUpdateAgent._build_context
# ---------------------------------------------------------------------------


def test_build_context_includes_phase():
    """_build_context should include the current phase."""
    agent = _make_agent_instance()
    session = _make_session(phase=OnboardingPhase.GOAL_SETTING.value)
    state = _make_state(session_state=session)
    context = agent._build_context(state, session)
    assert "goal_setting" in context


def test_build_context_includes_user_message():
    """_build_context should include the user message."""
    agent = _make_agent_instance()
    session = _make_session()
    state = _make_state(user_message="Let's talk about my goals.")
    context = agent._build_context(state, session)
    assert "Let's talk about my goals." in context


def test_build_context_includes_final_response():
    """_build_context should include the final coaching response."""
    agent = _make_agent_instance()
    session = _make_session()
    state = _make_state(final_response="Great! Let's explore that.")
    context = agent._build_context(state, session)
    assert "Great! Let's explore that." in context


def test_build_context_includes_collected_data():
    """_build_context should include collected_data as JSON."""
    agent = _make_agent_instance()
    session = _make_session(collected_data={"role": "CTO"})
    state = _make_state(session_state=session)
    context = agent._build_context(state, session)
    assert "CTO" in context


def test_build_context_includes_turn_number():
    """_build_context should include the turn number."""
    agent = _make_agent_instance()
    session = _make_session(turn_count=4)
    state = _make_state(session_state=session)
    context = agent._build_context(state, session)
    assert "5" in context  # turn_count + 1


# ---------------------------------------------------------------------------
# Integration test (skipped unless OPENROUTER_API_KEY is set)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
def test_session_update_agent_real_llm():
    """Integration test: SessionUpdateAgent produces a valid update with a real LLM."""
    from mentat.agents.session_update import SessionUpdateAgent

    agent = SessionUpdateAgent()
    session = _make_session(
        phase=OnboardingPhase.SET_EXPECTATIONS.value,
        scratchpad="",
    )
    state = _make_state(
        session_state=session,
        user_message="Yes, I'm happy to proceed with the coaching.",
        final_response=(
            "Wonderful! I'm glad you're on board. Let's start by learning "
            "about your background and current situation."
        ),
    )
    result = agent.run(state)

    updated_session = result.get("session_state")
    assert updated_session is not None
    # Should have advanced or stayed at set_expectations
    assert updated_session.phase in [
        OnboardingPhase.SET_EXPECTATIONS.value,
        OnboardingPhase.BACKGROUND_360.value,
    ]
