"""Tests for the Session Update Agent."""

import os
from unittest.mock import MagicMock

import pytest
from helpers import make_session, make_state

from mentat.core.models import Intent, OrchestrationResult
from mentat.session.models import OnboardingPhase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output(
    phase_complete: bool = False,
    updated_scratchpad: str = "Updated notes.",
    extracted_data: dict | None = None,
    reasoning: str = "No change.",
):
    from mentat.agents.session_update import _ExtractedData, _SessionUpdateOutput

    # Build _ExtractedData from the dict (only known keys, rest ignored)
    ed_kwargs = extracted_data or {}
    return _SessionUpdateOutput(
        phase_complete=phase_complete,
        updated_scratchpad=updated_scratchpad,
        extracted_data=_ExtractedData(**ed_kwargs),
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# SessionUpdateAgent.run — basic operation
# ---------------------------------------------------------------------------


def test_run_with_no_session_state_skips_update(make_agent):
    """run() should return state unchanged when session_state is None."""
    from mentat.agents.session_update import SessionUpdateAgent

    agent = make_agent(SessionUpdateAgent, llm=MagicMock(), prompt_template=MagicMock())
    state = make_state(
        user_message="Yes, that sounds great!",
        coaching_response="Welcome to our coaching journey.",
        final_response="Welcome to our coaching journey.",
        session_state=None,
    )

    # Should not call LLM
    agent._call_llm = MagicMock()
    result = agent.run(state)

    agent._call_llm.assert_not_called()
    assert result.get("session_state") is None


def test_run_advances_turn_count(make_agent):
    """run() should increment turn_count in the updated session."""
    from mentat.agents.session_update import SessionUpdateAgent

    agent = make_agent(SessionUpdateAgent, llm=MagicMock(), prompt_template=MagicMock())
    output = _make_output(phase_complete=False, updated_scratchpad="New notes.")
    agent._call_llm = MagicMock(return_value=output)

    state = make_state(
        user_message="Yes, that sounds great!",
        coaching_response="Welcome to our coaching journey.",
        final_response="Welcome to our coaching journey.",
        session_state=make_session(turn_count=2),
    )
    result = agent.run(state)

    assert result["session_state"] is not None
    assert result["session_state"].turn_count == 3


def test_run_phase_advances_when_complete(make_agent):
    """run() should advance the phase when phase_complete is True."""
    from mentat.agents.session_update import SessionUpdateAgent

    agent = make_agent(SessionUpdateAgent, llm=MagicMock(), prompt_template=MagicMock())
    output = _make_output(phase_complete=True, updated_scratchpad="Phase done.")
    agent._call_llm = MagicMock(return_value=output)

    state = make_state(
        user_message="Yes, that sounds great!",
        coaching_response="Welcome to our coaching journey.",
        final_response="Welcome to our coaching journey.",
        session_state=make_session(phase=OnboardingPhase.SET_EXPECTATIONS.value),
    )
    result = agent.run(state)

    assert result["session_state"] is not None
    assert result["session_state"].phase == OnboardingPhase.BACKGROUND_360.value


def test_run_phase_stays_when_not_complete(make_agent):
    """run() should keep the phase when phase_complete is False."""
    from mentat.agents.session_update import SessionUpdateAgent

    agent = make_agent(SessionUpdateAgent, llm=MagicMock(), prompt_template=MagicMock())
    output = _make_output(phase_complete=False)
    agent._call_llm = MagicMock(return_value=output)

    state = make_state(
        user_message="Yes, that sounds great!",
        coaching_response="Welcome to our coaching journey.",
        final_response="Welcome to our coaching journey.",
        session_state=make_session(phase=OnboardingPhase.BACKGROUND_360.value),
    )
    result = agent.run(state)

    assert result["session_state"] is not None
    assert result["session_state"].phase == OnboardingPhase.BACKGROUND_360.value


def test_run_merges_extracted_data(make_agent):
    """run() should merge extracted_data into collected_data."""
    from mentat.agents.session_update import SessionUpdateAgent

    agent = make_agent(SessionUpdateAgent, llm=MagicMock(), prompt_template=MagicMock())
    output = _make_output(
        phase_complete=False,
        extracted_data={"role": "VP Engineering"},
    )
    agent._call_llm = MagicMock(return_value=output)

    state = make_state(
        user_message="Yes, that sounds great!",
        coaching_response="Welcome to our coaching journey.",
        final_response="Welcome to our coaching journey.",
        session_state=make_session(collected_data={"existing": "value"}),
    )
    result = agent.run(state)

    assert result["session_state"] is not None
    assert result["session_state"].collected_data["role"] == "VP Engineering"
    assert result["session_state"].collected_data["existing"] == "value"


def test_run_preserves_other_state_fields(make_agent):
    """run() should pass through all other state fields unchanged."""
    from mentat.agents.session_update import SessionUpdateAgent

    agent = make_agent(SessionUpdateAgent, llm=MagicMock(), prompt_template=MagicMock())
    output = _make_output()
    agent._call_llm = MagicMock(return_value=output)

    orch = OrchestrationResult(
        intent=Intent.COACHING_SESSION,
        confidence=0.85,
        reasoning="Onboarding.",
    )
    state = make_state(
        user_message="Hello!",
        orchestration_result=orch,
        coaching_response="Welcome to our coaching journey.",
        final_response="Welcome to our coaching journey.",
        session_state=make_session(),
    )
    result = agent.run(state)

    assert result["orchestration_result"] is orch
    assert result["user_message"] == "Hello!"
    assert result["final_response"] == state["final_response"]


# ---------------------------------------------------------------------------
# SessionUpdateAgent._build_context
# ---------------------------------------------------------------------------


def test_build_context_includes_phase(make_agent):
    """_build_context should include the current phase."""
    from mentat.agents.session_update import SessionUpdateAgent

    agent = make_agent(SessionUpdateAgent)
    session = make_session(phase=OnboardingPhase.GOAL_SETTING.value)
    state = make_state(
        user_message="Yes, that sounds great!",
        final_response="Welcome to our coaching journey.",
        session_state=session,
    )
    context = agent._build_context(state, session)
    assert "goal_setting" in context


def test_build_context_includes_user_message(make_agent):
    """_build_context should include the user message."""
    from mentat.agents.session_update import SessionUpdateAgent

    agent = make_agent(SessionUpdateAgent)
    session = make_session()
    state = make_state(user_message="Let's talk about my goals.")
    context = agent._build_context(state, session)
    assert "Let's talk about my goals." in context


def test_build_context_includes_final_response(make_agent):
    """_build_context should include the final coaching response."""
    from mentat.agents.session_update import SessionUpdateAgent

    agent = make_agent(SessionUpdateAgent)
    session = make_session()
    state = make_state(final_response="Great! Let's explore that.")
    context = agent._build_context(state, session)
    assert "Great! Let's explore that." in context


def test_build_context_includes_collected_data(make_agent):
    """_build_context should include collected_data as JSON."""
    from mentat.agents.session_update import SessionUpdateAgent

    agent = make_agent(SessionUpdateAgent)
    session = make_session(collected_data={"role": "CTO"})
    state = make_state(session_state=session)
    context = agent._build_context(state, session)
    assert "CTO" in context


def test_build_context_includes_turn_number(make_agent):
    """_build_context should include the turn number."""
    from mentat.agents.session_update import SessionUpdateAgent

    agent = make_agent(SessionUpdateAgent)
    session = make_session(turn_count=4)
    state = make_state(session_state=session)
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
    from mentat.session.models import OnboardingPhase

    agent = SessionUpdateAgent()
    session = make_session(
        phase=OnboardingPhase.SET_EXPECTATIONS.value,
        scratchpad="",
    )
    state = make_state(
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
