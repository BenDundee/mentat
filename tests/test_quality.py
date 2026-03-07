"""Tests for the Quality Agent."""

import os
from unittest.mock import MagicMock

import pytest
from helpers import make_cm_result, make_state

from mentat.core.models import (
    Intent,
    OrchestrationResult,
)

_USER_MESSAGE = "How can I get my team more engaged?"

# ---------------------------------------------------------------------------
# QualityAgent.run — good response (rating > 3)
# ---------------------------------------------------------------------------


def _mock_assessment(rating: int, feedback: str = ""):
    """Return a mock _QualityAssessment-like object."""
    from mentat.agents.quality import _QualityAssessment

    return _QualityAssessment(rating=rating, feedback=feedback)


def test_run_good_response_sets_rating(make_agent):
    """run() should set quality_rating from the LLM assessment."""
    from mentat.agents.quality import QualityAgent

    agent = make_agent(
        QualityAgent,
        _recent_message_count=6,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )
    assessment = _mock_assessment(rating=5, feedback="")
    agent._call_llm = MagicMock(return_value=assessment)

    state = make_state(
        user_message=_USER_MESSAGE,
        coaching_response="What does engagement mean to you right now?",
    )
    new_state = agent.run(state)

    assert new_state["quality_rating"] == 5


def test_run_good_response_clears_feedback(make_agent):
    """run() should set quality_feedback to None when rating > 3."""
    from mentat.agents.quality import QualityAgent

    agent = make_agent(
        QualityAgent,
        _recent_message_count=6,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )
    assessment = _mock_assessment(rating=4, feedback="")
    agent._call_llm = MagicMock(return_value=assessment)

    state = make_state(
        user_message=_USER_MESSAGE,
        coaching_response="What does engagement mean to you right now?",
    )
    new_state = agent.run(state)

    assert new_state["quality_feedback"] is None


# ---------------------------------------------------------------------------
# QualityAgent.run — poor response (rating ≤ 3)
# ---------------------------------------------------------------------------


def test_run_poor_response_sets_feedback(make_agent):
    """run() should populate quality_feedback when rating ≤ 3."""
    from mentat.agents.quality import QualityAgent

    agent = make_agent(
        QualityAgent,
        _recent_message_count=6,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )
    assessment = _mock_assessment(
        rating=2, feedback="Response is too generic; ask a specific question."
    )
    agent._call_llm = MagicMock(return_value=assessment)

    state = make_state(
        user_message=_USER_MESSAGE,
        coaching_response="What does engagement mean to you right now?",
    )
    new_state = agent.run(state)

    assert new_state["quality_feedback"] == (
        "Response is too generic; ask a specific question."
    )


def test_run_poor_response_sets_rating(make_agent):
    """run() should set quality_rating correctly for a poor response."""
    from mentat.agents.quality import QualityAgent

    agent = make_agent(
        QualityAgent,
        _recent_message_count=6,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )
    assessment = _mock_assessment(rating=3, feedback="Needs more personalisation.")
    agent._call_llm = MagicMock(return_value=assessment)

    state = make_state(
        user_message=_USER_MESSAGE,
        coaching_response="What does engagement mean to you right now?",
    )
    new_state = agent.run(state)

    assert new_state["quality_rating"] == 3


def test_run_boundary_rating_3_sets_feedback(make_agent):
    """run() should set quality_feedback when rating == 3 (boundary case)."""
    from mentat.agents.quality import QualityAgent

    agent = make_agent(
        QualityAgent,
        _recent_message_count=6,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )
    assessment = _mock_assessment(rating=3, feedback="Improve specificity.")
    agent._call_llm = MagicMock(return_value=assessment)

    state = make_state(
        user_message=_USER_MESSAGE,
        coaching_response="What does engagement mean to you right now?",
    )
    new_state = agent.run(state)

    assert new_state["quality_feedback"] == "Improve specificity."


def test_run_boundary_rating_4_clears_feedback(make_agent):
    """run() should NOT set quality_feedback when rating == 4 (boundary case)."""
    from mentat.agents.quality import QualityAgent

    agent = make_agent(
        QualityAgent,
        _recent_message_count=6,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )
    assessment = _mock_assessment(rating=4, feedback="Minor tweak possible.")
    agent._call_llm = MagicMock(return_value=assessment)

    state = make_state(
        user_message=_USER_MESSAGE,
        coaching_response="What does engagement mean to you right now?",
    )
    new_state = agent.run(state)

    assert new_state["quality_feedback"] is None


# ---------------------------------------------------------------------------
# QualityAgent.run — passthrough
# ---------------------------------------------------------------------------


def test_run_preserves_other_state_fields(make_agent):
    """run() should pass through all other state fields unchanged."""
    from mentat.agents.quality import QualityAgent

    agent = make_agent(
        QualityAgent,
        _recent_message_count=6,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )
    assessment = _mock_assessment(rating=5, feedback="")
    agent._call_llm = MagicMock(return_value=assessment)

    orch = OrchestrationResult(
        intent=Intent.COACHING_SESSION,
        confidence=0.9,
        reasoning="Coaching.",
    )
    cm = make_cm_result()
    state = make_state(
        user_message=_USER_MESSAGE,
        coaching_response="What does engagement mean to you right now?",
        orchestration_result=orch,
        context_management_result=cm,
        coaching_attempts=2,
    )
    new_state = agent.run(state)

    assert new_state["orchestration_result"] is orch
    assert new_state["user_message"] == state["user_message"]
    assert new_state["context_management_result"] is cm
    assert new_state["coaching_response"] == state["coaching_response"]
    assert new_state["coaching_attempts"] == 2


# ---------------------------------------------------------------------------
# QualityAgent._build_context
# ---------------------------------------------------------------------------


def test_build_context_includes_coaching_response(make_agent):
    """_build_context should include the coaching response."""
    from mentat.agents.quality import QualityAgent

    agent = make_agent(QualityAgent, _recent_message_count=6)
    state = make_state(coaching_response="Let's explore that question together.")
    context = agent._build_context(state)
    assert "Let's explore that question together." in context


def test_build_context_includes_user_message(make_agent):
    """_build_context should include the user message."""
    from mentat.agents.quality import QualityAgent

    agent = make_agent(QualityAgent, _recent_message_count=6)
    state = make_state(user_message=_USER_MESSAGE)
    context = agent._build_context(state)
    assert "How can I get my team more engaged?" in context


def test_build_context_includes_coaching_brief_when_present(make_agent):
    """_build_context should include the coaching brief when cm_result is set."""
    from mentat.agents.quality import QualityAgent

    agent = make_agent(QualityAgent, _recent_message_count=6)
    cm = make_cm_result(coaching_brief="Use the GROW model.")
    state = make_state(context_management_result=cm)
    context = agent._build_context(state)
    assert "GROW model" in context


def test_build_context_handles_no_cm_result(make_agent):
    """_build_context should not raise when context_management_result is None."""
    from mentat.agents.quality import QualityAgent

    agent = make_agent(QualityAgent, _recent_message_count=6)
    state = make_state(user_message=_USER_MESSAGE, context_management_result=None)
    context = agent._build_context(state)
    assert "How can I get my team more engaged?" in context


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def test_route_after_quality_low_rating_routes_to_coaching():
    """_route_after_quality returns 'coaching' for rating ≤ 3 within attempt limit."""
    from mentat.graph.workflow import _route_after_quality

    state = make_state(quality_rating=2, coaching_attempts=1)
    assert _route_after_quality(state) == "coaching"


def test_route_after_quality_high_rating_routes_to_format_response():
    """_route_after_quality should return 'format_response' for rating > 3."""
    from mentat.graph.workflow import _route_after_quality

    state = make_state(quality_rating=4, coaching_attempts=1)
    assert _route_after_quality(state) == "format_response"


def test_route_after_quality_max_attempts_routes_to_format_response():
    """_route_after_quality routes to format_response when max attempts reached."""
    from mentat.graph.workflow import _MAX_COACHING_ATTEMPTS, _route_after_quality

    state = make_state(quality_rating=1, coaching_attempts=_MAX_COACHING_ATTEMPTS)
    assert _route_after_quality(state) == "format_response"


def test_route_after_quality_none_rating_routes_to_format_response():
    """_route_after_quality should route to format_response when rating is None."""
    from mentat.graph.workflow import _route_after_quality

    state = make_state(quality_rating=None, coaching_attempts=1)
    assert _route_after_quality(state) == "format_response"


# ---------------------------------------------------------------------------
# Integration test (skipped unless OPENROUTER_API_KEY is set)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
def test_quality_agent_real_llm():
    """Integration test: QualityAgent produces a valid assessment with a real LLM."""
    from mentat.agents.quality import QualityAgent

    agent = QualityAgent()
    state = make_state(
        user_message=_USER_MESSAGE,
        coaching_response=(
            "What does engagement actually look like on your team right now?"
        ),
        context_management_result=make_cm_result(
            coaching_brief="Explore what engagement means to the user.",
            session_phase="exploration",
            tone_guidance="Warm and curious.",
            key_information="Team of 8 engineers.",
            conversation_summary="User wants to improve team engagement.",
        ),
    )
    new_state = agent.run(state)

    assert new_state["quality_rating"] is not None
    assert 1 <= new_state["quality_rating"] <= 5
