"""Tests for the Quality Agent."""

import os
from unittest.mock import MagicMock, patch

import pytest

from mentat.core.models import (
    ContextManagementResult,
    Intent,
    OrchestrationResult,
)
from mentat.graph.state import GraphState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> GraphState:
    base: GraphState = {
        "messages": [],
        "user_message": "How can I get my team more engaged?",
        "orchestration_result": OrchestrationResult(
            intent=Intent.COACHING_SESSION,
            confidence=0.9,
            reasoning="User wants coaching on engagement.",
            suggested_agents=(),
        ),
        "search_results": None,
        "rag_results": None,
        "context_management_result": None,
        "persona_context": None,
        "plan_context": None,
        "coaching_response": "What does engagement mean to you right now?",
        "quality_rating": None,
        "quality_feedback": None,
        "coaching_attempts": None,
        "final_response": None,
    }
    return GraphState(**{**base, **overrides})


def _make_cm_result(**overrides) -> ContextManagementResult:
    defaults = {
        "coaching_brief": "Explore what engagement means to the user.",
        "session_phase": "exploration",
        "tone_guidance": "Warm and curious.",
        "key_information": "Team of 8 engineers.",
        "conversation_summary": "User wants to improve team engagement.",
    }
    return ContextManagementResult(**{**defaults, **overrides})


def _make_agent_instance():  # type: ignore[return]
    """Create a QualityAgent bypassing __init__ for unit tests."""
    from mentat.agents.quality import QualityAgent

    with patch("mentat.agents.quality.BaseAgent.__init__"):
        agent = object.__new__(QualityAgent)
        agent._logger = MagicMock()
        agent._recent_message_count = 6
        agent.llm = MagicMock()
        agent.prompt_template = MagicMock()
    return agent


def _mock_assessment(rating: int, feedback: str = ""):
    """Return a mock _QualityAssessment-like object."""
    from mentat.agents.quality import _QualityAssessment

    return _QualityAssessment(rating=rating, feedback=feedback)


# ---------------------------------------------------------------------------
# QualityAgent.run — good response (rating > 3)
# ---------------------------------------------------------------------------


def test_run_good_response_sets_rating():
    """run() should set quality_rating from the LLM assessment."""
    agent = _make_agent_instance()
    assessment = _mock_assessment(rating=5, feedback="")
    agent._call_llm = MagicMock(return_value=assessment)

    state = _make_state()
    new_state = agent.run(state)

    assert new_state["quality_rating"] == 5


def test_run_good_response_clears_feedback():
    """run() should set quality_feedback to None when rating > 3."""
    agent = _make_agent_instance()
    assessment = _mock_assessment(rating=4, feedback="")
    agent._call_llm = MagicMock(return_value=assessment)

    state = _make_state()
    new_state = agent.run(state)

    assert new_state["quality_feedback"] is None


# ---------------------------------------------------------------------------
# QualityAgent.run — poor response (rating ≤ 3)
# ---------------------------------------------------------------------------


def test_run_poor_response_sets_feedback():
    """run() should populate quality_feedback when rating ≤ 3."""
    agent = _make_agent_instance()
    assessment = _mock_assessment(
        rating=2, feedback="Response is too generic; ask a specific question."
    )
    agent._call_llm = MagicMock(return_value=assessment)

    state = _make_state()
    new_state = agent.run(state)

    assert new_state["quality_feedback"] == (
        "Response is too generic; ask a specific question."
    )


def test_run_poor_response_sets_rating():
    """run() should set quality_rating correctly for a poor response."""
    agent = _make_agent_instance()
    assessment = _mock_assessment(rating=3, feedback="Needs more personalisation.")
    agent._call_llm = MagicMock(return_value=assessment)

    state = _make_state()
    new_state = agent.run(state)

    assert new_state["quality_rating"] == 3


def test_run_boundary_rating_3_sets_feedback():
    """run() should set quality_feedback when rating == 3 (boundary case)."""
    agent = _make_agent_instance()
    assessment = _mock_assessment(rating=3, feedback="Improve specificity.")
    agent._call_llm = MagicMock(return_value=assessment)

    state = _make_state()
    new_state = agent.run(state)

    assert new_state["quality_feedback"] == "Improve specificity."


def test_run_boundary_rating_4_clears_feedback():
    """run() should NOT set quality_feedback when rating == 4 (boundary case)."""
    agent = _make_agent_instance()
    assessment = _mock_assessment(rating=4, feedback="Minor tweak possible.")
    agent._call_llm = MagicMock(return_value=assessment)

    state = _make_state()
    new_state = agent.run(state)

    assert new_state["quality_feedback"] is None


# ---------------------------------------------------------------------------
# QualityAgent.run — passthrough
# ---------------------------------------------------------------------------


def test_run_preserves_other_state_fields():
    """run() should pass through all other state fields unchanged."""
    agent = _make_agent_instance()
    assessment = _mock_assessment(rating=5, feedback="")
    agent._call_llm = MagicMock(return_value=assessment)

    orch = OrchestrationResult(
        intent=Intent.COACHING_SESSION,
        confidence=0.9,
        reasoning="Coaching.",
    )
    cm = _make_cm_result()
    state = _make_state(
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


def test_build_context_includes_coaching_response():
    """_build_context should include the coaching response."""
    agent = _make_agent_instance()
    state = _make_state(coaching_response="Let's explore that question together.")
    context = agent._build_context(state)
    assert "Let's explore that question together." in context


def test_build_context_includes_user_message():
    """_build_context should include the user message."""
    agent = _make_agent_instance()
    state = _make_state()
    context = agent._build_context(state)
    assert "How can I get my team more engaged?" in context


def test_build_context_includes_coaching_brief_when_present():
    """_build_context should include the coaching brief when cm_result is set."""
    agent = _make_agent_instance()
    cm = _make_cm_result(coaching_brief="Use the GROW model.")
    state = _make_state(context_management_result=cm)
    context = agent._build_context(state)
    assert "GROW model" in context


def test_build_context_handles_no_cm_result():
    """_build_context should not raise when context_management_result is None."""
    agent = _make_agent_instance()
    state = _make_state(context_management_result=None)
    context = agent._build_context(state)
    assert "How can I get my team more engaged?" in context


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def test_route_after_quality_low_rating_routes_to_coaching():
    """_route_after_quality returns 'coaching' for rating ≤ 3 within attempt limit."""
    from mentat.graph.workflow import _route_after_quality

    state = _make_state(quality_rating=2, coaching_attempts=1)
    assert _route_after_quality(state) == "coaching"


def test_route_after_quality_high_rating_routes_to_format_response():
    """_route_after_quality should return 'format_response' for rating > 3."""
    from mentat.graph.workflow import _route_after_quality

    state = _make_state(quality_rating=4, coaching_attempts=1)
    assert _route_after_quality(state) == "format_response"


def test_route_after_quality_max_attempts_routes_to_format_response():
    """_route_after_quality routes to format_response when max attempts reached."""
    from mentat.graph.workflow import _MAX_COACHING_ATTEMPTS, _route_after_quality

    state = _make_state(quality_rating=1, coaching_attempts=_MAX_COACHING_ATTEMPTS)
    assert _route_after_quality(state) == "format_response"


def test_route_after_quality_none_rating_routes_to_format_response():
    """_route_after_quality should route to format_response when rating is None."""
    from mentat.graph.workflow import _route_after_quality

    state = _make_state(quality_rating=None, coaching_attempts=1)
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
    state = _make_state(
        coaching_response=(
            "What does engagement actually look like on your team right now?"
        ),
        context_management_result=_make_cm_result(),
    )
    new_state = agent.run(state)

    assert new_state["quality_rating"] is not None
    assert 1 <= new_state["quality_rating"] <= 5
