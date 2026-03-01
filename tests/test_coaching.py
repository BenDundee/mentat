"""Tests for the Coaching Agent."""

from unittest.mock import MagicMock, patch

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
        "quality_feedback": None,
        "coaching_attempts": None,
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


def _make_agent_instance():  # type: ignore[return]
    """Create a CoachingAgent bypassing __init__ for unit tests."""
    from mentat.agents.coaching import CoachingAgent

    with patch("mentat.agents.coaching.BaseAgent.__init__"):
        agent = object.__new__(CoachingAgent)
        agent._logger = MagicMock()
        agent._recent_message_count = 10
        agent.llm = MagicMock()
        agent.prompt_template = MagicMock()
    return agent


# ---------------------------------------------------------------------------
# CoachingAgent.run
# ---------------------------------------------------------------------------


def test_run_populates_coaching_response():
    """run() should populate coaching_response in the returned state."""
    agent = _make_agent_instance()

    fake_llm_response = MagicMock()
    fake_llm_response.content = "What makes delegation feel so difficult right now?"

    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_llm_response
    agent.prompt_template.__or__ = MagicMock(return_value=chain_mock)

    cm = _make_cm_result()
    state = _make_state(context_management_result=cm)
    new_state = agent.run(state)

    assert (
        new_state["coaching_response"]
        == "What makes delegation feel so difficult right now?"
    )


def test_run_increments_coaching_attempts():
    """run() should increment coaching_attempts on each invocation."""
    agent = _make_agent_instance()

    fake_llm_response = MagicMock()
    fake_llm_response.content = "Coaching reply."
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_llm_response
    agent.prompt_template.__or__ = MagicMock(return_value=chain_mock)

    state = _make_state(coaching_attempts=None)
    new_state = agent.run(state)
    assert new_state["coaching_attempts"] == 1

    # Simulate second attempt (e.g. after quality rewrite loop)
    state2 = _make_state(coaching_attempts=1)
    new_state2 = agent.run(state2)
    assert new_state2["coaching_attempts"] == 2


def test_run_rewrite_includes_feedback_in_prompt():
    """run() should include quality_feedback in the prompt when set."""
    agent = _make_agent_instance()

    fake_llm_response = MagicMock()
    fake_llm_response.content = "Improved coaching reply."
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_llm_response
    agent.prompt_template.__or__ = MagicMock(return_value=chain_mock)

    state = _make_state(
        coaching_response="Previous poor response.",
        quality_feedback="Be more specific; ask about a concrete example.",
        coaching_attempts=1,
    )
    agent.run(state)

    # Inspect what was passed to the chain
    call_args = chain_mock.invoke.call_args
    prompt_input = call_args[0][0]["user_message"]
    assert "REWRITE INSTRUCTIONS" in prompt_input
    assert "Be more specific" in prompt_input
    assert "Previous poor response." in prompt_input


def test_run_preserves_other_state_fields():
    """run() should pass through all other state fields unchanged."""
    agent = _make_agent_instance()

    fake_llm_response = MagicMock()
    fake_llm_response.content = "Coaching reply."
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_llm_response
    agent.prompt_template.__or__ = MagicMock(return_value=chain_mock)

    orch = OrchestrationResult(
        intent=Intent.COACHING_SESSION,
        confidence=0.85,
        reasoning="Wants guidance.",
    )
    cm = _make_cm_result()
    state = _make_state(
        orchestration_result=orch,
        context_management_result=cm,
        final_response="previous",
    )
    new_state = agent.run(state)

    assert new_state["orchestration_result"] is orch
    assert new_state["user_message"] == state["user_message"]
    assert new_state["context_management_result"] is cm
    assert new_state["final_response"] == "previous"


def test_run_handles_no_cm_result():
    """run() should log a warning and still set coaching_response with no cm_result."""
    agent = _make_agent_instance()

    fake_llm_response = MagicMock()
    fake_llm_response.content = "Let me help you with that."
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_llm_response
    agent.prompt_template.__or__ = MagicMock(return_value=chain_mock)

    state = _make_state(context_management_result=None)
    new_state = agent.run(state)

    assert new_state["coaching_response"] == "Let me help you with that."
    agent._logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# CoachingAgent._build_prompt_input
# ---------------------------------------------------------------------------


def test_build_prompt_input_includes_user_message():
    """_build_prompt_input should include the user message."""
    agent = _make_agent_instance()
    state = _make_state()
    prompt = agent._build_prompt_input(state)
    assert "struggling to delegate" in prompt


def test_build_prompt_input_includes_coaching_brief():
    """_build_prompt_input should include the coaching brief when cm_result present."""
    agent = _make_agent_instance()
    cm = _make_cm_result(coaching_brief="Use the GROW model: start with Goal.")
    state = _make_state(context_management_result=cm)
    prompt = agent._build_prompt_input(state)
    assert "GROW model" in prompt


def test_build_prompt_input_includes_key_information():
    """_build_prompt_input should include key information from the cm_result."""
    agent = _make_agent_instance()
    cm = _make_cm_result(key_information="User leads a team of 5 engineers.")
    state = _make_state(context_management_result=cm)
    prompt = agent._build_prompt_input(state)
    assert "5 engineers" in prompt


def test_build_prompt_input_excludes_old_messages():
    """_build_prompt_input should truncate to recent_message_count messages."""
    from langchain_core.messages import HumanMessage

    agent = _make_agent_instance()
    agent._recent_message_count = 2
    messages = [HumanMessage(content=f"msg {i}") for i in range(10)]
    state = _make_state(messages=messages)
    prompt = agent._build_prompt_input(state)

    assert "msg 8" in prompt
    assert "msg 9" in prompt
    assert "msg 0" not in prompt


# ---------------------------------------------------------------------------
# format_response integration
# ---------------------------------------------------------------------------


def test_format_response_prefers_coaching_response():
    """format_response should use coaching_response when it is set."""
    from mentat.graph.workflow import format_response

    cm = _make_cm_result(coaching_brief="Use the GROW model.")
    state = _make_state(
        context_management_result=cm,
        coaching_response="What does success look like for you in 90 days?",
    )
    new_state = format_response(state)

    assert (
        new_state["final_response"] == "What does success look like for you in 90 days?"
    )
    assert "90 days" in new_state["messages"][-1].content


def test_format_response_falls_back_to_coaching_brief():
    """format_response falls back to coaching_brief when coaching_response is None."""
    from mentat.graph.workflow import format_response

    cm = _make_cm_result(coaching_brief="Start with empathy, then ask open questions.")
    state = _make_state(context_management_result=cm, coaching_response=None)
    new_state = format_response(state)

    assert new_state["final_response"] == "Start with empathy, then ask open questions."
    assert "empathy" in new_state["messages"][-1].content
