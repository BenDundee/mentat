"""Tests for the Coaching Agent."""

from unittest.mock import MagicMock

from helpers import make_cm_result, make_state

from mentat.core.models import (
    Intent,
    OrchestrationResult,
)

_USER_MESSAGE = "I'm struggling to delegate work to my team."

# ---------------------------------------------------------------------------
# CoachingAgent.run
# ---------------------------------------------------------------------------


def test_run_populates_coaching_response(make_agent):
    """run() should populate coaching_response in the returned state."""
    from mentat.agents.coaching import CoachingAgent

    agent = make_agent(
        CoachingAgent,
        _recent_message_count=10,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )

    fake_llm_response = MagicMock()
    fake_llm_response.content = "What makes delegation feel so difficult right now?"

    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_llm_response
    agent.prompt_template.__or__ = MagicMock(return_value=chain_mock)

    cm = make_cm_result()
    state = make_state(user_message=_USER_MESSAGE, context_management_result=cm)
    new_state = agent.run(state)

    assert (
        new_state["coaching_response"]
        == "What makes delegation feel so difficult right now?"
    )


def test_run_increments_coaching_attempts(make_agent):
    """run() should increment coaching_attempts on each invocation."""
    from mentat.agents.coaching import CoachingAgent

    agent = make_agent(
        CoachingAgent,
        _recent_message_count=10,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )

    fake_llm_response = MagicMock()
    fake_llm_response.content = "Coaching reply."
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_llm_response
    agent.prompt_template.__or__ = MagicMock(return_value=chain_mock)

    state = make_state(coaching_attempts=None)
    new_state = agent.run(state)
    assert new_state["coaching_attempts"] == 1

    # Simulate second attempt (e.g. after quality rewrite loop)
    state2 = make_state(coaching_attempts=1)
    new_state2 = agent.run(state2)
    assert new_state2["coaching_attempts"] == 2


def test_run_rewrite_includes_feedback_in_prompt(make_agent):
    """run() should include quality_feedback in the prompt when set."""
    from mentat.agents.coaching import CoachingAgent

    agent = make_agent(
        CoachingAgent,
        _recent_message_count=10,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )

    fake_llm_response = MagicMock()
    fake_llm_response.content = "Improved coaching reply."
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_llm_response
    agent.prompt_template.__or__ = MagicMock(return_value=chain_mock)

    state = make_state(
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


def test_run_preserves_other_state_fields(make_agent):
    """run() should pass through all other state fields unchanged."""
    from mentat.agents.coaching import CoachingAgent

    agent = make_agent(
        CoachingAgent,
        _recent_message_count=10,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )

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
    cm = make_cm_result()
    state = make_state(
        orchestration_result=orch,
        context_management_result=cm,
        final_response="previous",
    )
    new_state = agent.run(state)

    assert new_state["orchestration_result"] is orch
    assert new_state["user_message"] == state["user_message"]
    assert new_state["context_management_result"] is cm
    assert new_state["final_response"] == "previous"


def test_run_handles_no_cm_result(make_agent):
    """run() should log a warning and still set coaching_response with no cm_result."""
    from mentat.agents.coaching import CoachingAgent

    agent = make_agent(
        CoachingAgent,
        _recent_message_count=10,
        llm=MagicMock(),
        prompt_template=MagicMock(),
    )

    fake_llm_response = MagicMock()
    fake_llm_response.content = "Let me help you with that."
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_llm_response
    agent.prompt_template.__or__ = MagicMock(return_value=chain_mock)

    state = make_state(context_management_result=None)
    new_state = agent.run(state)

    assert new_state["coaching_response"] == "Let me help you with that."
    agent._logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# CoachingAgent._build_prompt_input
# ---------------------------------------------------------------------------


def test_build_prompt_input_includes_user_message(make_agent):
    """_build_prompt_input should include the user message."""
    from mentat.agents.coaching import CoachingAgent

    agent = make_agent(CoachingAgent, _recent_message_count=10)
    state = make_state(user_message=_USER_MESSAGE)
    prompt = agent._build_prompt_input(state)
    assert "struggling to delegate" in prompt


def test_build_prompt_input_includes_coaching_brief(make_agent):
    """_build_prompt_input should include the coaching brief when cm_result present."""
    from mentat.agents.coaching import CoachingAgent

    agent = make_agent(CoachingAgent, _recent_message_count=10)
    cm = make_cm_result(coaching_brief="Use the GROW model: start with Goal.")
    state = make_state(context_management_result=cm)
    prompt = agent._build_prompt_input(state)
    assert "GROW model" in prompt


def test_build_prompt_input_includes_key_information(make_agent):
    """_build_prompt_input should include key information from the cm_result."""
    from mentat.agents.coaching import CoachingAgent

    agent = make_agent(CoachingAgent, _recent_message_count=10)
    cm = make_cm_result(key_information="User leads a team of 5 engineers.")
    state = make_state(context_management_result=cm)
    prompt = agent._build_prompt_input(state)
    assert "5 engineers" in prompt


def test_build_prompt_input_excludes_old_messages(make_agent):
    """_build_prompt_input should truncate to recent_message_count messages."""
    from langchain_core.messages import HumanMessage

    from mentat.agents.coaching import CoachingAgent

    agent = make_agent(CoachingAgent, _recent_message_count=2)
    messages = [HumanMessage(content=f"msg {i}") for i in range(10)]
    state = make_state(messages=messages)
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

    cm = make_cm_result(coaching_brief="Use the GROW model.")
    state = make_state(
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

    cm = make_cm_result(coaching_brief="Start with empathy, then ask open questions.")
    state = make_state(context_management_result=cm, coaching_response=None)
    new_state = format_response(state)

    assert new_state["final_response"] == "Start with empathy, then ask open questions."
    assert "empathy" in new_state["messages"][-1].content
