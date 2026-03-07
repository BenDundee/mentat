"""Tests for OrchestrationAgent."""

import os
from unittest.mock import MagicMock, patch

import pytest
from helpers import make_state
from pydantic import ValidationError

from mentat.core.models import Intent, OrchestrationResult


def test_orchestration_agent_run_mocked():
    """OrchestrationAgent.run() should populate orchestration_result."""
    from mentat.agents.orchestration import OrchestrationAgent, _IntentClassification

    fake_classification = _IntentClassification(
        intent=Intent.COACHING_SESSION,
        confidence=0.88,
        reasoning="Team leadership challenge.",
        suggested_agents=[],
    )

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = fake_classification

    with patch.object(OrchestrationAgent, "__init__", return_value=None):
        agent = OrchestrationAgent.__new__(OrchestrationAgent)
        agent._logger = MagicMock()
        agent.prompt_template = MagicMock()
        agent.llm = MagicMock()
        agent.llm.with_structured_output.return_value = MagicMock()

        # Make prompt_template | structured_llm return mock_chain
        agent.prompt_template.__or__ = MagicMock(return_value=mock_chain)

        state = make_state()
        result_state = agent.run(state)

    result = result_state["orchestration_result"]
    assert result is not None
    assert result.intent == Intent.COACHING_SESSION
    assert result.confidence == pytest.approx(0.88)
    assert result.reasoning == "Team leadership challenge."
    assert result.suggested_agents == ()


def test_orchestration_result_is_immutable():
    """OrchestrationResult must be frozen."""
    result = OrchestrationResult(
        intent=Intent.QUESTION,
        confidence=0.7,
        reasoning="User asked a question.",
    )
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        result.intent = Intent.OFF_TOPIC  # type: ignore[misc]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
def test_orchestration_agent_real_llm():
    """Integration test: call real LLM. Requires OPENROUTER_API_KEY."""
    from mentat.agents.orchestration import OrchestrationAgent

    agent = OrchestrationAgent()
    state = make_state(
        user_message="I've been struggling with delegating work to my team."
    )
    result_state = agent.run(state)

    result = result_state["orchestration_result"]
    assert result is not None
    assert result.intent in list(Intent)
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.reasoning) > 0
