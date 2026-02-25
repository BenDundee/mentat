"""Shared pytest fixtures."""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from mentat.core.models import Intent, OrchestrationResult


@pytest.fixture
def mock_orchestration_result() -> OrchestrationResult:
    """A sample OrchestrationResult for testing."""
    return OrchestrationResult(
        intent=Intent.COACHING_SESSION,
        confidence=0.9,
        reasoning="User is describing a leadership challenge.",
        suggested_agents=(),
    )


@pytest.fixture
def mock_graph(mock_orchestration_result: OrchestrationResult):
    """A mock compiled graph that returns a predictable final state."""

    async def fake_ainvoke(state, **kwargs):
        return {
            **state,
            "orchestration_result": mock_orchestration_result,
            "final_response": (
                "**Intent detected:** coaching-session (90% confidence)\n\n"
                "**Reasoning:** User is describing a leadership challenge."
            ),
        }

    graph = MagicMock()
    graph.ainvoke = fake_ainvoke
    return graph


@pytest.fixture
async def async_client(mock_graph):
    """AsyncClient wired to the FastAPI app with a mocked graph."""
    from mentat.api.app import create_app

    app = create_app()
    app.state.graph = mock_graph

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
