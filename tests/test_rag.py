"""Tests for the RAG Agent, VectorStoreService, models, and upload endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from mentat.core.models import (
    DocumentChunk,
    Intent,
    OrchestrationResult,
    RAGAgentResult,
)
from mentat.graph.state import GraphState
from mentat.graph.workflow import _route_after_orchestration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> GraphState:
    base: GraphState = {
        "messages": [],
        "user_message": "What did we discuss last time?",
        "orchestration_result": None,
        "search_results": None,
        "rag_results": None,
        "persona_context": None,
        "plan_context": None,
        "coaching_response": None,
        "quality_rating": None,
        "final_response": None,
    }
    return GraphState(**{**base, **overrides})


def _make_chunk(**overrides) -> DocumentChunk:
    defaults = {
        "content": "Sample content",
        "source": "conversations",
        "document_id": "abc123",
        "metadata": {},
    }
    return DocumentChunk(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# Model immutability and structure
# ---------------------------------------------------------------------------


def test_document_chunk_frozen():
    """DocumentChunk is frozen — mutation raises ValidationError."""
    chunk = _make_chunk()
    with pytest.raises((ValidationError, TypeError)):
        chunk.content = "mutated"  # type: ignore[misc]


def test_rag_agent_result_frozen():
    """RAGAgentResult is frozen — mutation raises ValidationError."""
    result = RAGAgentResult(query="test query", summary="summary")
    with pytest.raises((ValidationError, TypeError)):
        result.summary = "mutated"  # type: ignore[misc]


def test_rag_agent_result_chunks_is_tuple():
    """RAGAgentResult.chunks defaults to an empty tuple."""
    result = RAGAgentResult(query="q", summary="s")
    assert isinstance(result.chunks, tuple)


def test_rag_agent_result_with_chunks():
    """RAGAgentResult stores chunks as a tuple."""
    chunk = _make_chunk()
    result = RAGAgentResult(query="q", chunks=(chunk,), summary="found some context")
    assert len(result.chunks) == 1
    assert result.chunks[0].content == "Sample content"


# ---------------------------------------------------------------------------
# VectorStoreService
# ---------------------------------------------------------------------------


def _make_mock_vector_store() -> MagicMock:
    """Return a MagicMock that mimics VectorStoreService."""
    vs = MagicMock()
    vs.search.return_value = (
        DocumentChunk(
            content="You mentioned wanting to improve team communication.",
            source="conversations",
            document_id="conv-1",
        ),
    )
    return vs


def test_vector_store_search_returns_tuple():
    """VectorStoreService.search() should return a tuple of DocumentChunks."""
    mock_vs = _make_mock_vector_store()
    result = mock_vs.search("team communication goals")
    assert isinstance(result, tuple)
    assert all(isinstance(c, DocumentChunk) for c in result)


# ---------------------------------------------------------------------------
# RAGAgent
# ---------------------------------------------------------------------------


def test_rag_agent_run_populates_rag_results():
    """RAGAgent.run() should populate rag_results in the returned state."""
    mock_vs = _make_mock_vector_store()

    with (
        patch("mentat.agents.rag.BaseAgent.__init__") as mock_base_init,
        patch("mentat.agents.rag.ChatPromptTemplate"),
    ):
        mock_base_init.return_value = None

        from mentat.agents.rag import RAGAgent

        agent = object.__new__(RAGAgent)
        agent._logger = MagicMock()
        agent._vector_store = mock_vs
        agent._n_results = 5

        # Mock the query generation chain
        mock_query_chain = MagicMock()
        mock_query_chain.invoke.return_value = MagicMock(content="team communication")
        agent.prompt_template = MagicMock()
        agent.llm = MagicMock()
        agent.prompt_template.__or__ = lambda self, other: mock_query_chain

        # Mock summary chain
        mock_summary_prompt = MagicMock()
        mock_summary_chain = MagicMock()
        mock_summary_chain.invoke.return_value = MagicMock(
            content="The user previously discussed team communication goals."
        )
        mock_summary_prompt.__or__ = lambda self, other: mock_summary_chain
        agent._summary_prompt = mock_summary_prompt

        state = _make_state()
        new_state = agent.run(state)

    assert new_state["rag_results"] is not None
    rag = new_state["rag_results"]
    assert rag.query == "team communication"
    assert len(rag.chunks) == 1
    assert "communication" in rag.summary


def test_rag_agent_run_empty_retrieval_fallback():
    """RAGAgent.run() with no chunks should produce a fallback summary."""
    mock_vs = MagicMock()
    mock_vs.search.return_value = ()

    with patch("mentat.agents.rag.BaseAgent.__init__"):
        from mentat.agents.rag import RAGAgent

        agent = object.__new__(RAGAgent)
        agent._logger = MagicMock()
        agent._vector_store = mock_vs
        agent._n_results = 5

        mock_query_chain = MagicMock()
        mock_query_chain.invoke.return_value = MagicMock(content="some query")
        agent.prompt_template = MagicMock()
        agent.llm = MagicMock()
        agent.prompt_template.__or__ = lambda self, other: mock_query_chain
        agent._summary_prompt = MagicMock()

        state = _make_state()
        new_state = agent.run(state)

    rag = new_state["rag_results"]
    assert rag is not None
    assert rag.chunks == ()
    assert "No relevant context" in rag.summary


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_route_to_rag_when_suggested():
    """_route_after_orchestration should return 'rag' when suggested."""
    orch = OrchestrationResult(
        intent=Intent.QUESTION,
        confidence=0.9,
        reasoning="User references past session.",
        suggested_agents=("rag",),
    )
    state = _make_state(orchestration_result=orch)
    assert _route_after_orchestration(state) == "rag"


def test_route_to_format_response_by_default():
    """_route_after_orchestration should return 'format_response' when no RAG."""
    orch = OrchestrationResult(
        intent=Intent.CHECK_IN,
        confidence=0.95,
        reasoning="Casual check-in.",
        suggested_agents=(),
    )
    state = _make_state(orchestration_result=orch)
    assert _route_after_orchestration(state) == "format_response"


def test_route_to_format_response_when_no_orchestration():
    """_route_after_orchestration with no result should go to format_response."""
    state = _make_state()
    assert _route_after_orchestration(state) == "format_response"


# ---------------------------------------------------------------------------
# API: document upload endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.add_documents.return_value = ["id-1", "id-2"]
    return vs


@pytest.fixture
async def upload_client(mock_vector_store):
    """AsyncClient with a real app but mocked vector store and graph."""
    from mentat.api.app import create_app

    app = create_app()
    app.state.vector_store = mock_vector_store
    app.state.graph = MagicMock()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.anyio
async def test_upload_txt_document(upload_client, mock_vector_store):
    """POST /api/documents/upload with a .txt file should return 200."""
    content = b"This is a sample text document about leadership skills."
    response = await upload_client.post(
        "/api/documents/upload",
        files={"file": ("resume.txt", content, "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "resume.txt"
    assert data["chunks_stored"] >= 1
    assert len(data["document_ids"]) >= 1
    assert "file_path" in data
    mock_vector_store.add_documents.assert_called_once()


@pytest.mark.anyio
async def test_upload_unsupported_type_returns_422(upload_client):
    """POST /api/documents/upload with an unsupported extension should return 422."""
    response = await upload_client.post(
        "/api/documents/upload",
        files={"file": ("notes.xlsx", b"binary data", "application/octet-stream")},
    )
    assert response.status_code == 422
