"""Tests for RAGAgent models and routing (Phase 9: Neo4j backend).

ChromaDB / VectorStoreService tests removed — replaced by test_neo4j_phase9.py.
"""

import pytest
from helpers import make_state
from pydantic import ValidationError

from mentat.core.models import (
    DocumentChunk,
    Intent,
    OrchestrationResult,
    RAGAgentResult,
)
from mentat.graph.workflow import _route_after_orchestration


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
# Routing
# ---------------------------------------------------------------------------


def test_route_to_rag_when_suggested():
    """_route_after_orchestration should return ['rag'] when suggested."""
    orch = OrchestrationResult(
        intent=Intent.QUESTION,
        confidence=0.9,
        reasoning="User references past session.",
        suggested_agents=("rag",),
    )
    state = make_state(orchestration_result=orch)
    assert _route_after_orchestration(state) == ["rag"]


def test_route_to_context_management_by_default():
    """_route_after_orchestration returns ['context_management'] when no RAG."""
    orch = OrchestrationResult(
        intent=Intent.CHECK_IN,
        confidence=0.95,
        reasoning="Casual check-in.",
        suggested_agents=(),
    )
    state = make_state(orchestration_result=orch)
    assert _route_after_orchestration(state) == ["context_management"]


def test_route_to_context_management_when_no_orchestration():
    """_route_after_orchestration with no result returns ['context_management']."""
    state = make_state()
    assert _route_after_orchestration(state) == ["context_management"]
