"""Tests for RAGAgent models and routing (Phase 9: Neo4j backend).

ChromaDB / VectorStoreService tests removed — replaced by test_neo4j_phase9.py.
"""

import pytest
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
        "context_management_result": None,
        "persona_context": None,
        "plan_context": None,
        "coaching_response": None,
        "quality_rating": None,
        "quality_feedback": None,
        "coaching_attempts": None,
        "final_response": None,
        "session_state": None,
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
    state = _make_state(orchestration_result=orch)
    assert _route_after_orchestration(state) == ["rag"]


def test_route_to_context_management_by_default():
    """_route_after_orchestration returns ['context_management'] when no RAG."""
    orch = OrchestrationResult(
        intent=Intent.CHECK_IN,
        confidence=0.95,
        reasoning="Casual check-in.",
        suggested_agents=(),
    )
    state = _make_state(orchestration_result=orch)
    assert _route_after_orchestration(state) == ["context_management"]


def test_route_to_context_management_when_no_orchestration():
    """_route_after_orchestration with no result returns ['context_management']."""
    state = _make_state()
    assert _route_after_orchestration(state) == ["context_management"]
