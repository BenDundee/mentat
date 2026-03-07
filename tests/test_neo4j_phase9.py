"""Tests for Phase 9: Neo4j memory architecture.

Covers:
- BlobStore
- EmbeddingService (mocked)
- Neo4jService methods (mocked driver)
- RAGAgent (mocked Neo4j + embedding)
- IngestAgent (mocked Neo4j + embedding)
- ConsolidationAgent (mocked Neo4j + embedding)
- New API endpoints (/consolidate, /memories, /documents/upload)
- Text splitting helper
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mentat.agents.consolidation import _parse_llm_response
from mentat.agents.ingest import _split_text
from mentat.core.blob_store import BlobStore
from mentat.core.neo4j_service import (
    ChunkNode,
    ChunkResult,
    InsightNode,
    MemoryNode,
    MemoryResult,
    SubgraphResult,
)
from mentat.graph.state import GraphState

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


def _make_embedding() -> list[float]:
    return [0.1] * 1024


# ---------------------------------------------------------------------------
# BlobStore
# ---------------------------------------------------------------------------


def test_blob_store_put_and_get(tmp_path):
    """BlobStore.put writes bytes; .get retrieves them."""
    store = BlobStore(base_dir=str(tmp_path / "blobs"))
    store.put("key1", b"hello world")
    assert store.get("key1") == b"hello world"


def test_blob_store_exists(tmp_path):
    """BlobStore.exists returns True after put, False before."""
    store = BlobStore(base_dir=str(tmp_path / "blobs"))
    assert not store.exists("missing")
    store.put("present", b"data")
    assert store.exists("present")


def test_blob_store_get_missing_raises(tmp_path):
    """BlobStore.get raises FileNotFoundError for unknown key."""
    store = BlobStore(base_dir=str(tmp_path / "blobs"))
    with pytest.raises(FileNotFoundError):
        store.get("no-such-key")


# ---------------------------------------------------------------------------
# Text splitting helper
# ---------------------------------------------------------------------------


def test_split_text_basic():
    """_split_text returns non-empty chunks for normal text."""
    words = ["word"] * 100
    text = " ".join(words)
    chunks = _split_text(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 20


def test_split_text_empty():
    """_split_text returns empty list for empty input."""
    assert _split_text("", chunk_size=10, chunk_overlap=2) == []


def test_split_text_shorter_than_chunk():
    """_split_text with text shorter than chunk_size returns one chunk."""
    chunks = _split_text("hello world", chunk_size=100, chunk_overlap=10)
    assert chunks == ["hello world"]


def test_split_text_overlap():
    """Consecutive chunks share overlapping words."""
    words = [str(i) for i in range(20)]
    text = " ".join(words)
    chunks = _split_text(text, chunk_size=10, chunk_overlap=3)
    # The last word of chunk[0] should appear somewhere in chunk[1]
    last_word_of_first = chunks[0].split()[-1]
    assert last_word_of_first in chunks[1].split()


# ---------------------------------------------------------------------------
# Neo4jService — embedding model fingerprint validation
# ---------------------------------------------------------------------------


def _make_neo4j_with_mock_driver(record_data: dict | None) -> object:
    """Build a Neo4jService instance whose driver returns controlled data."""
    from mentat.core.neo4j_service import Neo4jService

    svc = object.__new__(Neo4jService)

    # Build a mock result that returns the provided record (or None)
    mock_result = AsyncMock()
    if record_data is not None:
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: record_data[key]
        mock_result.single = AsyncMock(return_value=mock_record)
    else:
        mock_result.single = AsyncMock(return_value=None)

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    svc._driver = mock_driver
    return svc


@pytest.mark.anyio
async def test_validate_stamps_fresh_database():
    """validate_embedding_model stamps fingerprint when no EmbeddingConfig exists."""
    from mentat.core.neo4j_service import Neo4jService

    svc = object.__new__(Neo4jService)

    write_calls: list = []

    async def _run_side_effect(query, **kwargs):
        result = AsyncMock()
        if "MATCH (cfg:EmbeddingConfig)" in query:
            result.single = AsyncMock(return_value=None)
        else:
            write_calls.append(kwargs)
            result.single = AsyncMock(return_value=None)
        return result

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(side_effect=_run_side_effect)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    svc._driver = mock_driver

    # Should not raise
    await svc.validate_embedding_model("embed-english-v3.0", 1024)
    # The write call should have been made
    assert mock_session.run.call_count >= 2


@pytest.mark.anyio
async def test_validate_passes_on_matching_model():
    """validate_embedding_model succeeds when stored model matches configured model."""
    from mentat.core.neo4j_service import Neo4jService

    svc = object.__new__(Neo4jService)

    stored = {"model": "embed-english-v3.0", "dims": 1024}
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: stored[key]

    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    svc._driver = mock_driver

    # Should not raise
    await svc.validate_embedding_model("embed-english-v3.0", 1024)


@pytest.mark.anyio
async def test_validate_raises_on_model_mismatch():
    """validate_embedding_model raises EmbeddingModelMismatchError on model change."""
    from mentat.core.neo4j_service import EmbeddingModelMismatchError, Neo4jService

    svc = object.__new__(Neo4jService)

    stored = {"model": "embed-english-v3.0", "dims": 1024}
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: stored[key]

    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    svc._driver = mock_driver

    with pytest.raises(EmbeddingModelMismatchError, match="embed-english-v3.0"):
        await svc.validate_embedding_model("embed-multilingual-v3.0", 1024)


@pytest.mark.anyio
async def test_validate_raises_on_dims_mismatch():
    """validate_embedding_model raises EmbeddingModelMismatchError on dims change."""
    from mentat.core.neo4j_service import EmbeddingModelMismatchError, Neo4jService

    svc = object.__new__(Neo4jService)

    stored = {"model": "embed-english-v3.0", "dims": 1024}
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: stored[key]

    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=mock_record)

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    svc._driver = mock_driver

    with pytest.raises(EmbeddingModelMismatchError, match="dims=1024"):
        await svc.validate_embedding_model("embed-english-v3.0", 512)


# ---------------------------------------------------------------------------
# EmbeddingService — model property
# ---------------------------------------------------------------------------


def test_embedding_service_model_property():
    """EmbeddingService.model returns the configured model name."""
    with patch("mentat.core.embedding_service.CohereEmbeddings"):
        from mentat.core.embedding_service import EmbeddingService

        svc = EmbeddingService(model="embed-english-v3.0")
        assert svc.model == "embed-english-v3.0"


def test_embedding_service_dims_property():
    """EmbeddingService.dims returns 1024."""
    with patch("mentat.core.embedding_service.CohereEmbeddings"):
        from mentat.core.embedding_service import EmbeddingService

        svc = EmbeddingService()
        assert svc.dims == 1024


# ---------------------------------------------------------------------------
# JSON response parser (ConsolidationAgent helper)
# ---------------------------------------------------------------------------


def test_parse_llm_response_valid():
    raw = '{"insight": "User focuses on leadership.", "connections": []}'
    result = _parse_llm_response(raw)
    assert result is not None
    assert result["insight"] == "User focuses on leadership."
    assert result["connections"] == []


def test_parse_llm_response_with_fences():
    raw = '```json\n{"insight": "Growth mindset pattern.", "connections": []}\n```'
    result = _parse_llm_response(raw)
    assert result is not None
    assert "Growth mindset" in result["insight"]


def test_parse_llm_response_invalid_returns_none():
    assert _parse_llm_response("not json at all") is None


def test_parse_llm_response_empty_insight():
    raw = '{"insight": "", "connections": []}'
    result = _parse_llm_response(raw)
    assert result is not None
    assert result["insight"] == ""


# ---------------------------------------------------------------------------
# Neo4jService data transfer objects (frozen)
# ---------------------------------------------------------------------------


def test_chunk_node_frozen():
    chunk = ChunkNode(
        chunk_id="c1",
        text="Hello",
        embedding=[0.1] * 1024,
        chunk_type="conversation",
    )
    with pytest.raises((TypeError, AttributeError)):
        chunk.text = "mutated"  # type: ignore[misc]


def test_memory_node_frozen():
    mem = MemoryNode(
        memory_id="m1",
        text="User wants to improve leadership.",
        embedding=[0.2] * 1024,
    )
    with pytest.raises((TypeError, AttributeError)):
        mem.text = "mutated"  # type: ignore[misc]


def test_insight_node_frozen():
    insight = InsightNode(
        insight_id="i1",
        text="Pattern: leadership focus across sessions.",
        embedding=[0.3] * 1024,
        created_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises((TypeError, AttributeError)):
        insight.text = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RAGAgent
# ---------------------------------------------------------------------------


def _make_mock_neo4j() -> MagicMock:
    """Return an AsyncMock that mimics Neo4jService."""
    neo4j = MagicMock()
    neo4j.vector_search_chunks = AsyncMock(
        return_value=[
            ChunkResult(
                chunk_id="c1",
                text="User discussed leadership goals.",
                score=0.92,
                chunk_type="conversation",
                session_id="sess-1",
            )
        ]
    )
    neo4j.vector_search_memories = AsyncMock(
        return_value=[
            MemoryResult(
                memory_id="m1",
                text="User wants to become a better leader.",
                score=0.88,
                session_id="sess-1",
            )
        ]
    )
    neo4j.graph_expand = AsyncMock(
        return_value=SubgraphResult(chunks=[], memories=[], insights=[])
    )
    return neo4j


def _make_mock_embedding() -> MagicMock:
    emb = MagicMock()
    emb.embed.return_value = _make_embedding()
    return emb


def test_rag_agent_run_populates_rag_results():
    """RAGAgent.run() populates rag_results in the returned state."""
    mock_neo4j = _make_mock_neo4j()
    mock_emb = _make_mock_embedding()

    with patch("mentat.agents.rag.BaseAgent.__init__") as mock_base_init:
        mock_base_init.return_value = None

        from mentat.agents.rag import RAGAgent

        agent = object.__new__(RAGAgent)
        agent._logger = MagicMock()
        agent._neo4j = mock_neo4j
        agent._embedding = mock_emb
        agent._n_chunks = 5
        agent._n_memories = 5
        agent._max_nodes = 20

        # Mock query generation
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MagicMock(content="leadership goals query")
        agent.prompt_template = MagicMock()
        agent.llm = MagicMock()
        agent.prompt_template.__or__ = lambda self, other: mock_chain

        # Mock synthesis prompt
        mock_summary_chain = MagicMock()
        mock_summary_chain.invoke.return_value = MagicMock(
            content="User previously focused on leadership and team communication."
        )
        mock_summary_prompt = MagicMock()
        mock_summary_prompt.__or__ = lambda self, other: mock_summary_chain
        agent._summary_prompt = mock_summary_prompt

        state = _make_state()
        new_state = agent.run(state)

    assert new_state["rag_results"] is not None
    rag = new_state["rag_results"]
    assert "leadership" in rag.query or rag.query != ""
    assert isinstance(rag.chunks, tuple)
    assert "leadership" in rag.summary or rag.summary != ""


def test_rag_agent_run_empty_retrieval_fallback():
    """RAGAgent.run() with no hits returns a fallback summary."""
    mock_neo4j = MagicMock()
    mock_neo4j.vector_search_chunks = AsyncMock(return_value=[])
    mock_neo4j.vector_search_memories = AsyncMock(return_value=[])
    mock_neo4j.graph_expand = AsyncMock(
        return_value=SubgraphResult(chunks=[], memories=[], insights=[])
    )
    mock_emb = _make_mock_embedding()

    with patch("mentat.agents.rag.BaseAgent.__init__"):
        from mentat.agents.rag import RAGAgent

        agent = object.__new__(RAGAgent)
        agent._logger = MagicMock()
        agent._neo4j = mock_neo4j
        agent._embedding = mock_emb
        agent._n_chunks = 5
        agent._n_memories = 5
        agent._max_nodes = 20

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MagicMock(content="some query")
        agent.prompt_template = MagicMock()
        agent.llm = MagicMock()
        agent.prompt_template.__or__ = lambda self, other: mock_chain
        agent._summary_prompt = MagicMock()

        state = _make_state()
        new_state = agent.run(state)

    rag = new_state["rag_results"]
    assert rag is not None
    assert rag.chunks == ()
    assert "No relevant context" in rag.summary


# ---------------------------------------------------------------------------
# IngestAgent
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ingest_turn_writes_chunk_and_memory():
    """IngestAgent.ingest_turn writes a Chunk and synthesises a Memory."""
    mock_neo4j = MagicMock()
    mock_neo4j.add_session = AsyncMock()
    mock_neo4j.add_chunks = AsyncMock()
    mock_neo4j.add_memory = AsyncMock()
    mock_neo4j.link_memory_to_chunks = AsyncMock()
    mock_emb = _make_mock_embedding()

    with patch("mentat.agents.ingest.BaseAgent.__init__") as mock_init:
        mock_init.return_value = None
        from mentat.agents.ingest import IngestAgent

        agent = object.__new__(IngestAgent)
        agent._logger = MagicMock()
        agent._neo4j = mock_neo4j
        agent._embedding = mock_emb
        agent._min_turn_words = 5
        agent._chunk_size = 400
        agent._chunk_overlap = 50

        # Mock LLM for memory synthesis
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(
            return_value=MagicMock(content="User wants to improve leadership skills.")
        )
        agent._memory_prompt = MagicMock()
        agent.llm = MagicMock()
        agent._memory_prompt.__or__ = lambda self, other: mock_chain

        await agent.ingest_turn(
            session_id="sess-1",
            user_msg="How can I become a better leader?",
            assistant_msg="Focus on active listening and empathy.",
        )

    mock_neo4j.add_session.assert_called_once()
    mock_neo4j.add_chunks.assert_called_once()
    mock_neo4j.add_memory.assert_called_once()
    mock_neo4j.link_memory_to_chunks.assert_called_once()


@pytest.mark.anyio
async def test_ingest_turn_skips_memory_on_skip():
    """IngestAgent.ingest_turn does NOT write a Memory when LLM returns SKIP."""
    mock_neo4j = MagicMock()
    mock_neo4j.add_session = AsyncMock()
    mock_neo4j.add_chunks = AsyncMock()
    mock_neo4j.add_memory = AsyncMock()
    mock_emb = _make_mock_embedding()

    with patch("mentat.agents.ingest.BaseAgent.__init__"):
        from mentat.agents.ingest import IngestAgent

        agent = object.__new__(IngestAgent)
        agent._logger = MagicMock()
        agent._neo4j = mock_neo4j
        agent._embedding = mock_emb
        agent._min_turn_words = 5
        agent._chunk_size = 400
        agent._chunk_overlap = 50

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="SKIP"))
        agent._memory_prompt = MagicMock()
        agent.llm = MagicMock()
        agent._memory_prompt.__or__ = lambda self, other: mock_chain

        await agent.ingest_turn(
            session_id="sess-1",
            user_msg="OK thanks.",
            assistant_msg="You're welcome!",
        )

    mock_neo4j.add_memory.assert_not_called()


@pytest.mark.anyio
async def test_ingest_document_writes_chunks():
    """IngestAgent.ingest_document writes Document + Chunk nodes."""
    mock_neo4j = MagicMock()
    mock_neo4j.add_document = AsyncMock()
    mock_neo4j.add_chunks = AsyncMock()
    mock_neo4j.link_chunks = AsyncMock()
    mock_neo4j.add_memory = AsyncMock()
    mock_neo4j.link_memory_to_chunks = AsyncMock()

    mock_emb = MagicMock()
    # embed() returns a single vector; embed_batch returns one per chunk
    mock_emb.embed.return_value = _make_embedding()
    mock_emb.embed_batch = MagicMock(return_value=[[0.1] * 1024, [0.2] * 1024])

    with patch("mentat.agents.ingest.BaseAgent.__init__"):
        from mentat.agents.ingest import IngestAgent

        agent = object.__new__(IngestAgent)
        agent._logger = MagicMock()
        agent._neo4j = mock_neo4j
        agent._embedding = mock_emb
        agent._min_turn_words = 5
        agent._chunk_size = 5  # small so 2 chunks from short text
        agent._chunk_overlap = 1

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="SKIP"))
        agent.llm = MagicMock()

        # Patch _synthesize_document_memory to return SKIP
        async def _skip(*args, **kwargs):
            return "SKIP"

        agent._synthesize_document_memory = _skip

        text = "word " * 12  # 12 words → 2 chunks of 5 with overlap 1
        await agent.ingest_document(
            upload_id="doc-1",
            title="test.txt",
            text=text.strip(),
            blob_key="doc-1",
        )

    mock_neo4j.add_document.assert_called_once()
    mock_neo4j.add_chunks.assert_called_once()
    mock_neo4j.link_chunks.assert_called_once()


# ---------------------------------------------------------------------------
# ConsolidationAgent
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_consolidation_skips_when_too_few_memories():
    """ConsolidationAgent.run_once exits early when < min_memories."""
    mock_neo4j = MagicMock()
    mock_neo4j.get_unconsolidated_memories = AsyncMock(
        return_value=[
            MemoryNode(memory_id="m1", text="One memory.", embedding=_make_embedding())
        ]
    )
    mock_emb = _make_mock_embedding()

    with patch("mentat.agents.consolidation.BaseAgent.__init__"):
        from mentat.agents.consolidation import ConsolidationAgent

        agent = object.__new__(ConsolidationAgent)
        agent._logger = MagicMock()
        agent._neo4j = mock_neo4j
        agent._embedding = mock_emb
        agent._batch_size = 20
        agent._min_memories = 3
        agent.prompt_template = MagicMock()
        agent.llm = MagicMock()

        await agent.run_once()

    # Should not write any insights since only 1 memory
    mock_neo4j.get_unconsolidated_memories.assert_called_once()


@pytest.mark.anyio
async def test_consolidation_writes_insight():
    """ConsolidationAgent.run_once writes Insight and marks memories consolidated."""
    memories = [
        MemoryNode(
            memory_id=f"m{i}",
            text=f"Memory about leadership {i}.",
            embedding=_make_embedding(),
        )
        for i in range(4)
    ]
    mock_neo4j = MagicMock()
    mock_neo4j.get_unconsolidated_memories = AsyncMock(return_value=memories)
    mock_neo4j.add_insight = AsyncMock()
    mock_neo4j.strengthen_connection = AsyncMock()
    mock_neo4j.mark_consolidated = AsyncMock()
    mock_emb = MagicMock()
    mock_emb.embed.return_value = _make_embedding()

    llm_response = (
        '{"insight": "User consistently focuses on leadership.", '
        '"connections": [{"memory_id_a": "m0", "memory_id_b": "m1", "weight": 0.8}]}'
    )

    with patch("mentat.agents.consolidation.BaseAgent.__init__"):
        from mentat.agents.consolidation import ConsolidationAgent

        agent = object.__new__(ConsolidationAgent)
        agent._logger = MagicMock()
        agent._neo4j = mock_neo4j
        agent._embedding = mock_emb
        agent._batch_size = 20
        agent._min_memories = 3

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content=llm_response))
        agent.prompt_template = MagicMock()
        agent.llm = MagicMock()
        agent.prompt_template.__or__ = lambda self, other: mock_chain

        await agent.run_once()

    mock_neo4j.add_insight.assert_called_once()
    mock_neo4j.strengthen_connection.assert_called_once_with("m0", "m1", 0.8)
    mock_neo4j.mark_consolidated.assert_called_once()


# ---------------------------------------------------------------------------
# API: /memories and /consolidate endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_neo4j_service():
    svc = MagicMock()
    svc.get_recent_memories = AsyncMock(
        return_value=[
            MemoryNode(
                memory_id="m1",
                text="User wants to lead better.",
                embedding=_make_embedding(),
                session_id="sess-1",
                intent="coaching-session",
            )
        ]
    )
    svc.create_indexes = AsyncMock()
    svc.close = AsyncMock()
    return svc


@pytest.fixture
def mock_consolidation_agent():
    agent = MagicMock()
    agent.run_once = AsyncMock()
    return agent


@pytest.fixture
async def neo4j_client(mock_neo4j_service, mock_consolidation_agent):
    """AsyncClient with mocked Neo4j state."""
    from httpx import ASGITransport, AsyncClient

    from mentat.api.app import create_app

    app = create_app()
    app.state.neo4j_service = mock_neo4j_service
    app.state.consolidation_agent = mock_consolidation_agent
    app.state.ingest_agent = MagicMock()
    app.state.graph = MagicMock()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.anyio
async def test_get_memories_returns_list(neo4j_client, mock_neo4j_service):
    """GET /api/memories returns a list of memory dicts."""
    response = await neo4j_client.get("/api/memories")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["memory_id"] == "m1"
    assert "text" in data[0]


@pytest.mark.anyio
async def test_post_consolidate_returns_ok(neo4j_client, mock_consolidation_agent):
    """POST /api/consolidate triggers run_once and returns ok."""
    response = await neo4j_client.post("/api/consolidate")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    mock_consolidation_agent.run_once.assert_awaited_once()


@pytest.mark.anyio
async def test_upload_txt_document_with_ingest_agent():
    """POST /api/documents/upload calls ingest_agent.ingest_document."""
    from httpx import ASGITransport, AsyncClient

    from mentat.api.app import create_app

    mock_ingest = MagicMock()
    mock_ingest.ingest_document = AsyncMock()
    mock_ingest._chunk_size = 400

    app = create_app()
    app.state.ingest_agent = mock_ingest
    app.state.graph = MagicMock()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        content = b"This is a test document about leadership and executive coaching."
        response = await client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", content, "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.txt"
    mock_ingest.ingest_document.assert_awaited_once()


@pytest.mark.anyio
async def test_upload_unsupported_type_returns_422():
    """POST /api/documents/upload with bad extension returns 422."""
    from httpx import ASGITransport, AsyncClient

    from mentat.api.app import create_app

    app = create_app()
    app.state.ingest_agent = MagicMock()
    app.state.graph = MagicMock()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/documents/upload",
            files={"file": ("notes.xlsx", b"data", "application/octet-stream")},
        )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Workflow routing (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_route_after_orchestration_rag():
    """_route_after_orchestration returns ['rag'] when rag is suggested."""
    from mentat.core.models import Intent, OrchestrationResult
    from mentat.graph.workflow import _route_after_orchestration

    orch = OrchestrationResult(
        intent=Intent.QUESTION,
        confidence=0.9,
        reasoning="References past session.",
        suggested_agents=("rag",),
    )
    state = _make_state(orchestration_result=orch)
    assert _route_after_orchestration(state) == ["rag"]


def test_route_after_orchestration_default():
    """_route_after_orchestration falls back to context_management."""
    from mentat.core.models import Intent, OrchestrationResult
    from mentat.graph.workflow import _route_after_orchestration

    orch = OrchestrationResult(
        intent=Intent.CHECK_IN,
        confidence=0.95,
        reasoning="Check-in.",
        suggested_agents=(),
    )
    state = _make_state(orchestration_result=orch)
    assert _route_after_orchestration(state) == ["context_management"]
