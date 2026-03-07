"""Neo4j AuraDB service — graph CRUD and vector search.

All Cypher queries live here so the rest of the codebase stays database-agnostic.
Vector indexes are created idempotently on startup.

Data model
----------
Nodes:
    Memory  — synthesised insight from a coaching turn
    Chunk   — raw text segment (conversation turn or document passage)
    Document — uploaded file
    Session  — conversation session
    Entity  — named entity extracted from memories
    Topic   — high-level theme
    Insight — cross-session pattern synthesised by ConsolidationAgent

Key relationships::

    (Document|Session) -[:CONTAINS]-> (Chunk)
    (Chunk)            -[:NEXT]->     (Chunk)
    (Session)          -[:PRODUCED]-> (Memory)
    (Memory)           -[:DERIVED_FROM]-> (Chunk)
    (Memory)           -[:MENTIONS]-> (Entity)
    (Memory)           -[:CONNECTED_TO {weight}]-> (Memory)
    (Insight)          -[:SYNTHESIZES]-> (Memory)
    (Entity)           -[:CO_OCCURS {count}]-> (Entity)
"""

from dataclasses import dataclass, field
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from mentat.core.logging import get_logger
from mentat.core.settings import settings

logger = get_logger(__name__)

# HNSW vector index configuration
_SIMILARITY = "cosine"


class EmbeddingModelMismatchError(RuntimeError):
    """Raised on startup when the configured embedding model differs from the
    model recorded in the Neo4j graph.

    Existing Chunk and Memory nodes were embedded with a different model, so
    vector search results would be meaningless.  You must either:
    - Revert to the previously-used embedding model, OR
    - Clear all Chunk / Memory / Insight nodes and re-ingest from scratch.
    """


# ---------------------------------------------------------------------------
# Data transfer objects (frozen dataclasses — no mutation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkNode:
    """Represents a Chunk node to be written to Neo4j."""

    chunk_id: str
    text: str
    embedding: list[float]
    chunk_type: str  # "conversation" | "document"
    session_id: str = ""
    document_id: str = ""
    chunk_index: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryNode:
    """Represents a Memory node to be written to Neo4j."""

    memory_id: str
    text: str
    embedding: list[float]
    session_id: str = ""
    intent: str = ""
    consolidated: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentNode:
    """Represents a Document node to be written to Neo4j."""

    document_id: str
    title: str
    blob_key: str
    uploaded_at: str


@dataclass(frozen=True)
class SessionNode:
    """Represents a Session node to be written to Neo4j."""

    session_id: str
    started_at: str


@dataclass(frozen=True)
class InsightNode:
    """Represents an Insight node to be written to Neo4j."""

    insight_id: str
    text: str
    embedding: list[float]
    created_at: str


@dataclass(frozen=True)
class ChunkResult:
    """A Chunk returned from a vector search."""

    chunk_id: str
    text: str
    score: float
    chunk_type: str
    session_id: str = ""
    document_id: str = ""


@dataclass(frozen=True)
class MemoryResult:
    """A Memory returned from a vector search."""

    memory_id: str
    text: str
    score: float
    session_id: str = ""
    intent: str = ""


@dataclass(frozen=True)
class SubgraphResult:
    """Expanded neighbourhood returned by :meth:`Neo4jService.graph_expand`."""

    chunks: list[ChunkResult]
    memories: list[MemoryResult]
    insights: list[str]  # plain text of related Insight nodes


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class Neo4jService:
    """Async Neo4j driver wrapper.

    Uses the official ``neo4j`` Python driver with async sessions.  All
    public methods are ``async`` so they integrate naturally with FastAPI
    and the LangGraph ``ainvoke`` path.

    Args:
        uri:      Bolt/AuraDB URI (e.g. ``neo4j+s://…``).
        username: Neo4j username.
        password: Neo4j password.
    """

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        resolved_uri = uri or settings.neo4j_uri
        resolved_user = username or settings.neo4j_username
        resolved_pass = password or settings.neo4j_password
        logger.info("Neo4jService connecting to %s", resolved_uri)
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            resolved_uri,
            auth=(resolved_user, resolved_pass),
        )

    async def close(self) -> None:
        """Close the underlying driver connection pool."""
        await self._driver.close()
        logger.info("Neo4jService closed.")

    # ------------------------------------------------------------------
    # Schema / index setup
    # ------------------------------------------------------------------

    async def create_indexes(self, dims: int) -> None:
        """Create HNSW vector indexes idempotently.

        Safe to call on every startup — Neo4j ignores the command if an
        index with the given name already exists.

        Args:
            dims: Embedding vector dimensions (read from configs/embedding.yml).
        """
        async with self._driver.session() as session:
            await session.run(
                """
                CREATE VECTOR INDEX `chunk-embeddings` IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $dims,
                    `vector.similarity_function`: $sim
                }}
                """,
                dims=dims,
                sim=_SIMILARITY,
            )
            await session.run(
                """
                CREATE VECTOR INDEX `memory-embeddings` IF NOT EXISTS
                FOR (m:Memory) ON (m.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $dims,
                    `vector.similarity_function`: $sim
                }}
                """,
                dims=dims,
                sim=_SIMILARITY,
            )
        logger.info("Neo4j vector indexes ready.")

    async def validate_embedding_model(self, model: str, dims: int) -> None:
        """Ensure the configured embedding model matches what is stored in Neo4j.

        On a fresh database (no ``EmbeddingConfig`` node) the fingerprint is
        written now and startup continues normally.

        On a database that already has data embedded with a *different* model,
        :class:`EmbeddingModelMismatchError` is raised immediately to prevent
        silently returning nonsensical ANN results.

        Args:
            model: Embedding model identifier (e.g. ``"embed-english-v3.0"``).
            dims:  Number of embedding dimensions (e.g. ``1024``).

        Raises:
            EmbeddingModelMismatchError: Stored model differs from *model*.
        """
        async with self._driver.session() as db:
            result = await db.run(
                """
                MATCH (cfg:EmbeddingConfig)
                RETURN cfg.model AS model, cfg.dims AS dims
                LIMIT 1
                """
            )
            record = await result.single()

        if record is None:
            # Fresh database — stamp the fingerprint now
            async with self._driver.session() as db:
                await db.run(
                    """
                    MERGE (cfg:EmbeddingConfig)
                    ON CREATE SET cfg.model = $model, cfg.dims = $dims
                    """,
                    model=model,
                    dims=dims,
                )
            logger.info(
                "Neo4j: stamped embedding fingerprint model=%s dims=%d", model, dims
            )
            return

        stored_model: str = record["model"]
        stored_dims: int = record["dims"]

        if stored_model != model or stored_dims != dims:
            raise EmbeddingModelMismatchError(
                f"Neo4j was indexed with model='{stored_model}' dims={stored_dims}, "
                f"but the current EmbeddingService uses model='{model}' dims={dims}. "
                f"Either revert to the original model or clear all "
                f"Chunk/Memory/Insight nodes and re-ingest from scratch."
            )

        logger.info(
            "Neo4j: embedding fingerprint validated (model=%s dims=%d).", model, dims
        )

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    async def add_session(self, session_node: SessionNode) -> None:
        """Merge a Session node (idempotent on session_id)."""
        async with self._driver.session() as db:
            await db.run(
                """
                MERGE (s:Session {session_id: $session_id})
                ON CREATE SET s.started_at = $started_at
                """,
                session_id=session_node.session_id,
                started_at=session_node.started_at,
            )

    async def add_document(self, doc: DocumentNode) -> None:
        """Merge a Document node (idempotent on document_id)."""
        async with self._driver.session() as db:
            await db.run(
                """
                MERGE (d:Document {document_id: $document_id})
                ON CREATE SET d.title = $title,
                              d.blob_key = $blob_key,
                              d.uploaded_at = $uploaded_at
                """,
                document_id=doc.document_id,
                title=doc.title,
                blob_key=doc.blob_key,
                uploaded_at=doc.uploaded_at,
            )

    async def add_chunks(self, chunks: list[ChunkNode]) -> None:
        """Create Chunk nodes and link them to their parent (Session or Document).

        Uses UNWIND for efficient batch writes.
        """
        if not chunks:
            return
        params: list[dict[str, Any]] = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "embedding": c.embedding,
                "chunk_type": c.chunk_type,
                "session_id": c.session_id,
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        async with self._driver.session() as db:
            await db.run(
                """
                UNWIND $chunks AS c
                MERGE (ch:Chunk {chunk_id: c.chunk_id})
                ON CREATE SET ch.text = c.text,
                              ch.embedding = c.embedding,
                              ch.chunk_type = c.chunk_type,
                              ch.session_id = c.session_id,
                              ch.document_id = c.document_id,
                              ch.chunk_index = c.chunk_index
                WITH ch, c
                CALL {
                    WITH ch, c
                    MATCH (s:Session {session_id: c.session_id})
                    WHERE c.session_id <> ''
                    MERGE (s)-[:CONTAINS]->(ch)
                }
                CALL {
                    WITH ch, c
                    MATCH (d:Document {document_id: c.document_id})
                    WHERE c.document_id <> ''
                    MERGE (d)-[:CONTAINS]->(ch)
                }
                """,
                chunks=params,
            )

    async def link_chunks(self, ids: list[str]) -> None:
        """Create sequential NEXT edges between Chunk nodes.

        Args:
            ids: Chunk IDs in document order; NEXT edge links id[i] → id[i+1].
        """
        if len(ids) < 2:
            return
        pairs = [{"a": ids[i], "b": ids[i + 1]} for i in range(len(ids) - 1)]
        async with self._driver.session() as db:
            await db.run(
                """
                UNWIND $pairs AS p
                MATCH (a:Chunk {chunk_id: p.a}), (b:Chunk {chunk_id: p.b})
                MERGE (a)-[:NEXT]->(b)
                """,
                pairs=pairs,
            )

    async def add_memory(self, memory: MemoryNode) -> None:
        """Create a Memory node and link it to its Session."""
        async with self._driver.session() as db:
            await db.run(
                """
                MERGE (m:Memory {memory_id: $memory_id})
                ON CREATE SET m.text = $text,
                              m.embedding = $embedding,
                              m.session_id = $session_id,
                              m.intent = $intent,
                              m.consolidated = $consolidated
                WITH m
                MATCH (s:Session {session_id: $session_id})
                WHERE $session_id <> ''
                MERGE (s)-[:PRODUCED]->(m)
                """,
                memory_id=memory.memory_id,
                text=memory.text,
                embedding=memory.embedding,
                session_id=memory.session_id,
                intent=memory.intent,
                consolidated=memory.consolidated,
            )

    async def link_memory_to_chunks(self, memory_id: str, chunk_ids: list[str]) -> None:
        """Create DERIVED_FROM edges from a Memory to its source Chunks."""
        if not chunk_ids:
            return
        async with self._driver.session() as db:
            await db.run(
                """
                MATCH (m:Memory {memory_id: $memory_id})
                UNWIND $chunk_ids AS cid
                MATCH (c:Chunk {chunk_id: cid})
                MERGE (m)-[:DERIVED_FROM]->(c)
                """,
                memory_id=memory_id,
                chunk_ids=chunk_ids,
            )

    async def upsert_co_occurs(self, entity_a: str, entity_b: str) -> None:
        """Increment the CO_OCCURS edge weight between two entities."""
        async with self._driver.session() as db:
            await db.run(
                """
                MERGE (a:Entity {name: $entity_a})
                MERGE (b:Entity {name: $entity_b})
                MERGE (a)-[r:CO_OCCURS]-(b)
                ON CREATE SET r.count = 1
                ON MATCH  SET r.count = r.count + 1
                """,
                entity_a=entity_a,
                entity_b=entity_b,
            )

    async def add_insight(self, insight: InsightNode, memory_ids: list[str]) -> None:
        """Write an Insight node and link it to the source Memory nodes."""
        async with self._driver.session() as db:
            await db.run(
                """
                MERGE (i:Insight {insight_id: $insight_id})
                ON CREATE SET i.text = $text,
                              i.embedding = $embedding,
                              i.created_at = $created_at
                WITH i
                UNWIND $memory_ids AS mid
                MATCH (m:Memory {memory_id: mid})
                MERGE (i)-[:SYNTHESIZES]->(m)
                """,
                insight_id=insight.insight_id,
                text=insight.text,
                embedding=insight.embedding,
                created_at=insight.created_at,
                memory_ids=memory_ids,
            )

    async def mark_consolidated(self, memory_ids: list[str]) -> None:
        """Set consolidated=true on a batch of Memory nodes."""
        if not memory_ids:
            return
        async with self._driver.session() as db:
            await db.run(
                """
                UNWIND $memory_ids AS mid
                MATCH (m:Memory {memory_id: mid})
                SET m.consolidated = true
                """,
                memory_ids=memory_ids,
            )

    async def strengthen_connection(
        self, memory_id_a: str, memory_id_b: str, weight: float
    ) -> None:
        """Upsert a weighted CONNECTED_TO edge between two Memory nodes."""
        async with self._driver.session() as db:
            await db.run(
                """
                MATCH (a:Memory {memory_id: $a}), (b:Memory {memory_id: $b})
                MERGE (a)-[r:CONNECTED_TO]-(b)
                ON CREATE SET r.weight = $weight
                ON MATCH  SET r.weight = r.weight + $weight
                """,
                a=memory_id_a,
                b=memory_id_b,
                weight=weight,
            )

    # ------------------------------------------------------------------
    # Read / search methods
    # ------------------------------------------------------------------

    async def vector_search_chunks(
        self, embedding: list[float], k: int = 5
    ) -> list[ChunkResult]:
        """ANN vector search over the chunk-embeddings index.

        Args:
            embedding: Query embedding vector (1024 dims).
            k:         Number of nearest neighbours to return.

        Returns:
            List of :class:`ChunkResult` ordered by descending similarity.
        """
        async with self._driver.session() as db:
            result = await db.run(
                """
                CALL db.index.vector.queryNodes('chunk-embeddings', $k, $embedding)
                YIELD node AS c, score
                RETURN c.chunk_id   AS chunk_id,
                       c.text       AS text,
                       c.chunk_type AS chunk_type,
                       c.session_id AS session_id,
                       c.document_id AS document_id,
                       score
                ORDER BY score DESC
                """,
                k=k,
                embedding=embedding,
            )
            records = await result.data()
        return [
            ChunkResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                chunk_type=r["chunk_type"],
                session_id=r.get("session_id") or "",
                document_id=r.get("document_id") or "",
            )
            for r in records
        ]

    async def vector_search_memories(
        self, embedding: list[float], k: int = 5
    ) -> list[MemoryResult]:
        """ANN vector search over the memory-embeddings index.

        Args:
            embedding: Query embedding vector (1024 dims).
            k:         Number of nearest neighbours to return.

        Returns:
            List of :class:`MemoryResult` ordered by descending similarity.
        """
        async with self._driver.session() as db:
            result = await db.run(
                """
                CALL db.index.vector.queryNodes('memory-embeddings', $k, $embedding)
                YIELD node AS m, score
                RETURN m.memory_id  AS memory_id,
                       m.text       AS text,
                       m.session_id AS session_id,
                       m.intent     AS intent,
                       score
                ORDER BY score DESC
                """,
                k=k,
                embedding=embedding,
            )
            records = await result.data()
        return [
            MemoryResult(
                memory_id=r["memory_id"],
                text=r["text"],
                score=r["score"],
                session_id=r.get("session_id") or "",
                intent=r.get("intent") or "",
            )
            for r in records
        ]

    async def graph_expand(
        self,
        chunk_ids: list[str],
        memory_ids: list[str],
    ) -> SubgraphResult:
        """Expand the neighbourhood of seed nodes via graph traversal.

        Follows NEXT (sequential), CONNECTED_TO (thematic), MENTIONS, and
        SYNTHESIZES edges to gather additional context beyond raw vector hits.

        Args:
            chunk_ids:  Seed Chunk node IDs from vector search.
            memory_ids: Seed Memory node IDs from vector search.

        Returns:
            :class:`SubgraphResult` with expanded chunks, memories, and insights.
        """
        async with self._driver.session() as db:
            # Expand chunks: follow NEXT edges (prev + next)
            chunk_result = await db.run(
                """
                UNWIND $chunk_ids AS cid
                MATCH (seed:Chunk {chunk_id: cid})
                OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(seed)
                OPTIONAL MATCH (seed)-[:NEXT]->(nxt:Chunk)
                WITH collect(DISTINCT seed) + collect(DISTINCT prev)
                     + collect(DISTINCT nxt) AS all_chunks
                UNWIND all_chunks AS c
                WITH DISTINCT c WHERE c IS NOT NULL
                RETURN c.chunk_id   AS chunk_id,
                       c.text       AS text,
                       c.chunk_type AS chunk_type,
                       c.session_id AS session_id,
                       c.document_id AS document_id,
                       1.0          AS score
                """,
                chunk_ids=chunk_ids,
            )
            chunk_records = await chunk_result.data()

            # Expand memories: follow CONNECTED_TO edges
            mem_result = await db.run(
                """
                UNWIND $memory_ids AS mid
                MATCH (seed:Memory {memory_id: mid})
                OPTIONAL MATCH (seed)-[:CONNECTED_TO]-(neighbor:Memory)
                WITH collect(DISTINCT seed) + collect(DISTINCT neighbor) AS all_mems
                UNWIND all_mems AS m
                WITH DISTINCT m WHERE m IS NOT NULL
                RETURN m.memory_id  AS memory_id,
                       m.text       AS text,
                       m.session_id AS session_id,
                       m.intent     AS intent,
                       1.0          AS score
                """,
                memory_ids=memory_ids,
            )
            mem_records = await mem_result.data()

            # Fetch related insights via SYNTHESIZES
            insight_result = await db.run(
                """
                UNWIND $memory_ids AS mid
                MATCH (i:Insight)-[:SYNTHESIZES]->(m:Memory {memory_id: mid})
                RETURN DISTINCT i.text AS text
                LIMIT 5
                """,
                memory_ids=memory_ids,
            )
            insight_records = await insight_result.data()

        return SubgraphResult(
            chunks=[
                ChunkResult(
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    score=r["score"],
                    chunk_type=r["chunk_type"],
                    session_id=r.get("session_id") or "",
                    document_id=r.get("document_id") or "",
                )
                for r in chunk_records
            ],
            memories=[
                MemoryResult(
                    memory_id=r["memory_id"],
                    text=r["text"],
                    score=r["score"],
                    session_id=r.get("session_id") or "",
                    intent=r.get("intent") or "",
                )
                for r in mem_records
            ],
            insights=[r["text"] for r in insight_records if r.get("text")],
        )

    async def get_unconsolidated_memories(self) -> list[MemoryNode]:
        """Return all Memory nodes where consolidated=false."""
        async with self._driver.session() as db:
            result = await db.run(
                """
                MATCH (m:Memory {consolidated: false})
                RETURN m.memory_id  AS memory_id,
                       m.text       AS text,
                       m.embedding  AS embedding,
                       m.session_id AS session_id,
                       m.intent     AS intent
                """,
            )
            records = await result.data()
        return [
            MemoryNode(
                memory_id=r["memory_id"],
                text=r["text"],
                embedding=r["embedding"] or [],
                session_id=r.get("session_id") or "",
                intent=r.get("intent") or "",
                consolidated=False,
            )
            for r in records
        ]

    async def get_recent_memories(self, limit: int = 20) -> list[MemoryNode]:
        """Return the most recently created Memory nodes."""
        async with self._driver.session() as db:
            result = await db.run(
                """
            MATCH (m:Memory)
            RETURN m.memory_id  AS memory_id,
                   m.text       AS text,
                   m.embedding  AS embedding,
                   m.session_id AS session_id,
                   m.intent     AS intent,
                   m.consolidated AS consolidated
            ORDER BY m.memory_id DESC
            LIMIT $limit
            """,
                limit=limit,
            )
            records = await result.data()
        return [
            MemoryNode(
                memory_id=r["memory_id"],
                text=r["text"],
                embedding=r["embedding"] or [],
                session_id=r.get("session_id") or "",
                intent=r.get("intent") or "",
                consolidated=bool(r.get("consolidated", False)),
            )
            for r in records
        ]
