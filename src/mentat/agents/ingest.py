"""IngestAgent — stores conversation turns and uploaded documents in Neo4j.

Called from the API routes after each chat turn and document upload.
Not a LangGraph node — it is invoked directly as a service.
"""

import uuid
from datetime import datetime, timezone

from langchain_core.prompts import ChatPromptTemplate

from mentat.agents.base import BaseAgent
from mentat.core.embedding_service import EmbeddingService
from mentat.core.neo4j_service import (
    ChunkNode,
    DocumentNode,
    MemoryNode,
    Neo4jService,
    SessionNode,
)
from mentat.graph.state import GraphState


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestAgent(BaseAgent):
    """Ingests conversation turns and documents into the Neo4j graph.

    This agent is not wired into the LangGraph workflow.  It is called
    directly from route handlers as a post-processing step.

    Two public methods:
    - :meth:`ingest_turn` — store one user/assistant exchange.
    - :meth:`ingest_document` — chunk, embed, and store an uploaded file.
    """

    AGENT_NAME = "ingest"

    def __init__(
        self,
        neo4j_service: Neo4jService,
        embedding_service: EmbeddingService,
    ) -> None:
        super().__init__()
        self._neo4j = neo4j_service
        self._embedding = embedding_service
        self._min_turn_words: int = self.config.extra_config["min_turn_words"]
        self._chunk_size: int = self.config.extra_config["chunk_size"]
        self._chunk_overlap: int = self.config.extra_config["chunk_overlap"]

        self._memory_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.config.system_prompt),
                ("human", "User: {user_msg}\nAssistant: {assistant_msg}"),
            ]
        )

    def run(self, state: GraphState) -> GraphState:  # pragma: no cover
        """Not used — IngestAgent is invoked directly, not as a graph node."""
        return state

    async def ingest_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        intent: str = "",
    ) -> None:
        """Store a conversation turn as a Chunk node (and optionally a Memory).

        Args:
            session_id:    The active session identifier.
            user_msg:      User's message text.
            assistant_msg: Assistant's reply text.
            intent:        Intent string from OrchestrationResult (for metadata).
        """
        self._logger.info("IngestAgent.ingest_turn session_id=%s", session_id)

        # Ensure Session node exists
        await self._neo4j.add_session(
            SessionNode(session_id=session_id, started_at=_utc_now())
        )

        turn_text = f"User: {user_msg}\nAssistant: {assistant_msg}"
        chunk_id = str(uuid.uuid4())
        embedding = self._embedding.embed(turn_text)

        chunk = ChunkNode(
            chunk_id=chunk_id,
            text=turn_text,
            embedding=embedding,
            chunk_type="conversation",
            session_id=session_id,
            chunk_index=0,
        )
        await self._neo4j.add_chunks([chunk])

        # Synthesise a Memory if the turn is substantive
        word_count = len(turn_text.split())
        if word_count >= self._min_turn_words:
            memory_text = await self._synthesize_memory(user_msg, assistant_msg)
            if memory_text and memory_text.strip().upper() != "SKIP":
                memory_id = str(uuid.uuid4())
                mem_embedding = self._embedding.embed(memory_text)
                memory = MemoryNode(
                    memory_id=memory_id,
                    text=memory_text.strip(),
                    embedding=mem_embedding,
                    session_id=session_id,
                    intent=intent,
                    consolidated=False,
                )
                await self._neo4j.add_memory(memory)
                await self._neo4j.link_memory_to_chunks(memory_id, [chunk_id])
                self._logger.debug("Synthesised Memory memory_id=%s", memory_id)

    async def ingest_document(
        self,
        upload_id: str,
        title: str,
        text: str,
        blob_key: str,
    ) -> None:
        """Chunk, embed, and store an uploaded document.

        Args:
            upload_id: Unique ID for this upload (used as document_id).
            title:     Original filename or title.
            text:      Extracted plain text of the document.
            blob_key:  Key under which raw bytes are stored in BlobStore.
        """
        self._logger.info(
            "IngestAgent.ingest_document upload_id=%s title=%s", upload_id, title
        )

        # Ensure Document node exists
        await self._neo4j.add_document(
            DocumentNode(
                document_id=upload_id,
                title=title,
                blob_key=blob_key,
                uploaded_at=_utc_now(),
            )
        )

        # Chunk text
        raw_chunks = _split_text(text, self._chunk_size, self._chunk_overlap)
        self._logger.info(
            "Ingesting %d chunks for document %s", len(raw_chunks), upload_id
        )

        # Embed all chunks in one batch call
        embeddings = self._embedding.embed_batch(raw_chunks)

        chunk_ids: list[str] = []
        chunk_nodes: list[ChunkNode] = []
        for i, (chunk_text, emb) in enumerate(zip(raw_chunks, embeddings)):
            cid = str(uuid.uuid4())
            chunk_ids.append(cid)
            chunk_nodes.append(
                ChunkNode(
                    chunk_id=cid,
                    text=chunk_text,
                    embedding=emb,
                    chunk_type="document",
                    document_id=upload_id,
                    chunk_index=i,
                )
            )

        await self._neo4j.add_chunks(chunk_nodes)
        await self._neo4j.link_chunks(chunk_ids)

        # Synthesise per-section Memories from larger passages
        section_size = 3  # group every N chunks into one memory
        for start in range(0, len(raw_chunks), section_size):
            section_chunks = raw_chunks[start : start + section_size]
            section_text = " ".join(section_chunks)
            if len(section_text.split()) < self._min_turn_words:
                continue
            memory_text = await self._synthesize_document_memory(section_text, title)
            if memory_text and memory_text.strip().upper() != "SKIP":
                memory_id = str(uuid.uuid4())
                mem_emb = self._embedding.embed(memory_text)
                memory = MemoryNode(
                    memory_id=memory_id,
                    text=memory_text.strip(),
                    embedding=mem_emb,
                    session_id="",
                    intent="document-review",
                    consolidated=False,
                )
                await self._neo4j.add_memory(memory)
                section_chunk_ids = chunk_ids[start : start + section_size]
                await self._neo4j.link_memory_to_chunks(memory_id, section_chunk_ids)

    async def _synthesize_memory(self, user_msg: str, assistant_msg: str) -> str:
        """Ask the LLM to distill a single memory sentence from the turn."""
        chain = self._memory_prompt | self.llm
        response = await chain.ainvoke(
            {"user_msg": user_msg, "assistant_msg": assistant_msg}
        )
        return str(response.content).strip()

    async def _synthesize_document_memory(self, section_text: str, title: str) -> str:
        """Ask the LLM to distill a memory from a document section."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.config.system_prompt),
                ("human", "Document: {title}\n\nSection:\n{section}"),
            ]
        )
        chain = prompt | self.llm
        response = await chain.ainvoke({"title": title, "section": section_text})
        return str(response.content).strip()


# ---------------------------------------------------------------------------
# Text splitting helper
# ---------------------------------------------------------------------------


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Simple word-based text splitter.

    Args:
        text:         Input text.
        chunk_size:   Target number of words per chunk.
        chunk_overlap: Number of words to overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks
