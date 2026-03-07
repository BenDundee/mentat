"""RAG Agent (QueryAgent) — retrieves relevant context from Neo4j.

Replaces the ChromaDB-backed implementation with a hybrid graph + vector
query pipeline:
  1. Embed the user message via EmbeddingService.
  2. ANN vector search: top-k Chunks + top-k Memories.
  3. Graph expansion: follow NEXT / CONNECTED_TO / SYNTHESIZES edges.
  4. Prune to at most ``max_nodes`` nodes by importance score.
  5. One LLM synthesis call to produce the RAGAgentResult.
"""

import asyncio

from langchain_core.prompts import ChatPromptTemplate

from mentat.agents.base import BaseAgent
from mentat.core.embedding_service import EmbeddingService
from mentat.core.models import DocumentChunk, RAGAgentResult
from mentat.core.neo4j_service import (
    ChunkResult,
    MemoryResult,
    Neo4jService,
    SubgraphResult,
)
from mentat.graph.state import GraphState


class RAGAgent(BaseAgent):
    """Retrieves relevant context from Neo4j and summarises it.

    Four-step pipeline:
    1. Embed the user message.
    2. Vector search: top-k Chunks + top-k Memories.
    3. Graph expand: neighbourhood traversal for additional context.
    4. LLM synthesis → RAGAgentResult.
    """

    AGENT_NAME = "rag"

    def __init__(
        self, neo4j_service: Neo4jService, embedding_service: EmbeddingService
    ) -> None:
        super().__init__()
        self._neo4j = neo4j_service
        self._embedding = embedding_service
        self._n_chunks: int = self.config.extra_config["n_chunks"]
        self._n_memories: int = self.config.extra_config["n_memories"]
        self._max_nodes: int = self.config.extra_config["max_nodes"]
        summary_prompt_text: str = self.config.extra_config["summary_system_prompt"]
        self._summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", summary_prompt_text),
                (
                    "human",
                    "User message: {user_message}\n\nRetrieved context:\n{context}",
                ),
            ]
        )

    def run(self, state: GraphState) -> GraphState:
        """Retrieve relevant context and populate rag_results.

        Args:
            state: Current graph state containing ``user_message``.

        Returns:
            New GraphState with ``rag_results`` populated.
        """
        user_message = state["user_message"]
        self._logger.info("RAGAgent running for message: %.80s", user_message)

        # Run async retrieval synchronously (LangGraph nodes are sync)
        loop = asyncio.new_event_loop()
        try:
            rag_result = loop.run_until_complete(
                self._retrieve_and_synthesize(user_message)
            )
        finally:
            loop.close()

        return self._return_state(state, rag_results=rag_result)

    async def _retrieve_and_synthesize(self, user_message: str) -> RAGAgentResult:
        """Async inner pipeline: embed → search → expand → synthesize."""
        # Step 1: generate search query via LLM
        query = self._generate_query(user_message)
        self._logger.debug("Generated RAG query: %s", query)

        # Step 2: embed query
        embedding = self._embedding.embed(query)

        # Step 3: ANN vector search (parallel)
        chunk_hits, memory_hits = await asyncio.gather(
            self._neo4j.vector_search_chunks(embedding, k=self._n_chunks),
            self._neo4j.vector_search_memories(embedding, k=self._n_memories),
        )
        self._logger.info(
            "Vector search: %d chunks, %d memories", len(chunk_hits), len(memory_hits)
        )

        # Step 4: graph expansion
        chunk_ids = [c.chunk_id for c in chunk_hits]
        memory_ids = [m.memory_id for m in memory_hits]
        subgraph: SubgraphResult = await self._neo4j.graph_expand(chunk_ids, memory_ids)

        # Step 5: merge + prune
        all_chunks = _merge_chunks(chunk_hits, subgraph.chunks, self._max_nodes)
        all_memories = _merge_memories(memory_hits, subgraph.memories, self._max_nodes)
        insights = subgraph.insights

        # Step 6: build DocumentChunk objects for result
        doc_chunks = tuple(
            DocumentChunk(
                content=c.text,
                source=c.chunk_type,
                document_id=c.document_id or c.session_id,
                metadata={"chunk_id": c.chunk_id},
            )
            for c in all_chunks
        )

        # Step 7: synthesize
        summary = self._synthesize(user_message, all_chunks, all_memories, insights)

        return RAGAgentResult(query=query, chunks=doc_chunks, summary=summary)

    def _generate_query(self, user_message: str) -> str:
        """Use the LLM to convert the user message into a search query."""
        chain = self.prompt_template | self.llm
        response = chain.invoke({"user_message": user_message})
        return str(response.content).strip()

    def _synthesize(
        self,
        user_message: str,
        chunks: list[ChunkResult],
        memories: list[MemoryResult],
        insights: list[str],
    ) -> str:
        """Summarize retrieved content in context of the user's message."""
        if not chunks and not memories and not insights:
            return "No relevant context found in past conversations or documents."

        parts: list[str] = []
        if memories:
            parts.append("## Memories\n" + "\n".join(f"- {m.text}" for m in memories))
        if chunks:
            parts.append(
                "## Excerpts\n"
                + "\n\n".join(f"[{c.chunk_type}] {c.text}" for c in chunks)
            )
        if insights:
            parts.append("## Patterns\n" + "\n".join(f"- {i}" for i in insights))

        context = "\n\n".join(parts)
        chain = self._summary_prompt | self.llm
        response = chain.invoke({"user_message": user_message, "context": context})
        return str(response.content).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_chunks(
    hits: list[ChunkResult],
    expanded: list[ChunkResult],
    max_nodes: int,
) -> list[ChunkResult]:
    """Merge hit + expanded chunks, deduplicate, cap at max_nodes."""
    seen: set[str] = set()
    merged: list[ChunkResult] = []
    for c in hits + expanded:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            merged.append(c)
        if len(merged) >= max_nodes:
            break
    return merged


def _merge_memories(
    hits: list[MemoryResult],
    expanded: list[MemoryResult],
    max_nodes: int,
) -> list[MemoryResult]:
    """Merge hit + expanded memories, deduplicate, cap at max_nodes."""
    seen: set[str] = set()
    merged: list[MemoryResult] = []
    for m in hits + expanded:
        if m.memory_id not in seen:
            seen.add(m.memory_id)
            merged.append(m)
        if len(merged) >= max_nodes:
            break
    return merged
