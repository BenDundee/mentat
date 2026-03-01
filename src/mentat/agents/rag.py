"""RAG Agent — retrieves relevant context from past conversations and documents."""

from langchain_core.prompts import ChatPromptTemplate

from mentat.agents.base import BaseAgent
from mentat.core.models import DocumentChunk, RAGAgentResult
from mentat.core.vector_store import VectorStoreService
from mentat.graph.state import GraphState


class RAGAgent(BaseAgent):
    """Retrieves relevant context via ChromaDB and summarizes it.

    Three-step pipeline:
    1. Generate a semantic search query from the user's message.
    2. Retrieve matching chunks from the vector store.
    3. Summarize the retrieved chunks in context of the user's message.
    """

    AGENT_NAME = "rag"

    def __init__(self, vector_store: VectorStoreService) -> None:
        super().__init__()
        self._vector_store = vector_store
        self._n_results: int = self.config.extra_config["n_results"]
        summary_prompt_text: str = self.config.extra_config["summary_system_prompt"]
        self._summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", summary_prompt_text),
                (
                    "human",
                    "User message: {user_message}\n\nRetrieved excerpts:\n{excerpts}",
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

        query = self._generate_query(user_message)
        self._logger.debug("Generated RAG query: %s", query)

        chunks = self._retrieve(query)
        self._logger.info("Retrieved %d chunks", len(chunks))

        summary = self._summarize(user_message, chunks)

        rag_result = RAGAgentResult(query=query, chunks=chunks, summary=summary)

        return GraphState(
            messages=state["messages"],
            user_message=state["user_message"],
            orchestration_result=state["orchestration_result"],
            search_results=state["search_results"],
            rag_results=rag_result,
            context_management_result=state["context_management_result"],
            persona_context=state["persona_context"],
            plan_context=state["plan_context"],
            coaching_response=state["coaching_response"],
            quality_rating=state.get("quality_rating"),
            quality_feedback=state.get("quality_feedback"),
            coaching_attempts=state.get("coaching_attempts"),
            final_response=state["final_response"],
        )

    def _generate_query(self, user_message: str) -> str:
        """Use the LLM to convert the user message into a search query."""
        chain = self.prompt_template | self.llm
        response = chain.invoke({"user_message": user_message})
        return str(response.content).strip()

    def _retrieve(self, query: str) -> tuple[DocumentChunk, ...]:
        """Query the vector store for relevant chunks."""
        return self._vector_store.search(query, n_results=self._n_results)

    def _summarize(self, user_message: str, chunks: tuple[DocumentChunk, ...]) -> str:
        """Summarize retrieved chunks in context of the user's message.

        Returns a fallback string if no chunks were retrieved.
        """
        if not chunks:
            return "No relevant context found in past conversations or documents."

        excerpts = "\n\n---\n\n".join(
            f"[{chunk.source}] {chunk.content}" for chunk in chunks
        )
        chain = self._summary_prompt | self.llm
        response = chain.invoke({"user_message": user_message, "excerpts": excerpts})
        return str(response.content).strip()
