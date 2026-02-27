"""LangGraph workflow definition for Mentat."""

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from mentat.agents.orchestration import OrchestrationAgent
from mentat.agents.rag import RAGAgent
from mentat.core.logging import get_logger
from mentat.core.vector_store import VectorStoreService
from mentat.graph.state import GraphState

logger = get_logger(__name__)


def _route_after_orchestration(state: GraphState) -> str:
    """Determine which node to visit after orchestration.

    Returns:
        ``"rag"`` if the orchestration result suggests the RAG agent,
        otherwise ``"format_response"``.
    """
    result = state.get("orchestration_result")
    if result is not None and "rag" in result.suggested_agents:
        return "rag"
    return "format_response"


def format_response(state: GraphState) -> GraphState:
    """Render the OrchestrationResult as a human-readable assistant message.

    If RAG results are present their summary is appended to the response.
    """
    result = state.get("orchestration_result")
    if result is None:
        response_text = "I'm sorry, I couldn't process your message."
    else:
        confidence_pct = int(result.confidence * 100)
        response_text = (
            f"**Intent detected:** {result.intent.value} "
            f"({confidence_pct}% confidence)\n\n"
            f"**Reasoning:** {result.reasoning}"
        )

    rag_results = state.get("rag_results")
    if rag_results is not None and rag_results.summary:
        response_text = f"{response_text}\n\n**Context:** {rag_results.summary}"

    assistant_message = AIMessage(content=response_text)
    return GraphState(
        messages=[assistant_message],
        user_message=state["user_message"],
        orchestration_result=state["orchestration_result"],
        search_results=state["search_results"],
        rag_results=state["rag_results"],
        persona_context=state["persona_context"],
        plan_context=state["plan_context"],
        coaching_response=state["coaching_response"],
        quality_rating=state["quality_rating"],
        final_response=response_text,
    )


def build_graph(vector_store: VectorStoreService) -> StateGraph:
    """Construct the LangGraph workflow."""
    orchestration_agent = OrchestrationAgent()
    rag_agent = RAGAgent(vector_store)

    graph = StateGraph(GraphState)  # pyrefly: ignore[bad-specialization]
    # pyrefly: ignore[no-matching-overload]
    graph.add_node("orchestration", orchestration_agent.run)
    # pyrefly: ignore[no-matching-overload]
    graph.add_node("rag", rag_agent.run)
    # pyrefly: ignore[no-matching-overload]
    graph.add_node("format_response", format_response)

    graph.add_edge(START, "orchestration")
    graph.add_conditional_edges(  # pyrefly: ignore[no-matching-overload]
        "orchestration",
        _route_after_orchestration,
        {"rag": "rag", "format_response": "format_response"},
    )
    graph.add_edge("rag", "format_response")
    graph.add_edge("format_response", END)

    return graph


def compile_graph(vector_store: VectorStoreService):  # type: ignore[return]
    """Build and compile the LangGraph workflow."""
    graph = build_graph(vector_store)
    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return compiled
