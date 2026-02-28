"""LangGraph workflow definition for Mentat."""

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from mentat.agents.context_management import ContextManagementAgent
from mentat.agents.orchestration import OrchestrationAgent
from mentat.agents.output_testing import OutputTestingAgent
from mentat.agents.rag import RAGAgent
from mentat.agents.search import SearchAgent
from mentat.core.logging import get_logger
from mentat.core.vector_store import VectorStoreService
from mentat.graph.state import GraphState

logger = get_logger(__name__)


def _route_after_orchestration(state: GraphState) -> list[str]:
    """Determine which nodes to visit after orchestration (parallel fan-out).

    Returns:
        A list of node names to dispatch in parallel. Both ``"search"`` and
        ``"rag"`` are returned when both are suggested, enabling parallel
        execution. Falls back to ``["context_management"]`` when no agents
        are suggested or the result is None.
    """
    result = state.get("orchestration_result")
    if result is None:
        return ["context_management"]
    agents = result.suggested_agents
    targets = [a for a in ("search", "rag") if a in agents]
    return targets if targets else ["context_management"]


def format_response(state: GraphState) -> GraphState:
    """Render the coaching brief (or a fallback) as the assistant message.

    Uses the coaching_brief from ContextManagementResult when available.
    Falls back to the orchestration result summary for backwards compatibility.
    """
    cm_result = state.get("context_management_result")
    if cm_result is not None:
        response_text = cm_result.coaching_brief
    else:
        orch = state.get("orchestration_result")
        if orch is None:
            response_text = "I'm sorry, I couldn't process your message."
        else:
            confidence_pct = int(orch.confidence * 100)
            response_text = (
                f"**Intent detected:** {orch.intent.value} "
                f"({confidence_pct}% confidence)\n\n"
                f"**Reasoning:** {orch.reasoning}"
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
        context_management_result=state["context_management_result"],
        persona_context=state["persona_context"],
        plan_context=state["plan_context"],
        coaching_response=state["coaching_response"],
        quality_rating=state["quality_rating"],
        final_response=response_text,
    )


def build_graph(vector_store: VectorStoreService, debug: bool = False) -> StateGraph:
    """Construct the LangGraph workflow.

    Args:
        vector_store: Shared ChromaDB service for RAG retrieval.
        debug: When True, replace the format_response node with OutputTestingAgent,
               which dumps the full pipeline state to the chat window.
    """
    orchestration_agent = OrchestrationAgent()
    search_agent = SearchAgent()
    rag_agent = RAGAgent(vector_store)
    context_management_agent = ContextManagementAgent()
    final_node_fn = OutputTestingAgent().run if debug else format_response

    graph = StateGraph(GraphState)  # pyrefly: ignore[bad-specialization]
    # pyrefly: ignore[no-matching-overload]
    graph.add_node("orchestration", orchestration_agent.run)
    # pyrefly: ignore[no-matching-overload]
    graph.add_node("search", search_agent.run)
    # pyrefly: ignore[no-matching-overload]
    graph.add_node("rag", rag_agent.run)
    # pyrefly: ignore[no-matching-overload]
    graph.add_node("context_management", context_management_agent.run)
    # pyrefly: ignore[no-matching-overload]
    graph.add_node("format_response", final_node_fn)

    graph.add_edge(START, "orchestration")
    graph.add_conditional_edges(  # pyrefly: ignore[no-matching-overload]
        "orchestration",
        _route_after_orchestration,
    )
    graph.add_edge("search", "context_management")
    graph.add_edge("rag", "context_management")
    graph.add_edge("context_management", "format_response")
    graph.add_edge("format_response", END)

    return graph


def compile_graph(vector_store: VectorStoreService, debug: bool = False):  # type: ignore[return]
    """Build and compile the LangGraph workflow.

    Args:
        vector_store: Shared ChromaDB service for RAG retrieval.
        debug: Passed through to ``build_graph``; enables the debug state dump.
    """
    graph = build_graph(vector_store=vector_store, debug=debug)
    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully (debug=%s).", debug)
    return compiled
