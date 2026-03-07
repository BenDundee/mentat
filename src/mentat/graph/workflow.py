"""LangGraph workflow definition for Mentat."""

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from mentat.agents.coaching import CoachingAgent
from mentat.agents.context_management import ContextManagementAgent
from mentat.agents.orchestration import OrchestrationAgent
from mentat.agents.output_testing import OutputTestingAgent
from mentat.agents.quality import QualityAgent
from mentat.agents.rag import RAGAgent
from mentat.agents.search import SearchAgent
from mentat.agents.session_update import SessionUpdateAgent
from mentat.core.embedding_service import EmbeddingService
from mentat.core.logging import get_logger
from mentat.core.neo4j_service import Neo4jService
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


_MAX_COACHING_ATTEMPTS = 3


def _route_after_quality(state: GraphState) -> str:
    """Route back to coaching for a rewrite, or proceed to format_response.

    Returns:
        ``"coaching"`` when the quality rating is ≤ 3 and the coaching agent
        has not yet reached the attempt limit. ``"format_response"`` otherwise.
    """
    rating = state.get("quality_rating")
    attempts = state.get("coaching_attempts") or 0
    if rating is not None and rating <= 3 and attempts < _MAX_COACHING_ATTEMPTS:
        logger.info(
            "Quality rating %d ≤ 3 (attempt %d/%d) — routing back to coaching.",
            rating,
            attempts,
            _MAX_COACHING_ATTEMPTS,
        )
        return "coaching"
    return "format_response"


def format_response(state: GraphState) -> GraphState:
    """Render the coaching response (or fallbacks) as the assistant message.

    Prefers ``coaching_response`` from CoachingAgent, then falls back to the
    ``coaching_brief`` from ContextManagementResult, and finally to the
    orchestration result summary for backwards compatibility.
    """
    coaching_response = state.get("coaching_response")
    if coaching_response is not None:
        response_text = coaching_response
    elif (cm_result := state.get("context_management_result")) is not None:
        logger.warning(
            "format_response: no coaching_response; falling back to coaching_brief."
        )
        response_text = cm_result.coaching_brief
    else:
        logger.warning(
            "format_response: no coaching_response or cm_result; "
            "falling back to orchestration result."
        )
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
        quality_rating=state.get("quality_rating"),
        quality_feedback=state.get("quality_feedback"),
        coaching_attempts=state.get("coaching_attempts"),
        final_response=response_text,
        session_state=state.get("session_state"),
    )


def build_graph(
    neo4j_service: Neo4jService,
    embedding_service: EmbeddingService,
    debug: bool = False,
) -> StateGraph:
    """Construct the LangGraph workflow.

    Args:
        neo4j_service:    Shared Neo4j service for RAG retrieval.
        embedding_service: Cohere embedding service for RAGAgent.
        debug: When True, replace the format_response node with OutputTestingAgent,
               which dumps the full pipeline state to the chat window.
    """
    orchestration_agent = OrchestrationAgent()
    search_agent = SearchAgent()
    rag_agent = RAGAgent(neo4j_service, embedding_service)
    context_management_agent = ContextManagementAgent()
    coaching_agent = CoachingAgent()
    quality_agent = QualityAgent()
    session_update_agent = SessionUpdateAgent()
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
    graph.add_node("coaching", coaching_agent.run)
    # pyrefly: ignore[no-matching-overload]
    graph.add_node("quality", quality_agent.run)
    # pyrefly: ignore[no-matching-overload]
    graph.add_node("format_response", final_node_fn)
    # pyrefly: ignore[no-matching-overload]
    graph.add_node("session_update", session_update_agent.run)

    graph.add_edge(START, "orchestration")
    graph.add_conditional_edges(  # pyrefly: ignore[no-matching-overload]
        "orchestration",
        _route_after_orchestration,
    )
    graph.add_edge("search", "context_management")
    graph.add_edge("rag", "context_management")
    graph.add_edge("context_management", "coaching")
    graph.add_edge("coaching", "quality")
    graph.add_conditional_edges(  # pyrefly: ignore[no-matching-overload]
        "quality",
        _route_after_quality,
    )
    graph.add_edge("format_response", "session_update")
    graph.add_edge("session_update", END)

    return graph


def compile_graph(
    neo4j_service: Neo4jService,
    embedding_service: EmbeddingService,
    debug: bool = False,
):  # type: ignore[return]
    """Build and compile the LangGraph workflow.

    Args:
        neo4j_service:    Shared Neo4j service for RAG retrieval.
        embedding_service: Cohere embedding service for RAGAgent.
        debug: Passed through to ``build_graph``; enables the debug state dump.
    """
    graph = build_graph(
        neo4j_service=neo4j_service,
        embedding_service=embedding_service,
        debug=debug,
    )
    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully (debug=%s).", debug)
    return compiled
