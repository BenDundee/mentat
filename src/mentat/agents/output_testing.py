"""Output Testing Agent — dumps full GraphState to the chat window for debugging."""

from langchain_core.messages import AIMessage

from mentat.core.logging import get_logger
from mentat.graph.state import GraphState

logger = get_logger(__name__)


class OutputTestingAgent:
    """Formats the full GraphState as a readable markdown debug dump.

    Intended for local testing only — not for production use. Swap it into
    the graph via ``build_graph(debug=True)`` to inspect the complete pipeline
    state after each run.
    """

    def run(self, state: GraphState) -> GraphState:
        """Render all populated state fields as a markdown debug dump.

        Args:
            state: Current graph state after all upstream nodes have run.

        Returns:
            New GraphState with ``final_response`` set to the debug dump.
        """
        logger.info("OutputTestingAgent rendering debug state dump.")
        lines: list[str] = ["# Debug State Dump\n"]

        # User message
        lines.append(f"## User Message\n{state['user_message']}\n")

        # Orchestration result
        result = state.get("orchestration_result")
        if result is not None:
            confidence_pct = int(result.confidence * 100)
            lines.append("## Orchestration Result")
            lines.append(f"- **Intent:** {result.intent.value}")
            lines.append(f"- **Confidence:** {confidence_pct}%")
            lines.append(f"- **Reasoning:** {result.reasoning}")
            agents = ", ".join(result.suggested_agents) or "_(none)_"
            lines.append(f"- **Suggested agents:** {agents}\n")

        # Search results
        search = state.get("search_results")
        if search is not None:
            lines.append("## Search Results")
            lines.append(f"**Queries:** {', '.join(search.queries)}\n")
            for i, r in enumerate(search.results, 1):
                lines.append(f"**{i}. [{r.title}]({r.url})**")
                lines.append(f"{r.snippet}")
                lines.append(f"_Retrieved: {r.retrieved_at}_\n")
            lines.append(f"**Summary:**\n{search.summary}\n")

        # Remaining Phase 2+ string fields
        string_fields: list[tuple[str, str]] = [
            ("rag_results", "RAG Results"),
            ("persona_context", "Persona Context"),
            ("plan_context", "Plan Context"),
            ("coaching_response", "Coaching Response"),
            ("final_response", "Final Response"),
        ]
        for key, label in string_fields:
            value = state.get(key)  # type: ignore[literal-required]
            if value is not None:
                lines.append(f"## {label}\n{value}\n")

        # Quality rating
        quality = state.get("quality_rating")
        if quality is not None:
            lines.append(f"## Quality Rating\n{quality} / 5\n")

        # Message history summary
        messages = state.get("messages", [])
        lines.append(f"## Message History\n_{len(messages)} message(s) in context._\n")

        response_text = "\n".join(lines)
        return GraphState(
            messages=[AIMessage(content=response_text)],
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
