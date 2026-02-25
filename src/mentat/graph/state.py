"""LangGraph state definition for the Mentat workflow."""

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from mentat.core.models import OrchestrationResult


class GraphState(TypedDict):
    """Shared state passed between nodes in the Mentat LangGraph workflow.

    All Phase 2+ fields are declared here (defaulting to None) so that
    extending the graph never requires a schema migration.
    """

    messages: Annotated[list, add_messages]
    user_message: str

    # Populated by OrchestrationAgent
    orchestration_result: OrchestrationResult | None

    # Phase 2+ fields (unused in Phase 1)
    search_results: str | None
    rag_results: str | None
    persona_context: str | None
    plan_context: str | None
    coaching_response: str | None
    quality_rating: int | None
    final_response: str | None
