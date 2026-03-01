"""Orchestration Agent — classifies user intent."""

from typing import TypedDict, cast

from mentat.agents.base import BaseAgent
from mentat.core.models import Intent, OrchestrationResult
from mentat.graph.state import GraphState


class _IntentClassification(TypedDict):
    """Internal schema used for structured LLM output.

    TypedDict (not BaseModel) so LangChain stores a plain dict — not a
    Pydantic model — in AIMessage.parsed.  This avoids a Pydantic
    serialization warning that fires when a BaseModel subclass ends up in
    that extra field.
    """

    intent: Intent
    # Anthropic rejects min/max schema constraints; validated by OrchestrationResult
    confidence: float
    reasoning: str
    suggested_agents: list[str]


class OrchestrationAgent(BaseAgent):
    """Classifies the user's intent and populates orchestration_result."""

    AGENT_NAME = "orchestration"

    def run(self, state: GraphState) -> GraphState:
        """Classify user intent, return updated state.

        Args:
            state: Current graph state containing ``user_message``.

        Returns:
            New GraphState with ``orchestration_result`` populated.
        """
        self._logger.info(
            "Classifying intent for message: %.80s", state["user_message"]
        )

        structured_llm = self.llm.with_structured_output(
            _IntentClassification, strict=False
        )
        chain = self.prompt_template | structured_llm
        raw = cast(
            _IntentClassification,
            chain.invoke(
                {
                    "user_message": state["user_message"],
                    "current_datetime": self._now(),
                }
            ),
        )

        result = OrchestrationResult(
            intent=raw["intent"],
            confidence=raw["confidence"],
            reasoning=raw["reasoning"],
            suggested_agents=tuple(raw.get("suggested_agents", [])),
        )

        self._logger.info(
            "Intent: %s (confidence=%.2f)", result.intent, result.confidence
        )

        return GraphState(
            messages=state["messages"],
            user_message=state["user_message"],
            orchestration_result=result,
            search_results=state["search_results"],
            rag_results=state["rag_results"],
            context_management_result=state["context_management_result"],
            persona_context=state["persona_context"],
            plan_context=state["plan_context"],
            coaching_response=state["coaching_response"],
            quality_rating=state.get("quality_rating"),
            quality_feedback=state.get("quality_feedback"),
            coaching_attempts=state.get("coaching_attempts"),
            final_response=state["final_response"],
        )
