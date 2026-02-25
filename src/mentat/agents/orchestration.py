"""Orchestration Agent — classifies user intent."""

from typing import cast

from pydantic import BaseModel, Field

from mentat.agents.base import BaseAgent
from mentat.core.models import Intent, OrchestrationResult
from mentat.graph.state import GraphState


class _IntentClassification(BaseModel):
    """Internal schema used for structured LLM output."""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    suggested_agents: list[str] = Field(default_factory=list)


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

        structured_llm = self.llm.with_structured_output(_IntentClassification)
        chain = self.prompt_template | structured_llm
        raw = cast(
            _IntentClassification,
            chain.invoke({"user_message": state["user_message"]}),
        )

        result = OrchestrationResult(
            intent=raw.intent,
            confidence=raw.confidence,
            reasoning=raw.reasoning,
            suggested_agents=tuple(raw.suggested_agents),
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
            persona_context=state["persona_context"],
            plan_context=state["plan_context"],
            coaching_response=state["coaching_response"],
            quality_rating=state["quality_rating"],
            final_response=state["final_response"],
        )
