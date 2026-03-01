"""Quality Agent — reviews coaching responses and triggers rewrites when needed."""

from typing import cast

from pydantic import BaseModel

from mentat.agents.base import BaseAgent
from mentat.graph.state import GraphState


class _QualityAssessment(BaseModel):
    """Internal schema for structured quality evaluation output."""

    rating: int  # 1–5 overall score
    feedback: str  # Actionable rewrite guidance; empty string if rating > 3


class QualityAgent(BaseAgent):
    """Reviews coaching responses across five dimensions and triggers rewrites.

    Rates the coaching response 1–5 on:
      - Sanity Check: no hallucinations
      - Conversation Flow: contextually appropriate
      - Suitability for User: personalised and fitting
      - Plan Adherence: advancing the coaching plan
      - Voice: coach challenges client; client does the work

    If the overall rating is ≤ 3, ``quality_feedback`` is populated with
    actionable rewrite instructions. Callers use ``_route_after_quality`` to
    loop back to CoachingAgent when feedback is present.
    """

    AGENT_NAME = "quality"

    def __init__(self) -> None:
        super().__init__()
        self._recent_message_count: int = self.config.extra_config[
            "recent_message_count"
        ]

    def run(self, state: GraphState) -> GraphState:
        """Evaluate the coaching response and populate quality fields.

        Args:
            state: Current graph state after CoachingAgent has run.

        Returns:
            New GraphState with ``quality_rating`` and ``quality_feedback``
            populated.
        """
        user_message = state["user_message"]
        self._logger.info("QualityAgent reviewing response for: %.80s", user_message)

        context = self._build_context(state)
        assessment = self._call_llm(context)

        self._logger.info(
            "Quality assessment: rating=%d feedback_len=%d",
            assessment.rating,
            len(assessment.feedback),
        )

        feedback = assessment.feedback if assessment.rating <= 3 else None

        return GraphState(
            messages=state["messages"],
            user_message=state["user_message"],
            orchestration_result=state["orchestration_result"],
            search_results=state["search_results"],
            rag_results=state["rag_results"],
            context_management_result=state["context_management_result"],
            persona_context=state["persona_context"],
            plan_context=state["plan_context"],
            coaching_response=state["coaching_response"],
            quality_rating=assessment.rating,
            quality_feedback=feedback,
            coaching_attempts=state.get("coaching_attempts"),
            final_response=state["final_response"],
        )

    def _build_context(self, state: GraphState) -> str:
        """Assemble the review context for the LLM.

        Args:
            state: Current graph state.

        Returns:
            Formatted string containing the coaching response and relevant context.
        """
        messages = state.get("messages") or []
        recent = messages[-self._recent_message_count :]
        history_lines = []
        for msg in recent:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", "")
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "(no history)"

        coaching_response = state.get("coaching_response") or "(no response generated)"
        cm = state.get("context_management_result")

        parts = [
            f"Client message: {state['user_message']}",
            f"Coaching response to evaluate:\n{coaching_response}",
        ]

        if cm is not None:
            parts.append(f"Coaching brief (intended guidance):\n{cm.coaching_brief}")
            parts.append(f"Tone guidance: {cm.tone_guidance}")

        parts.append(f"Recent conversation:\n{history_text}")

        return "\n\n".join(parts)

    def _call_llm(self, context: str) -> _QualityAssessment:
        """Invoke the LLM to produce a structured quality assessment.

        Args:
            context: Assembled context string from _build_context.

        Returns:
            Structured _QualityAssessment from the LLM.
        """
        structured_llm = self.llm.with_structured_output(
            _QualityAssessment, strict=False
        )
        chain = self.prompt_template | structured_llm
        return cast(
            _QualityAssessment,
            chain.invoke({"user_message": context}),
        )
