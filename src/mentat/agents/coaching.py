"""Coaching Agent — produces the actual coaching reply from the coaching brief."""

import logging
from typing import cast

from mentat.agents.base import BaseAgent
from mentat.graph.state import GraphState


class CoachingAgent(BaseAgent):
    """Constructs an executive coaching reply using the coaching brief.

    Reads the ContextManagementResult (coaching brief, tone guidance, key
    information, conversation summary) and the recent conversation history,
    then produces a plain-text coaching response via a single LLM call.
    """

    AGENT_NAME = "coaching"

    def __init__(self) -> None:
        super().__init__()
        self._recent_message_count: int = self.config.extra_config[
            "recent_message_count"
        ]

    def run(self, state: GraphState) -> GraphState:
        """Generate the coaching response from the coaching brief.

        Args:
            state: Current graph state after ContextManagementAgent has run,
                or after QualityAgent has requested a rewrite.

        Returns:
            New GraphState with ``coaching_response`` and ``coaching_attempts``
            populated.
        """
        user_message = state["user_message"]
        attempts = (state.get("coaching_attempts") or 0) + 1
        self._logger.info(
            "CoachingAgent running (attempt %d) for message: %.80s",
            attempts,
            user_message,
        )

        prompt_input = self._build_prompt_input(state)
        chain = self.prompt_template | self.llm
        result = chain.invoke({"user_message": prompt_input})
        coaching_response: str = cast(str, result.content)

        self._logger.debug(
            "Coaching response generated (attempt=%d, chars=%d)",
            attempts,
            len(coaching_response),
        )

        return GraphState(
            messages=state["messages"],
            user_message=state["user_message"],
            orchestration_result=state["orchestration_result"],
            search_results=state["search_results"],
            rag_results=state["rag_results"],
            context_management_result=state["context_management_result"],
            persona_context=state["persona_context"],
            plan_context=state["plan_context"],
            coaching_response=coaching_response,
            quality_rating=state.get("quality_rating"),
            quality_feedback=state.get("quality_feedback"),
            coaching_attempts=attempts,
            final_response=state["final_response"],
            session_state=state.get("session_state"),
        )

    def _build_prompt_input(self, state: GraphState) -> str:
        """Format coaching brief + user message + history for the LLM.

        Args:
            state: Current graph state.

        Returns:
            Formatted prompt string to pass to the LLM.
        """
        cm = state.get("context_management_result")
        messages = state.get("messages") or []

        # Truncate message history to the most recent N messages
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

        parts = [f"Client message: {state['user_message']}"]

        if cm is not None:
            parts.append(f"Coaching brief:\n{cm.coaching_brief}")
            parts.append(f"Tone guidance: {cm.tone_guidance}")
            if cm.key_information:
                parts.append(f"Key information:\n{cm.key_information}")
            if cm.conversation_summary:
                parts.append(f"Conversation summary:\n{cm.conversation_summary}")
        else:
            self._logger.warning(
                "CoachingAgent: no context_management_result in state; "
                "responding with user message only."
            )

        parts.append(f"Recent conversation:\n{history_text}")

        quality_feedback = state.get("quality_feedback")
        if quality_feedback:
            previous_response = state.get("coaching_response") or ""
            parts.append(
                f"REWRITE INSTRUCTIONS:\n"
                f"Your previous response was rated poorly. Previous response:\n"
                f"{previous_response}\n\n"
                f"Quality reviewer feedback:\n{quality_feedback}\n\n"
                f"Please rewrite the response addressing the feedback above."
            )

        return "\n\n".join(parts)


# Silence noisy sentence-transformers / tokenizer debug output
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
