"""Session Update Agent — evaluates the completed turn and advances session state."""

import json
from typing import Any, cast

from pydantic import BaseModel

from mentat.agents.base import BaseAgent
from mentat.graph.state import GraphState
from mentat.session.models import ConversationSession, SessionUpdateResult


class _SessionUpdateOutput(BaseModel):
    """Internal schema for structured session update output."""

    phase_complete: bool
    updated_scratchpad: str
    extracted_data: dict[str, Any]
    reasoning: str


class SessionUpdateAgent(BaseAgent):
    """Evaluates the completed turn and updates session state.

    Runs after format_response. Reads the turn context and current session,
    then produces a SessionUpdateResult that the SessionService uses to
    advance the session.
    """

    AGENT_NAME = "session_update"

    def run(self, state: GraphState) -> GraphState:
        """Update session state after a completed coaching turn.

        Args:
            state: Current graph state after format_response.

        Returns:
            New GraphState with ``session_state`` updated.
        """
        session = state.get("session_state")
        if session is None:
            self._logger.warning("No session_state in state; skipping session update.")
            return GraphState(**{k: state.get(k) for k in state})  # type: ignore[misc]

        self._logger.info(
            "SessionUpdateAgent running (session=%s phase=%s turn=%d)",
            session.session_id,
            session.phase,
            session.turn_count,
        )

        context = self._build_context(state, session)
        output = self._call_llm(context)

        update_result = SessionUpdateResult(
            phase_complete=output.phase_complete,
            updated_scratchpad=output.updated_scratchpad,
            extracted_data=output.extracted_data,
            reasoning=output.reasoning,
        )

        from mentat.session.service import SessionService

        service = SessionService()
        updated_session = service.advance_phase(session, update_result)

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
            quality_rating=state.get("quality_rating"),
            quality_feedback=state.get("quality_feedback"),
            coaching_attempts=state.get("coaching_attempts"),
            final_response=state.get("final_response"),
            session_state=updated_session,
        )

    def _build_context(self, state: GraphState, session: ConversationSession) -> str:
        """Assemble context for the LLM.

        Args:
            state: Current graph state.
            session: Current session before this turn.

        Returns:
            Formatted string for the LLM.
        """
        user_message = state["user_message"]
        coaching_response = (
            state.get("final_response") or state.get("coaching_response") or ""
        )
        collected_json = json.dumps(session.collected_data, indent=2, default=str)

        parts = [
            f"SESSION TYPE: {session.conversation_type.value}",
            f"CURRENT PHASE: {session.phase}",
            f"TURN NUMBER: {session.turn_count + 1}",
            f"COACH SCRATCHPAD:\n{session.scratchpad or '(empty)'}",
            f"COLLECTED DATA:\n{collected_json}",
            f"CLIENT MESSAGE:\n{user_message}",
            f"COACH RESPONSE:\n{coaching_response}",
        ]

        return "\n\n".join(parts)

    def _call_llm(self, context: str) -> _SessionUpdateOutput:
        """Invoke the LLM to produce a structured session update.

        Args:
            context: Assembled context string from _build_context.

        Returns:
            Structured _SessionUpdateOutput from the LLM.
        """
        structured_llm = self.llm.with_structured_output(
            _SessionUpdateOutput, strict=False
        )
        chain = self.prompt_template | structured_llm
        return cast(
            _SessionUpdateOutput,
            chain.invoke({"user_message": context}),
        )
