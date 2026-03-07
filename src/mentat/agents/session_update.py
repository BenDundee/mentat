"""Session Update Agent — evaluates the completed turn and advances session state."""

import json
from typing import cast

from pydantic import BaseModel

from mentat.agents.base import BaseAgent
from mentat.graph.state import GraphState
from mentat.session.models import ConversationSession, SessionUpdateResult
from mentat.session.service import SessionService


class _ExtractedData(BaseModel):
    """Typed extracted-data fields for onboarding.

    All fields are Optional so the LLM only populates what was learned this turn.
    Using explicit fields (not dict[str, Any]) to satisfy Anthropic's structured
    output requirement that additionalProperties must be false.
    """

    role: str | None = None
    career_trajectory: str | None = None
    current_fires: str | None = None
    goals_near_term: list[str] | None = None
    goals_long_term: list[str] | None = None
    strengths: list[str] | None = None
    patterns: list[str] | None = None
    relationships_generative: list[str] | None = None
    relationships_draining: list[str] | None = None
    avoidances: list[str] | None = None
    coaching_plan_agreed: bool | None = None


class _SessionUpdateOutput(BaseModel):
    """Internal schema for structured session update output."""

    phase_complete: bool
    updated_scratchpad: str
    extracted_data: _ExtractedData
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
            return self._return_state(state)

        self._logger.info(
            "SessionUpdateAgent running (session=%s phase=%s turn=%d)",
            session.session_id,
            session.phase,
            session.turn_count,
        )

        context = self._build_context(state, session)
        output = self._call_llm(context)

        # Drop None values so we only merge facts actually learned this turn
        extracted = {
            k: v for k, v in output.extracted_data.model_dump().items() if v is not None
        }

        update_result = SessionUpdateResult(
            phase_complete=output.phase_complete,
            updated_scratchpad=output.updated_scratchpad,
            extracted_data=extracted,
            reasoning=output.reasoning,
        )

        service = SessionService()
        updated_session = service.advance_phase(session, update_result)

        return self._return_state(state, session_state=updated_session)

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
