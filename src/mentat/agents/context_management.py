"""Context Management Agent — ranks and filters context for the Coaching Agent."""

import json
from typing import cast

from pydantic import BaseModel

from mentat.agents.base import BaseAgent
from mentat.core.models import ContextManagementResult
from mentat.graph.state import GraphState
from mentat.session.models import ConversationSession


class _ContextBrief(BaseModel):
    """Internal schema for structured context management output."""

    session_phase: str
    tone_guidance: str
    key_information: str
    conversation_summary: str
    coaching_brief: str


class ContextManagementAgent(BaseAgent):
    """Ranks and filters available context to produce a coaching brief.

    Single LLM call: synthesizes the orchestration result, search summary,
    RAG summary, and recent conversation history into a structured brief
    for the Coaching Agent.
    """

    AGENT_NAME = "context_management"

    def __init__(self) -> None:
        super().__init__()
        self._recent_message_count: int = self.config.extra_config[
            "recent_message_count"
        ]

    def run(self, state: GraphState) -> GraphState:
        """Produce a coaching brief from all available context.

        Args:
            state: Current graph state after Search / RAG agents have run.

        Returns:
            New GraphState with ``context_management_result`` populated.
        """
        user_message = state["user_message"]
        self._logger.info(
            "ContextManagementAgent running for message: %.80s", user_message
        )

        context = self._build_context(state)
        brief = self._call_llm(context)

        result = ContextManagementResult(
            coaching_brief=brief.coaching_brief,
            session_phase=brief.session_phase,
            tone_guidance=brief.tone_guidance,
            key_information=brief.key_information,
            conversation_summary=brief.conversation_summary,
        )

        self._logger.debug("Coaching brief generated (phase=%s)", result.session_phase)

        return self._return_state(state, context_management_result=result)

    def _format_session_context(self, session: ConversationSession) -> str:
        """Format session state as a context block for the LLM.

        Args:
            session: Current conversation session.

        Returns:
            Formatted string summarising session state and phase guidance.
        """
        collected_json = json.dumps(session.collected_data, indent=2, default=str)
        return (
            f"SESSION CONTEXT\n"
            f"Type: {session.conversation_type.value}\n"
            f"Phase: {session.phase}\n"
            f"Turn: {session.turn_count + 1}\n"
            f"Coach scratchpad:\n{session.scratchpad or '(none yet)'}\n"
            f"Collected data:\n{collected_json}"
        )

    def _build_context(self, state: GraphState) -> str:
        """Assemble context string from all available pipeline state.

        Args:
            state: Current graph state.

        Returns:
            Formatted context string to pass to the LLM.
        """
        orch = state.get("orchestration_result")
        search = state.get("search_results")
        rag = state.get("rag_results")
        messages = state.get("messages") or []

        history_text = self._format_message_history(
            messages, self._recent_message_count
        )

        session = state.get("session_state")
        parts = []
        if session is not None:
            parts.append(self._format_session_context(session))

        parts.append(f"User message: {state['user_message']}")

        if orch:
            parts.append(
                f"Intent: {orch.intent.value} (confidence: {orch.confidence:.0%})\n"
                f"Reasoning: {orch.reasoning}"
            )

        parts.append(
            f"Search Agent summary:\n{search.summary}"
            if search and search.summary
            else "Search Agent: no results"
        )

        parts.append(
            f"RAG Agent summary:\n{rag.summary}"
            if rag and rag.summary
            else "RAG Agent: no results"
        )

        parts.append(f"Recent conversation:\n{history_text}")

        return "\n\n".join(parts)

    def _call_llm(self, context: str) -> _ContextBrief:
        """Invoke the LLM to produce a structured coaching brief.

        Args:
            context: Assembled context string from _build_context.

        Returns:
            Structured _ContextBrief from the LLM.
        """
        structured_llm = self.llm.with_structured_output(_ContextBrief, strict=False)
        chain = self.prompt_template | structured_llm
        return cast(
            _ContextBrief,
            chain.invoke({"user_message": context}),
        )
