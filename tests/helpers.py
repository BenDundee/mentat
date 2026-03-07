"""Shared test builder helpers.

Regular module (not conftest.py) so it can be imported directly in test files.
"""

from mentat.core.models import ContextManagementResult
from mentat.graph.state import GraphState
from mentat.session.models import ConversationSession, ConversationType, OnboardingPhase


def make_state(**overrides) -> GraphState:  # type: ignore[return]
    """Build a GraphState with all 14 fields defaulted to None/empty."""
    base: GraphState = {
        "messages": [],
        "user_message": "I need help with my team.",
        "orchestration_result": None,
        "search_results": None,
        "rag_results": None,
        "context_management_result": None,
        "persona_context": None,
        "plan_context": None,
        "coaching_response": None,
        "quality_rating": None,
        "quality_feedback": None,
        "coaching_attempts": None,
        "final_response": None,
        "session_state": None,
    }
    return GraphState(**{**base, **overrides})  # type: ignore[misc]


def make_cm_result(**overrides) -> ContextManagementResult:
    """Build a ContextManagementResult with sensible defaults."""
    defaults = {
        "coaching_brief": "Acknowledge difficulty, then explore delegation blockers.",
        "session_phase": "exploration",
        "tone_guidance": "Warm and Socratic — ask before advising.",
        "key_information": "User leads a team of 5 engineers.",
        "conversation_summary": "User is working on leadership skills.",
    }
    return ContextManagementResult(**{**defaults, **overrides})


def make_session(**overrides) -> ConversationSession:
    """Build a ConversationSession with sensible defaults."""
    defaults = {
        "session_id": "test-session",
        "conversation_type": ConversationType.ONBOARDING,
        "phase": OnboardingPhase.SET_EXPECTATIONS.value,
        "scratchpad": "",
        "collected_data": {},
        "turn_count": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    return ConversationSession(**{**defaults, **overrides})
