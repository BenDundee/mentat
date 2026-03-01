"""Data models for conversation session state."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConversationType(str, Enum):
    """Supported conversation types."""

    ONBOARDING = "onboarding"
    ADHOC = "adhoc"
    BIWEEKLY = "biweekly"


class OnboardingPhase(str, Enum):
    """Phases within an onboarding conversation."""

    SET_EXPECTATIONS = "set_expectations"
    BACKGROUND_360 = "background_360"
    GOAL_SETTING = "goal_setting"
    SELF_ASSESSMENT = "self_assessment"
    COACHING_PLAN = "coaching_plan"
    COMPLETE = "complete"


class ConversationSession(BaseModel, frozen=True):
    """Persistent session state that survives between turns.

    Stored as JSON at data/sessions/{session_id}.json.
    """

    session_id: str
    conversation_type: ConversationType
    # str for forward-compatibility; use OnboardingPhase values for onboarding
    phase: str
    scratchpad: str
    collected_data: dict[str, Any] = Field(default_factory=dict)
    turn_count: int = 0
    created_at: str  # ISO-8601 UTC
    updated_at: str  # ISO-8601 UTC


class SessionUpdateResult(BaseModel, frozen=True):
    """Structured output from the SessionUpdateAgent."""

    phase_complete: bool
    updated_scratchpad: str  # Full replacement; keep under ~400 words
    extracted_data: dict[str, Any]  # New facts to merge into collected_data
    reasoning: str  # Internal trace; not shown to client
