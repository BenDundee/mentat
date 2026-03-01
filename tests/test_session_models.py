"""Tests for session data models."""

import pytest
from pydantic import ValidationError

from mentat.session.models import (
    ConversationSession,
    ConversationType,
    OnboardingPhase,
    SessionUpdateResult,
)

# ---------------------------------------------------------------------------
# ConversationType
# ---------------------------------------------------------------------------


def test_conversation_type_values():
    """ConversationType enum should expose the expected string values."""
    assert ConversationType.ONBOARDING == "onboarding"
    assert ConversationType.ADHOC == "adhoc"
    assert ConversationType.BIWEEKLY == "biweekly"


# ---------------------------------------------------------------------------
# OnboardingPhase
# ---------------------------------------------------------------------------


def test_onboarding_phase_values():
    """OnboardingPhase enum should expose the expected string values."""
    assert OnboardingPhase.SET_EXPECTATIONS == "set_expectations"
    assert OnboardingPhase.BACKGROUND_360 == "background_360"
    assert OnboardingPhase.GOAL_SETTING == "goal_setting"
    assert OnboardingPhase.SELF_ASSESSMENT == "self_assessment"
    assert OnboardingPhase.COACHING_PLAN == "coaching_plan"
    assert OnboardingPhase.COMPLETE == "complete"


# ---------------------------------------------------------------------------
# ConversationSession
# ---------------------------------------------------------------------------


def _make_session(**overrides) -> ConversationSession:
    defaults = {
        "session_id": "test-session-123",
        "conversation_type": ConversationType.ONBOARDING,
        "phase": OnboardingPhase.SET_EXPECTATIONS.value,
        "scratchpad": "",
        "collected_data": {},
        "turn_count": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    return ConversationSession(**{**defaults, **overrides})


def test_session_creation_defaults():
    """ConversationSession should be created with expected defaults."""
    session = _make_session()
    assert session.session_id == "test-session-123"
    assert session.conversation_type == ConversationType.ONBOARDING
    assert session.phase == "set_expectations"
    assert session.scratchpad == ""
    assert session.collected_data == {}
    assert session.turn_count == 0


def test_session_is_frozen():
    """ConversationSession should be immutable (frozen=True)."""
    session = _make_session()
    with pytest.raises((TypeError, ValidationError)):
        session.turn_count = 1  # type: ignore[misc]


def test_session_with_collected_data():
    """ConversationSession should store arbitrary collected_data."""
    session = _make_session(
        collected_data={"role": "VP Engineering", "goals_near_term": ["improve hiring"]}
    )
    assert session.collected_data["role"] == "VP Engineering"
    assert session.collected_data["goals_near_term"] == ["improve hiring"]


def test_session_serialization_roundtrip():
    """ConversationSession should survive a model_dump/model_validate roundtrip."""
    session = _make_session(
        phase=OnboardingPhase.GOAL_SETTING.value,
        turn_count=3,
        scratchpad="Turn 3. Goals emerging.",
        collected_data={"role": "CTO"},
    )
    data = session.model_dump()
    restored = ConversationSession(**data)
    assert restored.session_id == session.session_id
    assert restored.phase == session.phase
    assert restored.turn_count == session.turn_count
    assert restored.scratchpad == session.scratchpad
    assert restored.collected_data == session.collected_data


# ---------------------------------------------------------------------------
# SessionUpdateResult
# ---------------------------------------------------------------------------


def test_session_update_result_creation():
    """SessionUpdateResult should be created with expected fields."""
    result = SessionUpdateResult(
        phase_complete=True,
        updated_scratchpad="New scratchpad.",
        extracted_data={"role": "VP Engineering"},
        reasoning="Phase criteria met.",
    )
    assert result.phase_complete is True
    assert result.updated_scratchpad == "New scratchpad."
    assert result.extracted_data == {"role": "VP Engineering"}
    assert result.reasoning == "Phase criteria met."


def test_session_update_result_is_frozen():
    """SessionUpdateResult should be immutable."""
    result = SessionUpdateResult(
        phase_complete=False,
        updated_scratchpad="",
        extracted_data={},
        reasoning="",
    )
    with pytest.raises((TypeError, ValidationError)):
        result.phase_complete = True  # type: ignore[misc]


def test_session_update_result_empty_extracted_data():
    """SessionUpdateResult should allow empty extracted_data."""
    result = SessionUpdateResult(
        phase_complete=False,
        updated_scratchpad="No new data this turn.",
        extracted_data={},
        reasoning="Client gave vague answers.",
    )
    assert result.extracted_data == {}
