"""Tests for SessionService."""

import json

import pytest

from mentat.session.models import (
    ConversationSession,
    ConversationType,
    OnboardingPhase,
    SessionUpdateResult,
)
from mentat.session.service import SessionService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(**overrides) -> ConversationSession:
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


def _make_update_result(**overrides) -> SessionUpdateResult:
    defaults = {
        "phase_complete": False,
        "updated_scratchpad": "Updated scratchpad.",
        "extracted_data": {},
        "reasoning": "No phase change.",
    }
    return SessionUpdateResult(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# load_or_create — new session
# ---------------------------------------------------------------------------


def test_load_or_create_creates_onboarding_for_new_session(tmp_path, monkeypatch):
    """load_or_create should create an onboarding session when no file exists."""
    monkeypatch.chdir(tmp_path)
    service = SessionService()
    session = service.load_or_create("new-session-id")

    assert session.session_id == "new-session-id"
    assert session.conversation_type == ConversationType.ONBOARDING
    assert session.phase == OnboardingPhase.SET_EXPECTATIONS.value
    assert session.turn_count == 0
    assert session.collected_data == {}


def test_load_or_create_sets_timestamps(tmp_path, monkeypatch):
    """load_or_create should populate created_at and updated_at."""
    monkeypatch.chdir(tmp_path)
    service = SessionService()
    session = service.load_or_create("ts-test")

    assert session.created_at
    assert session.updated_at


# ---------------------------------------------------------------------------
# load_or_create — existing session
# ---------------------------------------------------------------------------


def test_load_or_create_loads_existing_session(tmp_path, monkeypatch):
    """load_or_create should load and return an existing session file."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "sessions").mkdir(parents=True)

    existing = _make_session(
        session_id="existing-session",
        phase=OnboardingPhase.GOAL_SETTING.value,
        turn_count=5,
        scratchpad="Turn 5 notes.",
        collected_data={"role": "CTO"},
    )
    path = tmp_path / "data" / "sessions" / "existing-session.json"
    path.write_text(json.dumps(existing.model_dump(), default=str))

    service = SessionService()
    loaded = service.load_or_create("existing-session")

    assert loaded.session_id == "existing-session"
    assert loaded.phase == OnboardingPhase.GOAL_SETTING.value
    assert loaded.turn_count == 5
    assert loaded.collected_data == {"role": "CTO"}


def test_load_or_create_falls_back_on_corrupt_file(tmp_path, monkeypatch):
    """load_or_create should create a fresh session when the file is corrupt."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "sessions").mkdir(parents=True)
    path = tmp_path / "data" / "sessions" / "bad-session.json"
    path.write_text("{ not valid json }")

    service = SessionService()
    session = service.load_or_create("bad-session")

    assert session.session_id == "bad-session"
    assert session.phase == OnboardingPhase.SET_EXPECTATIONS.value


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


def test_save_writes_json_file(tmp_path, monkeypatch):
    """save should write a valid JSON file for the session."""
    monkeypatch.chdir(tmp_path)
    service = SessionService()
    session = _make_session(session_id="save-test")
    service.save(session)

    path = tmp_path / "data" / "sessions" / "save-test.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["session_id"] == "save-test"


def test_save_creates_directory_if_missing(tmp_path, monkeypatch):
    """save should create the data/sessions directory if it doesn't exist."""
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "data" / "sessions").exists()

    service = SessionService()
    session = _make_session(session_id="dir-test")
    service.save(session)

    assert (tmp_path / "data" / "sessions" / "dir-test.json").exists()


# ---------------------------------------------------------------------------
# advance_phase — no phase change
# ---------------------------------------------------------------------------


def test_advance_phase_no_change_when_not_complete(tmp_path, monkeypatch):
    """advance_phase should keep the same phase when phase_complete is False."""
    monkeypatch.chdir(tmp_path)
    service = SessionService()
    session = _make_session(phase=OnboardingPhase.BACKGROUND_360.value)
    result = _make_update_result(phase_complete=False)

    updated = service.advance_phase(session, result)
    assert updated.phase == OnboardingPhase.BACKGROUND_360.value


def test_advance_phase_increments_turn_count():
    """advance_phase should increment turn_count by 1."""
    service = SessionService()
    session = _make_session(turn_count=3)
    result = _make_update_result(phase_complete=False)

    updated = service.advance_phase(session, result)
    assert updated.turn_count == 4


def test_advance_phase_replaces_scratchpad():
    """advance_phase should replace the scratchpad with updated_scratchpad."""
    service = SessionService()
    session = _make_session(scratchpad="Old notes.")
    result = _make_update_result(updated_scratchpad="New notes.")

    updated = service.advance_phase(session, result)
    assert updated.scratchpad == "New notes."


def test_advance_phase_merges_extracted_data():
    """advance_phase should merge extracted_data into collected_data."""
    service = SessionService()
    session = _make_session(collected_data={"role": "VP Engineering"})
    result = _make_update_result(
        extracted_data={"goals_near_term": ["improve hiring"]},
        phase_complete=False,
    )

    updated = service.advance_phase(session, result)
    assert updated.collected_data["role"] == "VP Engineering"
    assert updated.collected_data["goals_near_term"] == ["improve hiring"]


# ---------------------------------------------------------------------------
# advance_phase — onboarding phase progression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current, expected_next",
    [
        (OnboardingPhase.SET_EXPECTATIONS, OnboardingPhase.BACKGROUND_360),
        (OnboardingPhase.BACKGROUND_360, OnboardingPhase.GOAL_SETTING),
        (OnboardingPhase.GOAL_SETTING, OnboardingPhase.SELF_ASSESSMENT),
        (OnboardingPhase.SELF_ASSESSMENT, OnboardingPhase.COACHING_PLAN),
        (OnboardingPhase.COACHING_PLAN, OnboardingPhase.COMPLETE),
    ],
)
def test_advance_phase_onboarding_progression(current, expected_next):
    """advance_phase should advance to the correct next onboarding phase."""
    service = SessionService()
    session = _make_session(phase=current.value)
    result = _make_update_result(phase_complete=True)

    updated = service.advance_phase(session, result)
    assert updated.phase == expected_next.value


def test_advance_phase_complete_stays_at_complete():
    """advance_phase should not advance past COMPLETE."""
    service = SessionService()
    session = _make_session(phase=OnboardingPhase.COMPLETE.value)
    result = _make_update_result(phase_complete=True)

    updated = service.advance_phase(session, result)
    assert updated.phase == OnboardingPhase.COMPLETE.value


def test_advance_phase_preserves_session_id():
    """advance_phase should preserve the session_id."""
    service = SessionService()
    session = _make_session(session_id="my-session")
    result = _make_update_result(phase_complete=False)

    updated = service.advance_phase(session, result)
    assert updated.session_id == "my-session"


def test_advance_phase_preserves_conversation_type():
    """advance_phase should preserve conversation_type."""
    service = SessionService()
    session = _make_session(conversation_type=ConversationType.ONBOARDING)
    result = _make_update_result(phase_complete=False)

    updated = service.advance_phase(session, result)
    assert updated.conversation_type == ConversationType.ONBOARDING


def test_advance_phase_updates_updated_at():
    """advance_phase should update the updated_at timestamp."""
    service = SessionService()
    original_ts = "2026-01-01T00:00:00+00:00"
    session = _make_session(updated_at=original_ts)
    result = _make_update_result(phase_complete=False)

    updated = service.advance_phase(session, result)
    assert updated.updated_at != original_ts


def test_advance_phase_unknown_phase_stays_put():
    """advance_phase should not crash on an unknown phase string."""
    service = SessionService()
    session = _make_session(phase="unknown_future_phase")
    result = _make_update_result(phase_complete=True)

    updated = service.advance_phase(session, result)
    assert updated.phase == "unknown_future_phase"
