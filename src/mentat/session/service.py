"""SessionService — load, save, and advance conversation sessions."""

import json
from datetime import datetime, timezone
from pathlib import Path

from mentat.core.logging import get_logger
from mentat.session.models import (
    ConversationSession,
    ConversationType,
    OnboardingPhase,
    SessionUpdateResult,
)

logger = get_logger(__name__)

_SESSION_DIR = Path("data/sessions")

# Ordered onboarding phases
_ONBOARDING_PHASE_ORDER = [
    OnboardingPhase.SET_EXPECTATIONS,
    OnboardingPhase.BACKGROUND_360,
    OnboardingPhase.GOAL_SETTING,
    OnboardingPhase.SELF_ASSESSMENT,
    OnboardingPhase.COACHING_PLAN,
    OnboardingPhase.COMPLETE,
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionService:
    """Stateless service for managing ConversationSession persistence.

    Instantiate once at module level in routes.py — no dependency injection
    required.
    """

    def load_or_create(self, session_id: str) -> ConversationSession:
        """Load an existing session or create a new onboarding session.

        Args:
            session_id: The unique session identifier from the chat request.

        Returns:
            An existing ConversationSession, or a freshly created one.
        """
        path = _SESSION_DIR / f"{session_id}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                session = ConversationSession(**data)
                logger.debug(
                    "Loaded session %s (type=%s phase=%s turn=%d)",
                    session_id,
                    session.conversation_type,
                    session.phase,
                    session.turn_count,
                )
                return session
            except Exception as exc:
                logger.warning(
                    "Failed to load session %s: %s — creating fresh session.",
                    session_id,
                    exc,
                )

        now = _utc_now()
        session = ConversationSession(
            session_id=session_id,
            conversation_type=ConversationType.ONBOARDING,
            phase=OnboardingPhase.SET_EXPECTATIONS.value,
            scratchpad="",
            collected_data={},
            turn_count=0,
            created_at=now,
            updated_at=now,
        )
        logger.info("Created new onboarding session %s.", session_id)
        return session

    def save(self, session: ConversationSession) -> None:
        """Persist a session to disk.

        Args:
            session: The session to save.
        """
        _SESSION_DIR.mkdir(parents=True, exist_ok=True)
        path = _SESSION_DIR / f"{session.session_id}.json"
        path.write_text(
            json.dumps(session.model_dump(), indent=2, default=str)
        )
        logger.debug("Saved session %s (phase=%s).", session.session_id, session.phase)

    def advance_phase(
        self, session: ConversationSession, update_result: SessionUpdateResult
    ) -> ConversationSession:
        """Apply a SessionUpdateResult to produce an updated session.

        Merges extracted_data, replaces the scratchpad, increments turn_count,
        and advances the phase when phase_complete is True.

        Args:
            session: Current immutable session.
            update_result: Output from SessionUpdateAgent.

        Returns:
            New ConversationSession with updated fields.
        """
        merged_data = {**session.collected_data, **update_result.extracted_data}
        next_phase = session.phase

        if update_result.phase_complete:
            if session.conversation_type == ConversationType.ONBOARDING:
                next_phase = self._next_onboarding_phase(session.phase)
            # Future: elif session.conversation_type == ConversationType.BIWEEKLY: ...

        updated = ConversationSession(
            session_id=session.session_id,
            conversation_type=session.conversation_type,
            phase=next_phase,
            scratchpad=update_result.updated_scratchpad,
            collected_data=merged_data,
            turn_count=session.turn_count + 1,
            created_at=session.created_at,
            updated_at=_utc_now(),
        )

        if next_phase != session.phase:
            logger.info(
                "Session %s advanced phase: %s → %s",
                session.session_id,
                session.phase,
                next_phase,
            )

        return updated

    @staticmethod
    def _next_onboarding_phase(current_phase: str) -> str:
        """Return the next onboarding phase after current_phase.

        Stays at COMPLETE if already there.
        """
        try:
            idx = _ONBOARDING_PHASE_ORDER.index(OnboardingPhase(current_phase))
        except (ValueError, KeyError):
            logger.warning("Unknown onboarding phase %r; staying put.", current_phase)
            return current_phase

        if idx < len(_ONBOARDING_PHASE_ORDER) - 1:
            return _ONBOARDING_PHASE_ORDER[idx + 1].value
        return current_phase  # already at COMPLETE
