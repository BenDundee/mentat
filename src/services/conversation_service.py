# src/services/conversation_service.py
import datetime as dt
from logging import getLogger
from typing import Optional, Tuple

from atomic_agents.lib.components.agent_memory import Message

from src.interfaces import ConversationState, Persona, TurnState, CoachingStage, CoachingSessionState, Intent
from src.utils import get_message, strip_message
from .rag_service import RAGService


logger = getLogger(__name__)


class ConversationService:
    """Service for conversation-specific operations."""

    def __init__(self, rag_service: RAGService, state: Optional[ConversationState] = None):
        self.rag_service = rag_service
        self.state = state or self.get_new_conversation()
        self.current_turn: Optional[TurnState] = None
        self.turn_counter: int = 0

    def initiate_turn(self, user_msg: str, conversation_id: Optional[str] = None):
        msg = get_message(role="user", message=user_msg, turn_id=f"{self.turn_counter}")
        if conversation_id and conversation_id != self.state.conversation_id:
            self.state = ConversationState(conversation_id=conversation_id)
        self.current_turn = TurnState(user_message=user_msg)

    def current_intent(self) -> Intent:
        if not self.current_turn.detected_intent:
            raise RuntimeError("Intent not yet determined")
        return self.current_turn.detected_intent

    def set_response(self, message: str, role: str):
        response = get_message(role=role, message=message, turn_id=f"{self.turn_counter}")
        if not self.current_turn:
            raise RuntimeError("Turn not initiated")
        self.current_turn.assistant_message = response

    def advance_conversation(self) -> Tuple[str, str]:
        if not self.current_turn or not self.current_turn.assistant_message:
            raise RuntimeError("Turn incomplete")
        self.current_turn.turn_end = dt.datetime.now().isoformat()
        self.state.history.append(self.current_turn)
        self.current_turn = None
        self.turn_counter += 1
        return strip_message(self.state.history[-1].response)

    @staticmethod
    def get_new_conversation():
        return ConversationState(
            conversation_id=None,
            user_id="guest",
            persona=Persona.get_empty_persona(),
            coaching_session=None,
            conversation_start=dt.datetime.now().isoformat(),
            conversation_end=None
        )