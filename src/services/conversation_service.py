# src/services/conversation_service.py
import datetime as dt
from logging import getLogger
from typing import Optional, Tuple

from atomic_agents.lib.components.agent_memory import AgentMemory

from src.interfaces import ConversationState, Persona, TurnState, CoachingStage, CoachingSessionState, Intent, \
    OrchestrationAgentOutputSchema
from src.utils import get_message, strip_message
from .rag_service import RAGService
from src.configurator import Configurator


logger = getLogger(__name__)


class ConversationService:
    """Service for conversation-specific operations."""

    def __init__(self, config: Configurator, rag_service: RAGService, state: Optional[ConversationState] = None):
        self.rag_service = rag_service
        self.config = config
        self.state = state or self.get_new_conversation()
        self.current_turn: Optional[TurnState] = None
        self.turn_counter: int = 0

        # Check for existing persona
        self.state.persona = self.config.persona

    def initiate_turn(self, user_msg: str, conversation_id: Optional[str] = None):
        msg = get_message(role="user", message=user_msg, turn_id=f"{self.turn_counter}")
        # if conversation_id and conversation_id != self.state.conversation_id:
        #     # TODO: Think about what to do in this case, probably want to write existing conversation to disk
        #     self.state = ConversationState(conversation_id=conversation_id)
        self.current_turn = TurnState(user_message=msg)

    def restart_turn(self):
        self.current_turn = None

    def orchestrate_turn(self, instructions: OrchestrationAgentOutputSchema):
        self.current_turn.intent = instructions.intent
        self.current_turn.intent_confidence = instructions.intent_confidence
        self.current_turn.intent_reasoning = instructions.intent_reasoning
        self.current_turn.action_directives = instructions.action_directives or [] # TODO: ist his ok?
        self.current_turn.response_outline = instructions.response_outline
        self.current_turn.conversation_summary = instructions.conversation_summary
        if instructions.errors: # TODO: make errors a map from agent name to error string to make debugging easier
            self.current_turn.errors.append(instructions.errors)

    def current_intent(self) -> Intent:
        if not self.current_turn.intent:
            raise RuntimeError("Intent not yet determined")
        return self.current_turn.intent

    def get_history(self, length: Optional[int] = None) -> AgentMemory:
        memory = AgentMemory()
        for h in self.state.history if not length else self.state.history[-length:]:
            memory.history.append(h.user_message)
            memory.history.append(h.response)
        return memory

    def set_response(self, message: str, role: str):
        response = get_message(role=role, message=message, turn_id=f"{self.turn_counter}")
        if not self.current_turn:
            raise RuntimeError("Turn not initiated")
        self.current_turn.assistant_message = response

    def advance_conversation(self) -> Tuple[str, str]:
        if not self.current_turn or not self.current_turn.response:
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