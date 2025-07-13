# src/services/conversation_service.py
import datetime as dt
from logging import getLogger
from typing import List, Optional

from atomic_agents.lib.components.agent_memory import Message

from src.interfaces import ConversationState, Persona
from .rag_service import RAGService


logger = getLogger(__name__)


class ConversationService:
    """Service for conversation-specific operations."""
    
    def __init__(self, rag_service: RAGService, state: Optional[ConversationState] = None):
        self.rag_service = rag_service
        self.state = state or self.get_new_conversation()
        self.current_turn_id = 0

    def initiate_turn(self, user_input: Message, history: Optional[List[Message]]=None, conversation_id: Optional[str]=None):
        if not self.state.conversation_id or self.state.conversation_id != conversation_id:
            logger.info("Starting new conversation...")
            self.state.conversation_id = conversation_id
            self.state.history = history
            self.state.user_message = user_input

        else:  # If conversation ID matches, update history
            # TODO: Figure out if this is needed and if so, what to do
            if set(history) ^ set(self.state.history):  # symmetric diff
                logger.error("Conversation history does not match, saving conversation and starting new one...")
            logger.info("Updating conversation history...")
            self.state.history = history
        
    def advance_conversation(self) -> Message:
        if not self.state.user_message or not self.state.response:
            logger.warning("Conversation not advanced because user message or response is empty.")
            raise Exception("Conversation not advanced because user message or response is empty.")
        tid = f"{len(self.state.history)}"
        self.state.history.append(self.state.user_message)
        self.state.history.append(self.state.response)

        # Reset user message and response, better way to do this?
        resp = self.state.response
        self.state.user_message = None
        self.state.response = None
        return resp


    @staticmethod
    def get_new_conversation():
        return ConversationState(
            conversation_id=None,
            conversation_start=dt.datetime.now().isoformat(),
            conversation_end=None,
            user_message=None,
            history=[],
            user_id="guest",
            detected_intent=None,
            persona=Persona.get_empty_persona(),
            response=None,
            errors=[],
            current_turn_id="0",
            coaching_session=None
        )