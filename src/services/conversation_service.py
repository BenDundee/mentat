# src/services/conversation_service.py
import datetime as dt
from logging import getLogger
from typing import Dict, List, Optional

from atomic_agents.lib.components.agent_memory import Message

from .rag_service import RAGService
from src.interfaces import ConversationState, Persona
from src.utils.helpers import get_message

logger = getLogger(__name__)


class ConversationService:
    """Service for conversation-specific operations."""
    
    def __init__(self, rag_service: RAGService, state: Optional[ConversationState] = None):
        self.rag_service = rag_service
        self.state = state or self.get_new_conversation()

    def initiate_turn(self, input: Message, history: Optional[List[Message]]=None, conversation_id: Optional[str]=None):
        if not self.state.conversation_id or self.state.conversation_id != conversation_id:
            logger.info("Starting new conversation...")
            self.state.conversation_id = conversation_id
            self.state.history = history
            self.state.user_message = input

        else:  # If conversation ID matches, update history
            # TODO: Figure out if this is needed and if so, what to do
            if set(history) ^ set(self.state.history):  # symmetric diff
                logger.error("Conversation history does not match, saving conversation and starting new one...")
            logger.info("Updating conversation history...")
            self.state.history = history
        
    def advance_conversation(self, response: str):
        self.state.set_response(response, turn_id=self.state.user_message.turn_id)
        if not self.state.user_message or not self.state.response:
            logger.warning("Conversation not advanced because user message or response is empty.")
            raise Exception("Conversation not advanced because user message or response is empty.")
        tid = f"{len(self.state.history)}"
        self.state.history.append(self.state.user_message)
        self.state.history.append(self.state.response)

        # Reset user message and response
        self.state.user_message = None
        self.state.response = None

    def find_by_intent(self, intent: str, user_id: str = None) -> List[Dict]:
        """Find conversations by intent."""
        query = f"conversations with intent {intent}"
        if user_id:
            query += f" for user {user_id}"
        return self.rag_service.query(query)
    
    def get_conversation_context(self, query_text: str, user_id: str = None) -> str:
        """Get conversation context for enriching responses."""
        context_query = f"relevant conversation context for: {query_text}"
        if user_id:
            context_query += f" for user {user_id}"
        return self.rag_service.query(context_query)

    @staticmethod
    def get_new_conversation():
        return ConversationState(
            conversation_id=None,
            conversation_start=dt.datetime.now().isoformat(),
            user_message=None,
            history=[],
            user_id="guest",
            detected_intent=None,
            persona=Persona.get_empty_persona(),
            response=None,
            errors=[],
            context={},
        )