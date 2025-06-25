# src/services/conversation_service.py
import datetime as dt
from logging import getLogger
from typing import Dict, List

from atomic_agents.lib.components.agent_memory import Message

from .rag_service import RAGService
from src.types import ConversationState, Persona

logger = getLogger(__name__)


class ConversationService:
    """Service for conversation-specific operations."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.state = self.get_new_conversation()
        
    def advance_conversation(self) -> str:
        bot_response = self.state.response

        if not self.state.user_message or not self.state.response:
            logger.warning("Conversation not advanced because user message or response is empty.")
            raise Exception("Conversation not advanced because user message or response is empty.")
        turn = len(self.state.history)
        self.state.history.append(Message(role= "user", content=self.state.user_message, turn_id=turn))
        self.state.history.append(Message(role="assistant", content=self.state.response, turn_id=(turn+1)))

        # Reset user message and response
        self.state.user_message = None
        self.state.response = None

        return bot_response

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