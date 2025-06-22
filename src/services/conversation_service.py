# src/services/conversation_service.py
from typing import Dict, List
from .rag_service import RAGService

class ConversationService:
    """Service for conversation-specific operations."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
    
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