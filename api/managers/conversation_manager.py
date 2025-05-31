import logging
from typing import List, Dict, Any
from datetime import datetime
import uuid

from api.interfaces import ConversationState
from api.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


class ConversationRepository:
    """Repository for storing and retrieving conversation data using a vector database."""

    def __init__(self, persist_directory="./vector_db"):
        """Initialize the conversation repository with a vector store service."""
        self.vector_service = VectorStoreService(
            persist_directory=persist_directory,
            collection_name="conversations"
        )

    def save_conversation(self, state: ConversationState) -> str:
        """Save a conversation exchange with appropriate chunking based on length."""
        # Generate a unique conversation ID if needed
        conversation_id = state.context.get("conversation_id", str(uuid.uuid4()))

        # Get the current timestamp
        timestamp = datetime.now().isoformat()

        # Prepare the document for the vector store
        document_content = f"User: {state.user_message}\nAssistant: {state.response.content if state.response else ''}"

        # Prepare metadata
        metadata = {
            "user_id": state.user_id,
            "conversation_id": conversation_id,
            "timestamp": timestamp,
            "document_type": "conversation",
            "message_type": "exchange",
            "detected_intent": state.detected_intent.value if state.detected_intent else "unknown"
        }

        # For longer coaching sessions, use hierarchical chunking
        # For short exchanges, use simple storage without chunking
        if len(document_content) > 2000:  # Use threshold to determine chunking strategy
            exchange_id = self.vector_service.add_text_hierarchical(
                text=document_content,
                metadata=metadata,
                id=f"{conversation_id}_{timestamp}"
            )
        else:
            exchange_id = self.vector_service.add_text(
                text=document_content,
                metadata=metadata,
                id=f"{conversation_id}_{timestamp}",
                chunk=False  # Disable chunking for short exchanges
            )

        # Update conversation metadata as before...

        return conversation_id

    def get_relevant_context(self, query: str, user_id: str, limit: int = 3) -> str:
        """Get relevant context from past conversations with hierarchical context."""
        results = self.find_similar_conversations(query, user_id, limit)

        if not results:
            return ""

        context = "Previous relevant conversations:\n\n"
        for i, result in enumerate(results):
            # Include both the chunk and its parent context if available
            context += f"Conversation {i + 1}:\n{result['content']}\n"

            # Add parent context if available
            if result.get('parent_context'):
                context += f"\nBroader context:\n{result['parent_context']}\n"

            context += "\n"

        return context

    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a user with chronological ordering."""
        # Query for this user's conversations
        metadata_docs = self.vector_service.get_by_metadata({
            "user_id": user_id,
            "document_type": "conversation_metadata"
        })

        conversations = []
        for doc in metadata_docs:
            conv_id = doc.metadata.get("id")
            if conv_id:
                # For each conversation, get the exchanges
                exchanges = self.vector_service.get_by_metadata({
                    "conversation_id": conv_id,
                    "document_type": "conversation"
                })

                if exchanges:
                    # Sort by timestamp
                    sorted_exchanges = sorted(exchanges, key=lambda x: x.metadata.get("timestamp", ""))
                    conversations.append({
                        "id": conv_id,
                        "created_at": doc.metadata.get("created_at"),
                        "updated_at": doc.metadata.get("updated_at"),
                        "exchanges": [
                            {
                                "content": ex.page_content,
                                "timestamp": ex.metadata.get("timestamp"),
                                "intent": ex.metadata.get("detected_intent")
                            } for ex in sorted_exchanges
                        ]
                    })

        # Sort conversations by updated_at
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return conversations[:limit]

    def find_similar_conversations(self, query: str, user_id: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Find conversations semantically similar to the query."""
        filter_dict = {
            "document_type": "conversation"
        }

        if user_id:
            filter_dict["user_id"] = user_id

        # Perform semantic search
        docs = self.vector_service.search_by_text(
            query=query,
            filter_metadata=filter_dict,
            limit=limit
        )

        return [
            {
                "content": doc.page_content,
                "conversation_id": doc.metadata.get("conversation_id"),
                "timestamp": doc.metadata.get("timestamp"),
                "similarity_score": doc.metadata.get("score", 0)
            } for doc in docs
        ]

    def analyze_trends(self, user_id: str) -> Dict[str, Any]:
        """Analyze conversation trends for a user."""
        # Get all conversations for this user
        conversations = self.get_conversation_history(user_id, limit=100)

        if not conversations:
            return {"error": "No conversations found for this user"}

        # Collect all intents and their frequencies
        intents = {}
        for conv in conversations:
            for exchange in conv.get("exchanges", []):
                intent = exchange.get("intent", "unknown")
                intents[intent] = intents.get(intent, 0) + 1

        return {
            "conversation_count": len(conversations),
            "first_conversation": conversations[-1].get("created_at") if conversations else None,
            "most_recent_conversation": conversations[0].get("updated_at") if conversations else None,
            "common_intents": intents
        }