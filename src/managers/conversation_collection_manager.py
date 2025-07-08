from typing import Type, Dict, Any, List
import logging

from src.managers.base_collection_manager import BaseCollectionManager
from src.interfaces.chat import ConversationState


logger = logging.getLogger(__name__)


class ConversationCollectionManager(BaseCollectionManager):
    """Manager for handling conversation storage and retrieval in ChromaDB."""

    def __init__(self, config):
        super().__init__(config, "conversations")

    def get_model_class(self) -> Type[ConversationState]:
        return ConversationState

    def get_document_type(self) -> str:
        return "conversation"

    def _extract_metadata(self, conversation: ConversationState) -> Dict[str, Any]:
        """Extract searchable metadata from conversation state."""
        metadata = {
            "user_id": conversation.user_id,
        }

        # Add intent if available
        if conversation.detected_intent:
            metadata["intent"] = conversation.detected_intent.value

        # Add any simple context values
        for key, value in conversation.context.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[f"context_{key}"] = value

        return metadata

    def _add_searchable_content(self, conversation_id: str, conversation: ConversationState, metadata: Dict[str, Any]) -> None:
        """Add individual messages as searchable chunks."""

        docs, mds, ids = [], [], []
        base_metadata = metadata.copy()
        base_metadata.update({
            "conversation_id": conversation.conversation_id,
            "type": "conversation_chunk"
        })

        for i, message in enumerate(conversation.history):
            if not message.content:
                continue

            msg_metadata = base_metadata.copy()
            msg_metadata.update({
                "message_index": i,
                "role": message.role,
                "turn_id": message.turn_id or ""
            })

            # fun times... https://docs.pydantic.dev/latest/concepts/serialization/#serializing-with-duck-typing
            docs.append(message.content.model_dump_json(serialize_as_any=True))
            mds.append(msg_metadata)
            ids.append(f"conversation_message:{conversation_id}:{i}")

        if docs:
            self.chroma_service.add_documents(documents=docs, metadatas=mds, ids=ids)

    def get_user_conversations(self, user_id: str, limit: int = 50) -> List[ConversationState]:
        """Get all conversations for a specific user."""
        return self.find_by_filter({"user_id": user_id})

    def find_by_intent(self, intent: str, user_id: str = None) -> List[ConversationState]:
        """Find conversations by detected intent."""
        filter_dict = {"intent": intent}
        if user_id:
            #filter_dict["user_id"] = user_id
            pass  # TODO: Figure this out
        return self.find_by_filter(filter_dict)

    def get_conversation_context(self, query_text: str, user_id: str = None, n_results: int = 3) -> List[
        ConversationState]:
        """Get conversation context for enriching new conversations."""
        # First do semantic search
        semantic_results = self.semantic_search(query_text, n_results * 2)

        # Filter by user if specified
        if user_id:
            # semantic_results = [conv for conv in semantic_results if conv.user_id == user_id]
            pass # TODO: Figure this out
        return semantic_results[:n_results]
    

if __name__ == "__main__":
    from src.configurator import Configurator
    from src.interfaces.chat import ConversationState, SimpleMessageContentIOSchema
    from src.utils import get_message
    from src.interfaces import Intent

    conversation_manager = ConversationCollectionManager(Configurator())
    conversation = ConversationState(
        conversation_id="12345",
        user_message=SimpleMessageContentIOSchema(content="Hello, how are you?"),
        history=[
            get_message(role="user", message="Hello, how are you?"),
            get_message(role="assistant", message="I'm doing great, thanks for asking."),
        ],
        user_id="user1",
        detected_intent=Intent.SIMPLE,
        context={
            "last_question": "How are you?",
        }
    )

    # Store conversation
    conv_id = conversation_manager.add(conversation)
    print(f"Stored conversation with ID: {conv_id}")

    # Retrieve conversation
    retrieved = conversation_manager.get_by_id(conv_id)
    print(f"Retrieved conversation: {retrieved}")

    # Search conversations by intent
    similar = conversation_manager.find_by_intent("GENERAL_QUERY")
    print(f"Found {len(similar)} conversations with same intent")

    # Get conversation context
    context = conversation_manager.get_conversation_context("Python programming")
    print(f"Found {len(context)} relevant conversations for context")
