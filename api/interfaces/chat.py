from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from api.interfaces import Intent
from langchain.schema import ChatMessage
from langchain_core.messages import BaseMessage


class ChatRequest(BaseModel):
    """Input schema for the chat endpoint."""
    message: str = Field(..., description="The message to send to the chatbot")
    history: Optional[List[ChatMessage]] = None
    user_id: Optional[str] = "default_user"


class ConversationState(BaseModel):
    conversation_id: Optional[str] = Field(None, description="Unique identifier for the conversation")
    user_message: str = Field(..., description="The current message from the user")
    user_id: str = Field("guest", description="Identifier for the current user")
    detected_intent: Optional[Intent] = Field(None, description="The detected intent from the message")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Conversation history")
    response: Optional[BaseMessage] = Field(None, description="Generated response to be returned to the user")
    errors: Optional[List[str]] = Field(default_factory=list, description="List of errors encountered during processing")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the conversation")
