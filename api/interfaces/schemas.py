from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal

from langchain.schema import ChatMessage


class ChatRequest(BaseModel):
    """Input schema for the chat endpoint."""
    message: str = Field(..., description="The message to send to the chatbot")
    history: Optional[List[ChatMessage]] = None
    user_id: Optional[str] = "default_user"

class Intent(Enum):
    pass

class IntentDetectionResponse(BaseModel):
    """Schema for the intent detection response."""
    model_config = ConfigDict(use_enum_values=True)
    intent: Intent = Field(description="Detected intent")
    confidence: float = Field(description="Confidence score for the detected intent (0-1)")
    reasoning: str = Field(description="Explanation of why this intent was chosen")


