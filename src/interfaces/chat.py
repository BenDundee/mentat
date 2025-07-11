import datetime as dt
from typing import Any, Dict, List, Optional

from pydantic import Field, ValidationError
from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.components.agent_memory import AgentMemory, Message

from src.interfaces.persona import Persona

from enum import Enum


class SimpleMessageContentIOSchema(BaseIOSchema):
    """ A message in a conversation between a human and an AI agent """
    content: str = \
        Field(..., description="A message containing simple text input in a "
                               "conversation between a human and an AI agent")


class Intent(Enum):
    SIMPLE = "simple_message"
    COACHING_SESSION_REQUEST = "coaching_session_request"
    COACHING_SESSION_RESPONSE = "coaching_session_response"
    FEEDBACK = "feedback"

    @staticmethod
    def intent_descriptions():
        return {  # TODO: May need to make these more...robust
            Intent.SIMPLE.value: "General conversation, questions, or statements",
            Intent.COACHING_SESSION_REQUEST.value: "A request for a coaching session.",
            Intent.COACHING_SESSION_RESPONSE.value: "A response during a coaching session",
            Intent.FEEDBACK.value: "A request for feedback or help from the user.",
        }

    @staticmethod
    def get_intent(intent: str) -> "Intent":
        for i in Intent:
            if i.value == intent:
                return i
        raise Exception(f"Invalid intent: {intent}")

    @staticmethod
    def llm_rep():
        return '\n'.join([f"- {intent}: {desc}" for intent, desc in Intent.intent_descriptions().items()])


class IntentDetectionResponse(BaseIOSchema):
    """Schema for the intent detection response."""
    intent: Intent = Field(description="Detected intent")
    confidence: int = Field(description="Confidence score for the detected intent, between 0 and 100")
    reasoning: str = Field(description="Explanation of why this intent was chosen")

    @staticmethod
    def parser(raw: Dict[str, str]) -> "IntentDetectionResponse":
        try:
            assert "intent" in raw and "confidence" in raw and "reasoning" in raw
        except AssertionError:
            raise ValidationError("Invalid raw response from LLM: missing required keys")
        return IntentDetectionResponse(
            intent=Intent.get_intent(raw["intent"]),
            confidence=int(raw["confidence"]),
            reasoning=raw["reasoning"]
        )


class ConversationState(BaseIOSchema):
    """Schema for the conversation state."""
    conversation_id: Optional[str] = Field(None, description="Unique identifier for the conversation")
    user_message: Optional[Message] = Field(..., description="The current message from the user")
    history: Optional[List[Message]] = Field(..., description="The conversation history")
    user_id: Optional[str] = Field("guest", description="Identifier for the current user")
    detected_intent: Optional[Intent] = Field(None, description="The detected intent from the message")
    persona: Optional[Persona] = Field(..., description="The persona for the current user")
    response: Optional[Message] = Field(None, description="Generated response to be returned to the user")
    errors: Optional[List[str]] = Field(default_factory=list, description="List of errors encountered during processing")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for the conversation")
    conversation_start: str = Field(default_factory=lambda: dt.datetime.now().isoformat(), description="Timestamp for the start of the conversation")
    conversation_end: Optional[str] = Field(None, description="Timestamp for the end of the conversation")

    def set_response(self, response: str, turn_id: str):
        self.response = Message(content=SimpleMessageContentIOSchema(content=response), role="assistant", turn_id=turn_id)





