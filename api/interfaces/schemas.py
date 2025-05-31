from enum import Enum
from pydantic import BaseModel, Field, ValidationError
from typing import Dict


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

class IntentDetectionResponse(BaseModel):
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

class SimpleResponderResponse(BaseModel):
    """Schema for the simple responder response."""
    response: str = Field(..., description="The generated response.")


