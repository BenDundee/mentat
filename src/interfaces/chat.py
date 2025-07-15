import datetime as dt
from typing import Any, Dict, List, Optional

from pydantic import Field, ValidationError
from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.components.agent_memory import AgentMemory, Message
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase

from .persona import Persona
from .coach import CoachingSessionState, CoachingStage

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
        return {
            Intent.SIMPLE.value: "General conversation, questions, or statements.",
            Intent.COACHING_SESSION_REQUEST.value:
                "A request for a coaching session, it may be either explicitly requested or implied by the conversaion. "
                "For example, a request for feedback about a specific work situation could transition into an ad hoc "
                "coaching session. ***Be proactive about finding opportunities for a coaching session.***",
            Intent.COACHING_SESSION_RESPONSE.value:
                "A response during a coaching session. Note that there must be an active coaching session for this "
                "intent to be valid.",
            Intent.FEEDBACK.value:
                "A request for feedback or help from the user. This may include things like a request for thoughts "
                "about whether a particular job posting aligns with long term goals, help drafting a resume/cover "
                "letter, or advice about a specific work situation.",
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
    user_message: Optional[Message] = Field(None, description="The current message from the user")
    history: Optional[List[Message]] = Field(None, description="The conversation history")
    user_id: Optional[str] = Field("guest", description="Identifier for the current user")
    current_turn_id: Optional[str] = Field(None, description="Unique identifier for the current turn")
    detected_intent: Optional[Intent] = Field(None, description="The detected intent from the message")
    persona: Optional[Persona] = Field(None, description="The persona for the current user")
    response: Optional[Message] = Field(None, description="Generated response to be returned to the user")
    errors: Optional[List[str]] = Field(default_factory=list, description="List of errors encountered during processing")
    coaching_session: Optional[CoachingSessionState] = Field(None, description="Details of the current coaching session")
    conversation_start: str = Field(default_factory=lambda: dt.datetime.now().isoformat(), description="Timestamp for the start of the conversation")
    conversation_end: Optional[str] = Field(None, description="Timestamp for the end of the conversation")


# class TurnState(BaseIOSchema):
#    user_message: Optional[Message] = Field(None, description="The current message from the user")
#    coaching_stage: Optional[CoachingStage] = Field(None, description="Current stage of the coaching session")
#    actions: Optional[List[str]] = Field(default_factory=list, description="List of actions to preform")
#    confidence: Optional[int] = Field(None, description="Confidence score for the detected intent, between 0 and 100")


# Update conversation state code above
# Refactor the code to pass full history, don't separate user_message, history, and response fields
# history should be a list of TurnState objects
#
#class ConversationState_(BaseIOSchema):
#    conversation_id: Optional[str] = None
#    user_id: str = "guest"
#    persona: Optional["Persona"] = None
#    coaching_session: Optional[CoachingSessionState] = None
#    history: List[TurnState] = Field(default_factory=list)
#    conversation_start: str = Field(default_factory=lambda: dt.datetime.now().isoformat())
#    conversation_end: Optional[str] = None
#    injected_context_sources: List[str] = Field(default_factory=list)

