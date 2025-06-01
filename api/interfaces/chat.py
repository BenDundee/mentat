from enum import Enum
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from langchain.schema import ChatMessage
from langchain_core.messages import BaseMessage


class ChatRequest(BaseModel):
    """Input schema for the chat endpoint."""
    message: str = Field(..., description="The message to send to the chatbot")
    history: Optional[List[ChatMessage]] = Field(None, description="Conversation history")
    user_id: Optional[str] = Field("guest", description="Identifier for the current user")
    session_id: Optional[str] = Field(None, description="Unique identifier for the conversation")


class ChatResponse(BaseModel):
    """Output schema for the chat endpoint."""
    message: str = Field(..., description="The response message from the chatbot")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    detected_intent: Optional[str] = Field(None, description="The detected intent from the user's message")
    success: bool = Field(True, description="Whether the request was processed successfully")


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


class CoachingStage(Enum):
    INITIATION = "initiation"
    GOAL_SETTING = "goal_setting"
    EXPLORATION = "exploration"
    CONCLUSION = "conclusion"
    ASSIGNMENT = "assignment"

    @staticmethod
    def state_descriptions():
        return {
            CoachingStage.INITIATION.value: "Initial state of the coaching session",
            CoachingStage.GOAL_SETTING.value: "Setting the goal for the coaching session",
            CoachingStage.EXPLORATION.value: "Exploring the area of knowledge for the coaching session",
            CoachingStage.CONCLUSION.value: "Concluding the coaching session",
            CoachingStage.ASSIGNMENT.value: "Assigning actions or tasks for the coaching session"
        }

    @staticmethod
    def get_stage(stage: str) -> "CoachingStage":
        for s in CoachingStage:
            if s.value == stage:
                return s
        raise Exception(f"Invalid stage: {stage}")

    @staticmethod
    def llm_rep():
        return '\n'.join([f"- {state}: {desc}" for state, desc in CoachingStage.state_descriptions().items()])


class ConversationState(BaseModel):
    conversation_id: Optional[str] = Field(None, description="Unique identifier for the conversation")
    user_message: str = Field(..., description="The current message from the user")
    user_id: str = Field("guest", description="Identifier for the current user")
    detected_intent: Optional[Intent] = Field(None, description="The detected intent from the message")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Conversation history")
    response: Optional[BaseMessage] = Field(None, description="Generated response to be returned to the user")
    errors: Optional[List[str]] = Field(default_factory=list, description="List of errors encountered during processing")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the conversation")


class CoachingSessionState(ConversationState):
    session_id: Optional[str] = Field(None, description="Unique identifier for this coaching session")
    stage: CoachingStage = Field(CoachingStage.INITIATION, description="Current stage of the coaching session")
    plan: Optional[str] = Field(None, description="The plan for the current session")
    goal: Optional[str] = Field(None, description="The defined goal for this coaching session")
    insights: List[str] = Field(default_factory=list, description="Key insights discovered during the session")
    assignments: List[str] = Field(default_factory=list, description="Assignments or actions for after the session")
    is_active: bool = Field(True, description="Whether this coaching session is still active")


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

