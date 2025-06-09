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
    CONTRACT = "contract"
    LISTEN = "listen"
    EXPLORE = "explore"
    ACTION_PLANNING = "action_planning"
    REVIEW = "review"

    @staticmethod
    def state_descriptions():
        return {
            CoachingStage.CONTRACT: \
                " - Establish the purpose of the session"
                " - Confirm long-term goals and user context"
                " - Clarify expectations for the session.",
            CoachingStage.LISTEN: \
                " - Encourage deep reflection and open expression."
                " - Explore emotions, beliefs, and challenges." 
                " - Incorporate feedback documents and past sessions.",
            CoachingStage.EXPLORE: \
                " - Help uncover patterns, obstacles, and possibilities."
                " - Link discoveries to long-term objectives."
                " - Use RAG (retrieval-augmented generation) for relevant context.",
            CoachingStage.ACTION_PLANNING: \
                " -  Identify actionable next steps"
                " -  Use GROW structure:"
                " -- **Goal**: What do you want to achieve next?"
                " -- **Reality**: Where are you now?"
                " -- **Options**: What can you try?"
                " -- **Will**: What will you commit to?",
            CoachingStage.REVIEW: \
                " - Reflect on insights gained during the session."
                " - Offer journaling or behavioral assignments."
                " - Reinforce connection to long-term goals.",
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
    history: Optional[List[BaseMessage]] = Field(default_factory=list, description="Conversation history")
    response: Optional[BaseMessage] = Field(None, description="Generated response to be returned to the user")
    errors: Optional[List[str]] = Field(default_factory=list, description="List of errors encountered during processing")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the conversation")

    @property
    def stringify_history(self) -> str:
        return '\n'.join(message.pretty_print() for message in self.history)

class CoachingSessionState(ConversationState):
    session_id: Optional[str] = Field(None, description="Unique identifier for this coaching session")
    stage: CoachingStage = Field(CoachingStage.CONTRACT, description="Current stage of the coaching session")
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

