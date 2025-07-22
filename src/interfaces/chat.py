import datetime as dt
from typing import List, Optional

from pydantic import Field
from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.components.agent_memory import Message

from .persona import Persona
from .coach import CoachingSessionState, CoachingStage
from .orchestration import Intent

class SimpleMessageContentIOSchema(BaseIOSchema):
    """ A message in a conversation between a human and an AI agent """
    content: str = \
        Field(..., description="A message containing simple text input in a "
                               "conversation between a human and an AI agent")


class TurnState(BaseIOSchema):
    """Schema for the state of the current turn in a conversation between a user and an AI agent."""
    user_message: Optional[Message] = Field(None, description="The current message from the user")
    response: Optional[Message] = Field(None, description="The response to the user's message")
    coaching_stage: Optional[CoachingStage] = Field(None, description="Current stage of the coaching session")
    actions: Optional[List[str]] = Field(default_factory=list, description="List of actions to preform")
    errors: Optional[List[str]] = Field(default_factory=list, description="List of errors encountered during processing")
    detected_intent: Optional[Intent] = Field(None, description="The detected intent from the message")
    confidence: Optional[int] = Field(None, description="Confidence score for the detected intent, between 0 and 100")
    turn_start: str = Field(default_factory=lambda: dt.datetime.now().isoformat(), description="Timestamp for the start of the turn")
    turn_end: Optional[str] = Field(None, description="Timestamp for the end of the turn")


class ConversationState(BaseIOSchema):
    """Schema for the conversation state."""
    conversation_id: Optional[str] = Field(None, description="Unique identifier for the conversation")
    history: Optional[List[TurnState]] = Field(default_factory=list, description="The conversation history")
    user_id: Optional[str] = Field("guest", description="Identifier for the current user")
    persona: Optional[Persona] = Field(None, description="The persona for the current user")
    coaching_session: Optional[CoachingSessionState] = Field(None, description="Details of the current coaching session")
    conversation_start: str = Field(default_factory=lambda: dt.datetime.now().isoformat(), description="Timestamp for the start of the conversation")
    conversation_end: Optional[str] = Field(None, description="Timestamp for the end of the conversation")


