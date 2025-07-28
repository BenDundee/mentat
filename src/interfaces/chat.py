import datetime as dt
from typing import List, Optional

from pydantic import Field
from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.components.agent_memory import Message

from .action import ActionDirective
from .persona import Persona
from .coach import CoachingSessionState, CoachingStage
from .orchestration import Intent


class SimpleMessageContentIOSchema(BaseIOSchema):
    """Schema for the content of a simple message."""
    content: str = Field(..., description="The text of the message")


class TurnState(BaseIOSchema):
    """Schema for the state of the current turn in a conversation between a user and an AI agent."""
    user_message: Optional[Message] = Field(None, description="The current message from the user")
    response: Optional[Message] = Field(None, description="The response to the user's message")
    intent: Optional[Intent] = Field(None, description="The detected intent from the user's input")
    intent_reasoning: Optional[str] = Field(None, description="Explanation of why the detected intent was chosen")
    intent_confidence: Optional[int] = Field(None, description="Confidence score for the detected intent, between 0 and 100")
    conversation_summary: Optional[str] = Field(None, description="A brief summary of the conversation so far, it should not exceed 200 words.")
    response_outline: Optional[str] = Field(None, description="A high-level summary of what the response to the user should look like, it should not exceed 200 words.")
    coaching_stage: Optional[CoachingStage] = Field(None, description="Current stage of the coaching session")
    action_directives: Optional[List[ActionDirective]] = Field(default_factory=list, description="List of actions to preform")
    errors: Optional[List[str]] = Field(default_factory=list, description="List of errors encountered during processing")
    turn_start: str = Field(default_factory=lambda: dt.datetime.now().isoformat(), description="Timestamp for the start of the turn")
    turn_end: Optional[str] = Field(None, description="Timestamp for the end of the turn")


class ConversationState(BaseIOSchema):
    """Schema for the conversation state."""
    conversation_id: Optional[str] = Field(None, description="Unique identifier for the conversation")
    history: Optional[List[TurnState]] = Field(default_factory=list, description="The conversation history")
    user_id: Optional[str] = Field("guest", description="Identifier for the current user")
    persona: Optional[Persona] = Field(Persona.get_empty_persona(), description="The persona for the current user")
    coaching_session: Optional[CoachingSessionState] = Field(None, description="Details of the current coaching session")
    conversation_start: str = Field(default_factory=lambda: dt.datetime.now().isoformat(), description="Timestamp for the start of the conversation")
    conversation_end: Optional[str] = Field(None, description="Timestamp for the end of the conversation")



