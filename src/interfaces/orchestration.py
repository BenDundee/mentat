from pydantic import Field
from typing import Dict, Optional, List

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.components.agent_memory import Message

from .intent import Intent
from .action import ActionDirective
from .coach import CoachingStage


class OrchestrationAgentInputSchema(BaseIOSchema):
    """Schema for the orchestration request."""
    user_input: Message = Field(..., description="Input from a user, for which queries will be constructed")


class OrchestrationAgentOutputSchema(BaseIOSchema):
    """Schema for the orchestration response."""
    intent: Optional[Intent] = Field(None, description="The detected intent from the user's input")
    intent_reasoning: Optional[str] = Field(None, description="Explanation of why the detected intent was chosen")
    intent_confidence: Optional[int] = \
        Field(None, description="Confidence score for the detected intent, an integer between 0 (no confidence) and 100 (full confidence)")
    conversation_summary: Optional[str] = Field(None, description="A brief summary of the conversation so far, it should not exceed 200 words.")
    response_outline: Optional[str] = Field(None, description="A high-level summary of what the response to the user should look like, it should not exceed 200 words.")
    coaching_stage: Optional[CoachingStage] = Field(None, description="Current stage of the coaching session")
    action_directives: Optional[List[ActionDirective]] = Field(default_factory=list, description="List of actions to preform")
    errors: Optional[str] = Field(None, description="List of errors encountered during processing")

