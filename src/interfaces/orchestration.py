from pydantic import Field
from typing import Dict, Optional

from atomic_agents.lib.base.base_io_schema import BaseIOSchema

from .intent import Intent
from .action import ActionDirective


class OrchestrationAgentOutputSchema(BaseIOSchema):
    """Schema for the orchestration response."""
    intent: Intent = Field(..., description="The detected intent from the user's input")
    intent_reasoning: str = Field(..., description="Explanation of why the detected intent was chosen")
    intent_confidence: int = Field(..., description="Confidence score for the detected intent, between 0 and 100")
    conversation_summary: str = \
        Field(..., description="A brief summary of the conversation so far, it should not exceed 100 words.")
    response_outline: str = \
        Field(..., description="A high-level summary of what the response to the user should look like, "
                               "it should not exceed 100 words.")
    directives: Optional[list[ActionDirective]] = Field(default_factory=list, description="List of directives for the agent")
    errors: Optional[str] = Field(None, description="List of errors encountered during processing")

