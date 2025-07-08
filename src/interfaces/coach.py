from src.interfaces.chat import ConversationState
from typing import List, Optional
from pydantic import Field
from enum import Enum


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


class CoachingSessionState(ConversationState):
    """Schema for the coaching session state."""
    session_id: Optional[str] = Field(None, description="Unique identifier for this coaching session")
    stage: Optional[CoachingStage] = Field(CoachingStage.CONTRACT, description="Current stage of the coaching session")
    plan: Optional[str] = Field(None, description="The plan for the current session")
    goal: Optional[str] = Field(None, description="The defined goal for this coaching session")
    insights: Optional[List[str]] = Field(default_factory=list, description="Key insights discovered during the session")
    assignments: Optional[str] = Field(default_factory=list, description="Assignments or actions for after the session")
    is_active: bool = Field(False, description="Whether this coaching session is still active")
    summary: Optional[str] = Field("", description="Summary of the coaching session")
