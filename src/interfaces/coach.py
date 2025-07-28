from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional
from pydantic import Field
from enum import Enum
import datetime as dt

from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.components.agent_memory import Message

#if TYPE_CHECKING:
from .persona import Persona


class CoachingStage(Enum):
    CONTRACT = "contract"
    LISTEN = "listen"
    EXPLORE = "explore"
    ACTION_PLANNING = "action_planning"
    REVIEW = "review"

    @staticmethod
    def descriptions():
        return {
            CoachingStage.CONTRACT.value: [
                "Establish the purpose of the session",
                "Confirm long-term goals and user context"
                "Clarify expectations for the session",
            ],
            CoachingStage.LISTEN.value: [
                "Encourage deep reflection and open expression.",
                "Explore emotions, beliefs, and challenges.",
                "Incorporate feedback documents and past sessions."
            ],
            CoachingStage.EXPLORE.value: [
                "Help uncover patterns, obstacles, and possibilities.",
                "Link discoveries to long-term objectives.",
                "Use RAG (retrieval-augmented generation) for relevant context.",
            ],
            CoachingStage.ACTION_PLANNING.value: [
                "Build an actionable goal by asking the following questions:",
                "**Goal**: What do you want to achieve next?",
                "**Reality**: Where are you now?",
                "**Options**: What can you try?",
                "**Will**: What will you commit to?"
            ],
            CoachingStage.REVIEW.value: [
                "Reflect on insights gained during the session.",
                "Offer journaling or behavioral assignments.",
                "Reinforce connection to long-term goals."
            ]
        }

    @staticmethod
    def get_stage(stage: str) -> "CoachingStage":
        for s in CoachingStage:
            if s.value == stage:
                return s
        raise Exception(f"Invalid stage: {stage}")

    @staticmethod
    def llm_rep():
        out = []
        for stage, desc in CoachingStage.descriptions().items():
            out.append(f"  • `{stage}`:")
            for d in desc:
                out.append(f"    ○ {d}")
        return "\n".join(out)


class Assignment(BaseIOSchema):
    """Schema for an assignment or action to be taken during a coaching session."""
    title: str
    description: Optional[str] = None
    due_date: Optional[str] = None  # ISO format


class Goal(BaseIOSchema):
    """Schema for a long-term coaching goal."""
    title: str = Field(..., description="The title of the goal")
    description: str = Field(None, description="The description of the goal")
    due_date: dt.datetime = Field(None, description="The due date for this goal")


class CoachingSessionState(BaseIOSchema):
    """Schema for the coaching session state."""
    session_id: Optional[str] = Field(None, description="Unique identifier for this coaching session")
    stage: Optional[CoachingStage] = Field(CoachingStage.CONTRACT, description="Current stage of the coaching session")
    session_plan: Optional[str] = Field(None, description="The plan for the current session")
    session_goal: Optional[str] = Field(None, description="The defined goal for this coaching session")
    long_term_goal_ref: Optional[List[Goal]] = Field(None, description="The reference to the long-term goal")
    action_plan: Optional[str] = Field(None, description="The action plan for the session")
    insights: Optional[List[str]] = Field(default_factory=list, description="Key insights discovered during the session")
    assignments: Optional[str] = Field(default_factory=list, description="Assignments or actions for after the session")
    is_active: bool = Field(False, description="Whether this coaching session is still active")
    summary: Optional[str] = Field("", description="Summary of the coaching session")
    session_completion_reason: Optional[str] = Field(None, description="Reason for session completion")

    @staticmethod
    def get_new_session(session_id: str) -> "CoachingSessionState":
        return CoachingSessionState(
            session_id=session_id,
            stage=CoachingStage.CONTRACT,
            session_plan="",
            session_goal="",
            insights=[],
            assignments="",
            is_active=False,
            summary=""
        )


class CoachingAgentInputSchema(BaseIOSchema):
    """Schema for the input to the coaching agent."""
    user_message: Message = Field(..., description="User message from the user")
    persona: Persona = Field(..., description="The persona for the current user")
    conversation_summary: str = Field(..., description="A brief summary of the conversation so far, you may not have access to the full history.")
    response_outline: str = Field(..., description="A high-level recommendation for a response to the user.")
    coaching_stage: Optional[CoachingStage] = Field(None, description="Current stage of the coaching session, if a session is active")


class CoachResponse(BaseIOSchema):
    """Schema for the coach response."""
    response: str = Field(..., description="Response from the coach")
    reasoning: str = Field(..., description="Explanation of why this response was chosen")
    insights: Optional[List[str]] = Field(default_factory=list, description="Insight about the user, based on what youve seen so far.")


if __name__ == "__main__":
    print(CoachingStage.llm_rep())
    print("wait")