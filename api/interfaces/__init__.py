from dataclasses import dataclass
import datetime as dt
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union

from langchain.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate


class ChatRequest(BaseModel):
    """Input schema for the chat endpoint."""
    message: str = Field(..., description="The message to send to the chatbot")
    history: Optional[List[List[str]]] = None
    user_id: Optional[str] = "default_user"


class ChatResponse(BaseModel):
    response: str


class JournalEntry(BaseModel):
    content: str = Field(..., description="The content of the journal entry")
    goal_id: Optional[int] = Field(..., description="The ID of the goal associated with the journal entry")
    user_id: str = Field(..., description="The ID of the user")
    feedback: str = Field(..., description="The feedback provided by the coach")


class VectorDBQuery(BaseModel):
    """Input schema for searching past conversations."""
    query: str = Field(description="The search query")
    user_id: str = Field(default="default_user", description="The ID of the user")
    limit: int = Field(default=3, description="Maximum number of results to return")


class Goal(BaseModel):
    """This schema represents a single coaching goal"""
    title: str = Field(..., description="The title of the coaching goal")
    user_id: str = Field(..., description="The ID of the user")
    description: str = Field(..., description="The description of the coaching goal")
    created_at: str = \
        Field(default_factory=lambda: dt.datetime.now().isoformat(), description="The start date of the coaching goal")
    end_date: Optional[str] = Field(None, description="The end date of the coaching goal")
    due_date: Optional[str] = Field(None, description="The due date of the coaching goal")
    status: Literal["in progress", "on hold", "completed"] = Field(..., description="The status of the coaching goal")


class Persona(BaseModel):
    """Persona summary for user"""
    core_values: List[str] = Field(..., description="The list of core values of the persona")
    strengths: List[str] = Field(..., description="The list of strengths of the persona")
    growth_areas: List[str] = Field(..., description="The list of growth areas of the persona")
    communication_style: str = Field(..., description="The communication style of the persona")
    preferred_feedback_style: str = Field(..., description="The preferred feedback style of the persona")
    motivators: List[str] = Field(..., description="The list of motivators of the persona")

    def get_summary(self):
        return {
            "core_values": self.core_values,
            "strengths": self.strengths,
            "growth_areas": self.growth_areas,
            "communication_style": self.communication_style,
            "preferred_feedback_style": self.preferred_feedback_style,
            "motivators": self.motivators
        }

    def __eq__(self, other):
        return self.get_summary() == other.get_summary()

    def is_empty(self):
        return self.get_summary() == self.get_empty_persona().get_summary()

    @staticmethod
    def get_empty_persona():
        return Persona(
            core_values=[],
            strengths=[],
            growth_areas=[],
            communication_style="",
            preferred_feedback_style="",
            motivators=[]
        )


# TODO: use pydntic?
@dataclass
class IntentPattern:
    """Represents a pattern for intent detection with associated keywords and description."""
    name: str
    keywords: List[str]
    description: str
    tool_name: str
    example_phrases: Optional[List[str]]


@dataclass
class LLMCredentials:
    """Credentials for accessing LLMs."""
    openai_api_key: str
    openrouter_api_key: str


@dataclass
class ModelAPIParameters:
    """Parameters for accessing LLMs."""
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class LLMParameters:
    model_provider: str
    model: str
    model_api_parameters: Optional[ModelAPIParameters] = None


@dataclass
class PromptHandler:
    prompt_name: str
    prompt_template: Union[PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate]
    llm_parameters: LLMParameters