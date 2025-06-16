from atomic_agents.agents.base_agent import BaseIOSchema
from typing import List
from pydantic import Field

from src.types.query import QueryResult


class PersonaAgentInputSchema(BaseIOSchema):
    """Input schema for the PersonaAgent"""
    query_result: QueryResult = Field(..., description="A list of query results to use for context generation")


class Persona(BaseIOSchema):
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