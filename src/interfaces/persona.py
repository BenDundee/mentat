from atomic_agents.agents.base_agent import BaseIOSchema
from typing import Any, Dict, List, Optional
from pydantic import Field
import simplejson as sj

from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase


class Persona(BaseIOSchema):
    """Persona summary for user"""
    core_values: List[str] = Field(..., description="The list of core values of the persona")
    strengths: List[str] = Field(..., description="The list of strengths of the persona")
    growth_areas: List[str] = Field(..., description="The list of growth areas of the persona")
    communication_style: str = Field(..., description="The communication style of the persona")
    preferred_feedback_style: str = Field(..., description="The preferred feedback style of the persona")
    motivators: List[str] = Field(..., description="The list of motivators of the persona")

    def get_summary(self) -> Dict[str, Any]:
        return {
            "core_values": self.core_values,
            "strengths": self.strengths,
            "growth_areas": self.growth_areas,
            "communication_style": self.communication_style,
            "preferred_feedback_style": self.preferred_feedback_style,
            "motivators": self.motivators
        }

    def __str__(self):
        return sj.dumps(self.get_summary(), indent=4)

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


