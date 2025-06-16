from typing import Literal, Optional
from pydantic import Field
from atomic_agents.agents.base_agent import BaseIOSchema


class Goal(BaseIOSchema):
    """This schema represents a single coaching goal"""
    description: str = Field(..., description="The description of the coaching goal")
    start_date: str = Field(..., description="The start date of the coaching goal")
    end_date: Optional[str] = Field(None, description="The end date of the coaching goal")
    due_date: Optional[str] = Field(None, description="The due date of the coaching goal")
    status: Literal["in progress", "on hold", "completed"] = Field(..., description="The status of the coaching goal")