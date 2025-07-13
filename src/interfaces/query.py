from pydantic import Field
from typing import Dict, List
from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase


class Chunk(BaseIOSchema):
    """This schema represents a single chunk of context"""
    text: str = Field(..., description="The text of the chunk")
    metadata: Dict[str, str] = Field(..., description="The metadata associated with the chunk")
    distance: float = Field(..., description="The distance between the chunk and the query")
    id: str = Field(..., description="The ID of the chunk")


#class QueryResult(BaseIOSchema):
#    """This schema represents the result of a query"""
#    # results: List[Chunk] = Field(..., description="A list of chunks returned by the query")
#    results: str = Field(..., description="A string representation of the results")


class QueryAgentInputSchema(BaseIOSchema):
    """ Input schema for the QueryAgent """
    query_prompt: str = Field(..., description="Input from a user, for which queries will be constructed")


class QueryAgentOutputSchema(BaseIOSchema):
    """Output schema for the query agent."""
    reasoning: str = Field(..., description="The reasoning process leading up to the final query")
    queries: List[str] = Field(..., description="Semantic search queries to use for retrieving relevant chunks")


class QueryAgentContextProvider(SystemPromptContextProviderBase):
    """Context provider for the query agent."""

    def __init__(self, title="query_context"):
        super().__init__(title)
        self.query_prompt = ""

    def clear(self):
        self.query_prompt = ""

    def get_info(self) -> str:
        return f"Use the following directive to construct the queries:\n\n{self.query_prompt}"