from pydantic import Field
from typing import Dict, List
from atomic_agents.agents.base_agent import BaseIOSchema


class Chunk(BaseIOSchema):
    """This schema represents a single chunk of context"""
    text: str = Field(..., description="The text of the chunk")
    metadata: Dict[str, str] = Field(..., description="The metadata associated with the chunk")
    distance: float = Field(..., description="The distance between the chunk and the query")
    id: str = Field(..., description="The ID of the chunk")


class QueryResult(BaseIOSchema):
    """This schema represents the result of a query"""
    results: List[Chunk] = Field(..., description="A list of chunks returned by the query")


class QueryAgentInputSchema(BaseIOSchema):
    """ Input schema for the QueryAgent """
    query_prompt: str = Field(..., description="Input from a user, for which queries will be constructed")


class QueryAgentOutputSchema(BaseIOSchema):
    """Output schema for the query agent."""
    reasoning: str = Field(..., description="The reasoning process leading up to the final query")
    query: str = Field(..., description="The semantic search query to use for retrieving relevant chunks")