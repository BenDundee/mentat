from pydantic import BaseModel, Field
from typing import Dict, List


class SemanticQuery(BaseModel):
    """A query against a semantic database"""
    query: str = Field(..., description="The query to be executed")
    reasoning: str = Field(..., description="The reasoning for the query")

class Chunk(BaseModel):
    """This schema represents a single chunk of context"""
    text: str = Field(..., description="The text of the chunk")
    metadata: Dict[str, str] = Field(..., description="The metadata associated with the chunk")
    distance: float = Field(..., description="The distance between the chunk and the query")
    id: str = Field(..., description="The ID of the chunk")

class SemanticQueryResult(BaseModel):
    query: SemanticQuery = Field(..., description="The query that was executed")
    result: List[Chunk] = Field(..., description="The result of the query")