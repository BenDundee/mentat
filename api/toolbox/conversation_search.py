from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.vectorstores import Chroma

from api.interfaces import VectorDBQuery


class ConversationSearchTool(BaseTool):
    name: str = "conversation_search"
    description: str = "Search for relevant past conversations to provide context for the current discussion."
    args_schema: Type[BaseModel] = VectorDBQuery

    def __init__(self, vector_db):
        """Initialize with a vector database."""
        super().__init__()
        self.vector_db = vector_db

    def _run(self, query: str, user_id: str = "default_user", limit: int = 3) -> str:
        """Search for relevant past conversations."""
        results = self.vector_db.similarity_search(
            query=query,
            filter={"user_id": user_id},
            k=limit
        )

        if not results:
            return ""

        context = "Relevant past conversations:\n\n"
        for doc in results:
            context += doc.page_content + "\n\n"

        return context

    async def _arun(self, query: str, user_id: str = "default_user", limit: int = 3) -> str:
        """Async implementation of the tool."""
        return self._run(query, user_id, limit)