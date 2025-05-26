from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import sqlite3
from langchain_chroma import Chroma


# TODO: move to interfaces
class JournalInput(BaseModel):
    """Input schema for the Journal tool."""
    query: str = Field(description="The journal-related query or command")
    user_id: str = Field(default="default_user", description="The ID of the user")


class JournalManagerTool(BaseTool):
    name: str = "journal_manager"
    description: str = """Manage the user's reflective journal. Use this for:
    - Creating new entries: "add journal entry about today's meeting"
    - Generating prompts: "give me a journal prompt about leadership"
    - Reviewing past entries: "show my recent journal entries"
    """
    args_schema: Type[BaseModel] = JournalInput
    conn: sqlite3.Connection = None
    vector_db: Chroma = None

    def __init__(self, db_connection, vector_db):
        """Initialize with database connections and LLM."""
        super().__init__()
        self.conn = db_connection
        self.vector_db = vector_db

    @staticmethod
    def return_name():
        return "journal_manager"

    def _run(self, query: str, user_id: str = "default_user") -> str:
        """Execute journal-related operations."""
        cursor = self.conn.cursor()
        query_lower = query.lower()

        if "create prompt" in query_lower or "journal prompt" in query_lower:
            # Get goals for context
            cursor.execute("SELECT title FROM goals WHERE user_id = ? AND status = 'active'", (user_id,))
            goals = cursor.fetchall()
            goal_titles = [g[0] for g in goals] if goals else []

            # Get journal prompt template from prompt manager
            journal_prompt_template = self.prompt_manager.get_prompt("journal_prompt_generator")
            llm_settings = self.prompt_manager.get_llm_settings("journal_prompt_generator")

            # Format the prompt with context
            formatted_prompt = journal_prompt_template.format(
                goals=", ".join(goal_titles) if goal_titles else "No specific goals set yet",
                recent_entries=self._get_recent_entries(user_id)
            )

            # Use the LLM with specific settings for this prompt
            from langchain.schema import HumanMessage
            response = self.llm([HumanMessage(content=formatted_prompt)], **llm_settings)
            return response.content

        return "I can help with your journaling practice. Try asking for a prompt or creating a new entry."

    async def _arun(self, query: str, user_id: str = "default_user") -> str:
        """Async implementation of the tool."""
        return self._run(query, user_id)