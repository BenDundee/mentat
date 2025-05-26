from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type, Literal, ClassVar
import sqlite3


# TODO: move to `interfaces`
class GoalTrackerInput(BaseModel):
    """Input schema for the GoalTracker tool."""
    query: str = Field(description="The goal-related query or command")
    goal_action: Literal["list all", "list open", "add"] = Field(description="The goal action to perform")
    user_id: str = Field(default="default_user", description="The ID of the user")


class GoalTrackerTool(BaseTool):
    name: str = "goal_tracker"
    description: str = """Track and manage the user's professional goals. Use this for:
    - Creating new goals: "add goal to improve presentation skills"
    - Listing goals: "list my goals"
    - Updating goals: "update my presentation skills goal"
    - Checking progress: "check progress on presentation skills goal"
    """
    args_schema: Type[BaseModel] = GoalTrackerInput
    conn: sqlite3.Connection = None

    def __init__(self, db_connection):
        """Initialize with a database connection."""
        super().__init__()
        self.conn = db_connection

    @staticmethod
    def return_name():
        return "goal_tracker"

    def _run(self, query: str, user_id: str = "default_user") -> str:
        """Execute goal-related operations."""
        cursor = self.conn.cursor()

        # Simple intent detection for goal operations
        query_lower = query.lower()

        if "list all" in query_lower:
            cursor.execute("SELECT title, status FROM goals WHERE user_id = ?", (user_id,))
            goals = cursor.fetchall()

            if not goals:
                return "You don't have any goals set up yet. Would you like to create one?"

            response = "Here are your current goals:\n"
            for title, status in goals:
                response += f"- {title} (Status: {status})\n"
            return response

        elif "add goal" in query_lower or "create goal" in query_lower or "new goal" in query_lower:
            # Extract goal title
            title_start = query.find("goal") + 5
            title = query[title_start:].strip()
            if not title:
                title = "Untitled Goal"

            cursor.execute(
                "INSERT INTO goals (user_id, title) VALUES (?, ?)",
                (user_id, title)
            )
            self.conn.commit()
            return f"I've added your new goal: '{title}'. What's your timeline for achieving this?"

        # Handle more goal operations here

        return "I can help you track goals. Try saying 'list my goals' or 'add a new goal to...'"

    async def _arun(self, query: str, user_id: str = "default_user") -> str:
        """Async implementation of the tool."""
        return self._run(query, user_id)