from .conversation_search import ConversationSearchTool
from .goal_tracker import GoalTrackerTool
from .journal_manager import JournalManagerTool
from .tool_execution_context import ToolExecutionContext

from typing import Dict, List, Tuple, Optional
from langchain.tools import BaseTool

from sqlite3 import Connection
from langchain_chroma import Chroma

from api.interfaces import IntentPattern


class __ToolRegistry:
    """Registry for tools and their associated intents."""

    def __init__(self, vector_db: Chroma, db_connection: Connection) -> None:
        self.tools, self.intents = configure_tools(vector_db, db_connection)

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool in the registry."""
        self.tools[tool.name] = tool

    def register_tools(self, tools: List[BaseTool]) -> None:
        """Register a list of tools in the registry."""
        _ = [self.register_tool(tool) for tool in tools]

    def register_intent(self, intent: IntentPattern) -> None:
        """Register an intent pattern in the registry."""
        self.intents[intent.name] = intent

    def get_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self.tools.values())

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def detect_intent(self, message: str) -> List[str]:
        """Detect intents from a message."""
        message_lower = message.lower()
        matched_intents = []

        for intent_name, intent in self.intents.items():
            if any(keyword in message_lower for keyword in intent.keywords):
                matched_intents.append((intent_name, intent))

        return matched_intents

    def generate_system_prompt_section(self) -> str:
        """Generate a section of the system prompt describing available tools and intents."""
        prompt_sections = [
            "Based on the user's message, determine what they need help with and use the appropriate tools:"]

        for intent in self.intents.values():
            examples = f" (e.g., '{', '.join(intent.example_phrases)}')" if intent.example_phrases else ""
            prompt_sections.append(f"- Use {intent.tool_name} for {intent.description}{examples}")

        return "\n".join(prompt_sections)


def get_tool_registry(vector_db: Chroma, db_connection: Connection) -> __ToolRegistry:
    return __ToolRegistry(vector_db, db_connection)


# ----------------------------------------------------------------------------------------------------------------------
# --------------  Add tools and intents here ---------------------------------------------------------------------------
# When adding new tools, register them below
def configure_tools(
        vector_db: Chroma,
        db_connection: Connection
) -> Tuple[Dict[str, BaseTool], Dict[str, IntentPattern]]:
    """Configure tools with vector database, LLM, and database connection.

    NOTE TO SELF: LEVERAGE TOOL.name WHEREVER POSSIBLE!

    :param vector_db: ChromaDB vector database
    :param db_connection: SQLite database connection
    :return tools and intents: Tuple[Dict[str, BaseTool], List[IntentPattern]]
    """
    _tools = [
        GoalTrackerTool(db_connection),
        JournalManagerTool(db_connection, vector_db),
        ConversationSearchTool(vector_db)
    ]

    _intents = [
        IntentPattern(
            name="goal_tracking",
            keywords=["goal", "objective", "target", "achieve", "milestone"],
            description="setting, updating, or reviewing professional goals",
            tool_name=GoalTrackerTool.name,
            example_phrases=["I want to set a new goal", "How am I progressing on my goals?"]
        ), IntentPattern(
            name="journaling",
            keywords=["journal", "reflect", "write", "thought", "feelings"],
            description="reflective journaling, generating prompts, or reviewing past entries",
            tool_name=JournalManagerTool.name,
            example_phrases=["I'd like to journal about my day", "Give me a reflection prompt"]
        ), IntentPattern(
            name="conversation_recall",
            keywords=["remember", "last time", "previously", "you said", "recall"],
            description="recalling relevant previous discussions",
            tool_name=ConversationSearchTool.name,
            example_phrases=["What did we discuss last time?", "Can you recall our conversation about leadership?"]
        )
    ]

    return (
        {tool.name: tool for tool in _tools},
        {intent.name: intent for intent in _intents}
    )





