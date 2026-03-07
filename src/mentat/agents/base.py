"""Abstract base class for all Mentat agents."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from mentat.core.config import AgentConfig, load_agent_config
from mentat.core.logging import get_logger
from mentat.core.providers import build_llm
from mentat.graph.state import GraphState


class BaseAgent(ABC):
    """Base class all Mentat agents inherit from.

    Subclasses must set AGENT_NAME (used to locate the config file) and
    implement ``run()``.

    Adding a new agent:
        1. Create ``src/mentat/agents/<name>.py``
        2. Set ``AGENT_NAME = "<name>"``
        3. Implement ``run(state) -> GraphState``
        4. Add ``configs/<name>.yml``
    """

    AGENT_NAME: str  # subclasses must define this

    def __init__(self) -> None:
        self._logger = get_logger(self.__class__.__name__)
        self.config: AgentConfig = load_agent_config(self.AGENT_NAME)
        self.llm: ChatOpenAI = build_llm(self.config)
        self.prompt_template: ChatPromptTemplate = self._build_prompt_template()

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """Construct the default prompt template from the agent config."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.config.system_prompt),
                ("human", "{user_message}"),
            ]
        )

    def _return_state(self, state: GraphState, **updates: Any) -> GraphState:
        """Return a new GraphState merging current state with updates."""
        return GraphState(**{**state, **updates})  # type: ignore[misc]

    def _format_message_history(self, messages: list, recent_count: int) -> str:
        """Format recent message history as a string for LLM context.

        Args:
            messages: Full message list from state.
            recent_count: How many recent messages to include.

        Returns:
            Formatted string; ``"(no history)"`` when messages is empty.
        """
        recent = messages[-recent_count:]
        history_lines = []
        for msg in recent:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", "")
            history_lines.append(f"{role}: {content}")
        return "\n".join(history_lines) if history_lines else "(no history)"

    @staticmethod
    def _now() -> str:
        """Return the current UTC datetime as a human-readable string."""
        return datetime.now(timezone.utc).strftime("%A, %d %B %Y at %H:%M UTC")

    @abstractmethod
    def run(self, state: GraphState) -> GraphState:
        """Execute the agent's logic against the current graph state.

        Implementations must NOT mutate ``state``.  Return a new
        ``GraphState`` with updated fields.
        """
        ...
