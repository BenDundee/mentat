"""Abstract base class for all Mentat agents."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone

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
