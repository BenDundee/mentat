"""Core data models for Mentat."""

from enum import Enum

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Roles that can send messages in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel, frozen=True):
    """A single message in a conversation."""

    role: MessageRole
    content: str


class Intent(str, Enum):
    """Possible user intents classified by the Orchestration Agent."""

    CHECK_IN = "check-in"
    COACHING_SESSION = "coaching-session"
    QUESTION = "question"
    DOCUMENT_REVIEW = "document-review"
    OFF_TOPIC = "off-topic"
    UNKNOWN = "unknown"


class OrchestrationResult(BaseModel, frozen=True):
    """Result produced by the Orchestration Agent after classifying intent."""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    # Phase 2+: drives routing; empty in Phase 1
    suggested_agents: tuple[str, ...] = ()


class SearchResult(BaseModel, frozen=True):
    """A single search result retrieved from DuckDuckGo."""

    title: str
    url: str
    snippet: str
    retrieved_at: str  # ISO-8601 UTC timestamp string


class SearchAgentResult(BaseModel, frozen=True):
    """Result produced by the Search Agent."""

    queries: tuple[str, ...]
    results: tuple[SearchResult, ...]
    summary: str
