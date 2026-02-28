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


class DocumentChunk(BaseModel, frozen=True):
    """A single chunk of text retrieved from the vector store."""

    content: str
    source: str  # "conversations" or "documents"
    document_id: str
    metadata: dict[str, str] = {}


class RAGAgentResult(BaseModel, frozen=True):
    """Result produced by the RAG Agent."""

    query: str  # the generated search query (for debugging)
    chunks: tuple[DocumentChunk, ...] = ()
    summary: str


class ContextManagementResult(BaseModel, frozen=True):
    """Result produced by the Context Management Agent."""

    coaching_brief: str  # actionable instructions for the Coaching Agent
    # e.g. "opening", "exploration", "insight", "action-planning", "closing"
    session_phase: str
    # e.g. "exploratory and Socratic — resist giving direct advice"
    tone_guidance: str
    key_information: str  # ranked, compressed excerpts from Search + RAG
    conversation_summary: str  # compressed history (2-3 sentences)
