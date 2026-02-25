"""Pydantic schemas for the Mentat API."""

from pydantic import BaseModel, Field

from mentat.core.models import Message, OrchestrationResult


class ChatRequest(BaseModel, frozen=True):
    """Request body for POST /api/chat."""

    messages: list[Message] = Field(
        ..., description="Full conversation history, newest message last."
    )
    session_id: str | None = Field(
        default=None,
        description="Client-generated session identifier (used in Phase 2).",
    )


class ChatResponse(BaseModel, frozen=True):
    """Response body from POST /api/chat."""

    reply: str = Field(..., description="Assistant's reply text.")
    orchestration_result: OrchestrationResult | None = Field(
        default=None,
        description="Intent classification result (exposed for debugging).",
    )
    session_id: str | None = Field(default=None)
