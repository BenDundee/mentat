"""Tests for the FastAPI API layer."""

import pytest


@pytest.mark.anyio
async def test_health_check(async_client):
    """GET /api/health should return 200 ok."""
    response = await async_client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


@pytest.mark.anyio
async def test_chat_returns_reply(async_client):
    """POST /api/chat should return a reply with intent info."""
    payload = {
        "messages": [{"role": "user", "content": "I need help with my team"}],
        "session_id": "test-session-1",
    }
    response = await async_client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert len(data["reply"]) > 0
    assert data["session_id"] == "test-session-1"


@pytest.mark.anyio
async def test_chat_includes_orchestration_result(async_client):
    """POST /api/chat should include orchestration_result in the response."""
    payload = {
        "messages": [{"role": "user", "content": "How are best practices for 1:1s?"}]
    }
    response = await async_client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["orchestration_result"] is not None
    assert "intent" in data["orchestration_result"]
    assert "confidence" in data["orchestration_result"]


@pytest.mark.anyio
async def test_chat_empty_messages_returns_422(async_client):
    """POST /api/chat with empty messages should return 422."""
    payload = {"messages": []}
    response = await async_client.post("/api/chat", json=payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_chat_missing_messages_returns_422(async_client):
    """POST /api/chat without messages field should return 422."""
    response = await async_client.post("/api/chat", json={})
    assert response.status_code == 422
