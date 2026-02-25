"""API route handlers."""

from fastapi import APIRouter, HTTPException, Request

from mentat.api.schemas import ChatRequest, ChatResponse
from mentat.core.logging import get_logger
from mentat.graph.state import GraphState

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Simple liveness check."""
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def handle_chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Process a chat message through the Mentat agent graph.

    The client sends the full conversation history on every request.
    The server is stateless in Phase 1.
    """
    if not body.messages:
        raise HTTPException(status_code=422, detail="messages list cannot be empty")

    last_message = body.messages[-1]
    user_message = last_message.content

    logger.info("Received chat request. session_id=%s", body.session_id)

    graph = request.app.state.graph

    lc_messages = [
        {"role": msg.role.value, "content": msg.content} for msg in body.messages
    ]

    initial_state: GraphState = {
        "messages": lc_messages,
        "user_message": user_message,
        "orchestration_result": None,
        "search_results": None,
        "rag_results": None,
        "persona_context": None,
        "plan_context": None,
        "coaching_response": None,
        "quality_rating": None,
        "final_response": None,
    }

    try:
        final_state: GraphState = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Graph execution failed: %s", exc)
        raise HTTPException(status_code=500, detail="Agent processing failed") from exc

    reply = final_state.get("final_response") or "Sorry, I could not generate a reply."

    return ChatResponse(
        reply=reply,
        orchestration_result=final_state.get("orchestration_result"),
        session_id=body.session_id,
    )
