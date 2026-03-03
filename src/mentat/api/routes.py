"""API route handlers."""

import json
import re
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mentat.api.schemas import ChatRequest, ChatResponse, DocumentUploadResponse
from mentat.core.logging import get_logger
from mentat.core.vector_store import utc_now_iso
from mentat.graph.state import GraphState
from mentat.session.service import SessionService

logger = get_logger(__name__)
_session_service = SessionService()

router = APIRouter()

_UPLOAD_DIR = Path("data/uploads")

_NODE_STATUS: dict[str, str] = {
    "orchestration": "Understanding your message\u2026",
    "search": "Searching the web\u2026",
    "rag": "Reviewing our past conversations\u2026",
    "context_management": "Synthesizing context\u2026",
    "coaching": "Crafting a response\u2026",
    "quality": "Reviewing the response\u2026",
    "format_response": "Preparing your answer\u2026",
    "session_update": "Saving your session\u2026",
}
_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def _sanitize_filename(name: str) -> str:
    """Replace unsafe characters in a filename with underscores."""
    return re.sub(r"[^\w.\-]", "_", name)


def _build_think_content(state: dict) -> str | None:  # type: ignore[type-arg]
    """Build a markdown summary of coaching_brief and session_state for think blocks."""
    parts: list[str] = []

    cm = state.get("context_management_result")
    if cm is not None:
        parts.append(f"## Coaching Brief\n{cm.coaching_brief}")
        parts.append(f"**Phase:** {cm.session_phase}  |  **Tone:** {cm.tone_guidance}")
        if cm.key_information:
            parts.append(f"**Key Information:**\n{cm.key_information}")

    session = state.get("session_state")
    if session is not None:
        parts.append(
            f"## Session State\n"
            f"**Type:** {session.conversation_type}  |  "
            f"**Phase:** {session.phase}  |  **Turn:** {session.turn_count}"
        )
        if session.scratchpad:
            parts.append(f"**Scratchpad:**\n{session.scratchpad}")

    return "\n\n".join(parts) if parts else None


def _extract_text(raw_bytes: bytes, suffix: str) -> str:
    """Extract plain text from file bytes based on file extension.

    Args:
        raw_bytes: Raw file content.
        suffix: Lowercase file extension including the dot (e.g. ".pdf").

    Returns:
        Extracted text string.
    """
    if suffix == ".pdf":
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        import io

        from docx import Document

        doc = Document(io.BytesIO(raw_bytes))
        return "\n".join(para.text for para in doc.paragraphs)
    # Plain text fallback
    return raw_bytes.decode("utf-8", errors="replace")


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Simple liveness check used by Docker healthcheck and load balancers."""
    return {"status": "ok", "version": "0.1.0"}


@router.post("/chat", response_model=ChatResponse)
async def handle_chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Process a chat message through the Mentat agent graph.

    The client sends the full conversation history on every request.
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

    # Load or create session state (best-effort — None degrades gracefully)
    session_state = None
    if body.session_id:
        try:
            session_state = _session_service.load_or_create(body.session_id)
        except Exception as exc:
            logger.warning("Failed to load session %s: %s", body.session_id, exc)

    initial_state: GraphState = {
        "messages": lc_messages,
        "user_message": user_message,
        "orchestration_result": None,
        "search_results": None,
        "rag_results": None,
        "context_management_result": None,
        "persona_context": None,
        "plan_context": None,
        "coaching_response": None,
        "quality_rating": None,
        "quality_feedback": None,
        "coaching_attempts": None,
        "final_response": None,
        "session_state": session_state,
    }

    try:
        final_state: GraphState = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Graph execution failed: %s", exc)
        raise HTTPException(status_code=500, detail="Agent processing failed") from exc

    # Persist the updated session (best-effort)
    updated_session = final_state.get("session_state")
    if updated_session is not None:
        try:
            _session_service.save(updated_session)
        except Exception as exc:
            logger.warning("Failed to save session %s: %s", body.session_id, exc)

    # Store this conversation turn for future RAG retrieval (best-effort)
    try:
        vector_store = getattr(request.app.state, "vector_store", None)
        final_response = final_state.get("final_response")
        if vector_store is not None and body.session_id and final_response:
            orch_result = final_state.get("orchestration_result")
            intent_str = orch_result.intent.value if orch_result else "unknown"
            turn = f"User: {user_message}\nAssistant: {final_response}"
            vector_store.add_conversation(
                turn,
                {
                    "session_id": body.session_id,
                    "timestamp": utc_now_iso(),
                    "intent": intent_str,
                },
            )
    except Exception as exc:
        logger.warning("Failed to store conversation turn: %s", exc)

    reply = final_state.get("final_response") or "Sorry, I could not generate a reply."

    return ChatResponse(
        reply=reply,
        orchestration_result=final_state.get("orchestration_result"),
        session_id=body.session_id,
    )


@router.post("/chat/stream")
async def handle_chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    """Stream agent status events and the final reply as SSE.

    Emits ``data: {...}\\n\\n`` events with these shapes:
    - ``{"type": "status", "message": "<phrase>"}`` — one per agent node
    - ``{"type": "reply",  "content": "<text>"}``   — final assistant reply
    - ``{"type": "done"}``                           — stream complete
    """
    if not body.messages:
        raise HTTPException(status_code=422, detail="messages list cannot be empty")

    last_message = body.messages[-1]
    user_message = last_message.content

    logger.info("Received SSE chat request. session_id=%s", body.session_id)

    graph = request.app.state.graph
    vector_store = getattr(request.app.state, "vector_store", None)

    lc_messages = [
        {"role": msg.role.value, "content": msg.content} for msg in body.messages
    ]

    session_state = None
    if body.session_id:
        try:
            session_state = _session_service.load_or_create(body.session_id)
        except Exception as exc:
            logger.warning("Failed to load session %s: %s", body.session_id, exc)

    initial_state: GraphState = {
        "messages": lc_messages,
        "user_message": user_message,
        "orchestration_result": None,
        "search_results": None,
        "rag_results": None,
        "context_management_result": None,
        "persona_context": None,
        "plan_context": None,
        "coaching_response": None,
        "quality_rating": None,
        "quality_feedback": None,
        "coaching_attempts": None,
        "final_response": None,
        "session_state": session_state,
    }

    async def _generate() -> AsyncGenerator[str, None]:
        final_state: dict | None = None  # pyrefly: ignore[bad-assignment]
        seen_nodes: set[str] = set()

        try:
            # pyrefly: ignore[bad-argument-type]
            async for event in graph.astream_events(initial_state, version="v2"):
                event_type: str = event["event"]
                node: str | None = event.get("metadata", {}).get("langgraph_node")

                if (
                    event_type == "on_chain_start"
                    and node is not None
                    and node in _NODE_STATUS
                    and node not in seen_nodes
                ):
                    seen_nodes.add(node)
                    payload = json.dumps(
                        {"type": "status", "message": _NODE_STATUS[node]}
                    )
                    yield f"data: {payload}\n\n"

                elif event_type == "on_chain_end":
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict) and "final_response" in output:
                        final_state = output

        except Exception as exc:
            logger.exception("SSE graph execution failed: %s", exc)
            error_payload = json.dumps(
                {"type": "reply", "content": "Sorry, an error occurred."}
            )
            yield f"data: {error_payload}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Post-stream: persist session and store conversation turn
        if final_state is not None:
            updated_session = final_state.get("session_state")
            if updated_session is not None:
                try:
                    _session_service.save(updated_session)
                except Exception as exc:
                    logger.warning(
                        "Failed to save session %s: %s", body.session_id, exc
                    )
            try:
                final_response = final_state.get("final_response")
                if vector_store is not None and body.session_id and final_response:
                    orch_result = final_state.get("orchestration_result")
                    intent_str = orch_result.intent.value if orch_result else "unknown"
                    turn = f"User: {user_message}\nAssistant: {final_response}"
                    vector_store.add_conversation(
                        turn,
                        {
                            "session_id": body.session_id,
                            "timestamp": utc_now_iso(),
                            "intent": intent_str,
                        },
                    )
            except Exception as exc:
                logger.warning("Failed to store conversation turn: %s", exc)

        if final_state is not None:
            think_content = _build_think_content(final_state)
            if think_content:
                payload = json.dumps({"type": "think", "content": think_content})
                yield f"data: {payload}\n\n"

        reply = (
            final_state.get("final_response") if final_state else None
        ) or "Sorry, I could not generate a reply."
        yield f"data: {json.dumps({'type': 'reply', 'content': reply})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """Upload a document and store its text chunks in the vector store.

    Accepted formats: .pdf, .txt, .docx
    """
    if file.filename is None:
        raise HTTPException(status_code=422, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: {_ALLOWED_EXTENSIONS}",
        )

    upload_id = str(uuid.uuid4())
    safe_name = _sanitize_filename(file.filename)
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = _UPLOAD_DIR / f"{upload_id}_{safe_name}"

    raw_bytes = await file.read()

    # Persist original file before any processing
    dest_path.write_bytes(raw_bytes)
    logger.info("Persisted upload to %s", dest_path)

    text = _extract_text(raw_bytes, suffix)
    chunks = _SPLITTER.split_text(text)

    vector_store = request.app.state.vector_store
    metadatas = [
        {
            "filename": file.filename,
            "upload_id": upload_id,
            "chunk_index": str(i),
            "uploaded_at": utc_now_iso(),
            "file_path": str(dest_path),
        }
        for i in range(len(chunks))
    ]

    ids = vector_store.add_documents(chunks, metadatas)

    logger.info(
        "Stored %d chunks for upload_id=%s filename=%s",
        len(ids),
        upload_id,
        file.filename,
    )

    return DocumentUploadResponse(
        filename=file.filename,
        chunks_stored=len(ids),
        document_ids=tuple(ids),
        file_path=str(dest_path),
    )
