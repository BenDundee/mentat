"""API route handlers."""

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mentat.api.schemas import ChatRequest, ChatResponse, DocumentUploadResponse
from mentat.core.logging import get_logger
from mentat.core.vector_store import utc_now_iso
from mentat.graph.state import GraphState

logger = get_logger(__name__)

router = APIRouter()

_UPLOAD_DIR = Path("data/uploads")
_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def _sanitize_filename(name: str) -> str:
    """Replace unsafe characters in a filename with underscores."""
    return re.sub(r"[^\w.\-]", "_", name)


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
    """Simple liveness check."""
    return {"status": "ok"}


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
        "final_response": None,
    }

    try:
        final_state: GraphState = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Graph execution failed: %s", exc)
        raise HTTPException(status_code=500, detail="Agent processing failed") from exc

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
