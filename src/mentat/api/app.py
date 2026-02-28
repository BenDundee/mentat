"""FastAPI application factory."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mentat.api.routes import router
from mentat.core.logging import get_logger, setup_logging
from mentat.graph.workflow import compile_graph

load_dotenv()
setup_logging()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize services and compile the agent graph once at startup."""
    from mentat.core.config import load_agent_config
    from mentat.core.vector_store import VectorStoreService

    debug = os.environ.get("MENTAT_DEBUG", "").lower() in ("1", "true", "yes")
    if debug:
        logger.info("Starting Mentat — debug mode (Output Testing Agent active)...")
    else:
        logger.info("Starting Mentat — initializing vector store and agent graph...")

    rag_config = load_agent_config("rag")
    extra = rag_config.extra_config
    vector_store = VectorStoreService(
        embedding_model=extra["embedding_model"],
        persist_path=extra["persist_path"],
        collection_conversations=extra["collection_conversations"],
        collection_documents=extra["collection_documents"],
        meta_key=extra["meta_key"],
    )
    app.state.vector_store = vector_store
    app.state.graph = compile_graph(vector_store=vector_store, debug=debug)
    logger.info("Mentat ready.")
    yield
    logger.info("Mentat shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Mentat",
        description="AI executive coaching chatbot",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")

    # Serve frontend at root
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

    return app


app = create_app()
