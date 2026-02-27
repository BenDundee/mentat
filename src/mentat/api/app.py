"""FastAPI application factory."""

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
    """Compile the agent graph once at startup."""
    logger.info("Starting Mentat — compiling agent graph...")
    app.state.graph = compile_graph()
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
