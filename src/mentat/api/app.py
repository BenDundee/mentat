"""FastAPI application factory."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mentat.api.routes import router
from mentat.core.logging import get_logger, setup_logging
from mentat.core.settings import settings
from mentat.graph.workflow import compile_graph

load_dotenv()
setup_logging()

logger = get_logger(__name__)

_CONSOLIDATION_INTERVAL_SECONDS = 30 * 60  # 30 minutes


async def _consolidation_loop(agent) -> None:  # type: ignore[no-untyped-def]
    """Background task: run ConsolidationAgent every 30 minutes."""
    while True:
        await asyncio.sleep(_CONSOLIDATION_INTERVAL_SECONDS)
        try:
            await agent.run_once()
        except Exception as exc:
            logger.warning("Consolidation loop error (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize services and compile the agent graph once at startup."""
    from mentat.agents.consolidation import ConsolidationAgent
    from mentat.agents.ingest import IngestAgent
    from mentat.core.embedding_service import EmbeddingService
    from mentat.core.neo4j_service import Neo4jService

    debug = settings.mentat_debug
    if debug:
        logger.info("Starting Mentat — debug mode (Output Testing Agent active)...")
    else:
        logger.info("Starting Mentat — initializing Neo4j and agent graph...")

    embedding_service = EmbeddingService()
    neo4j_service = Neo4jService()
    await neo4j_service.create_indexes()
    await neo4j_service.validate_embedding_model(
        model=embedding_service.model,
        dims=embedding_service.dims,
    )

    ingest_agent = IngestAgent(
        neo4j_service=neo4j_service,
        embedding_service=embedding_service,
    )
    consolidation_agent = ConsolidationAgent(
        neo4j_service=neo4j_service,
        embedding_service=embedding_service,
    )

    app.state.neo4j_service = neo4j_service
    app.state.ingest_agent = ingest_agent
    app.state.consolidation_agent = consolidation_agent
    app.state.graph = compile_graph(
        neo4j_service=neo4j_service,
        embedding_service=embedding_service,
        debug=debug,
    )

    # Start background consolidation loop
    consolidation_task = asyncio.create_task(_consolidation_loop(consolidation_agent))

    logger.info("Mentat ready.")
    yield

    # Cleanup
    consolidation_task.cancel()
    try:
        await consolidation_task
    except asyncio.CancelledError:
        pass
    await neo4j_service.close()
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
