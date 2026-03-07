#!/bin/bash
# run.sh — start Mentat or run the test suite.
#
# Usage:
#   ./run.sh                 Start the development server (default)
#   ./run.sh --debug         Start the server with the Output Testing Agent
#   ./run.sh --all-tests     Run the full pytest suite
#   ./run.sh --cleanup       Delete logs, local data files, and Neo4j graph data
#   ./run.sh -h | --help     Show this help and exit
#
# The server requires a .env file with OPENROUTER_API_KEY set.
# See .env.example for a template.
set -e

# ── Argument parsing ────────────────────────────────────────────────────────
RUN_TESTS=false
DEBUG=false
CLEANUP=false

usage() {
    cat <<EOF
Usage: ./run.sh [OPTIONS]

Options:
  --debug        Start the server using the Output Testing Agent, which dumps
                 the full pipeline state to the chat window instead of a normal
                 response. Useful for inspecting agent output during development.
  --all-tests    Run the full pytest test suite instead of starting the server
  --cleanup      DEV ONLY: delete all log files and local data (sessions,
                 uploads, blobs), then wipe all nodes from the Neo4j graph.
                 Prompts for confirmation before making any changes.
  -h, --help     Show this help message and exit

Examples:
  ./run.sh               Start the Mentat server at http://localhost:8000
  ./run.sh --debug        Start the server in debug mode (state dump responses)
  ./run.sh --all-tests   Run all unit and graph tests (integration tests
                         require OPENROUTER_API_KEY and are skipped otherwise)
  ./run.sh --cleanup     Wipe dev state (logs, data files, Neo4j graph)
EOF
}

for arg in "$@"; do
    case "$arg" in
        --debug)
            DEBUG=true
            ;;
        --all-tests)
            RUN_TESTS=true
            ;;
        --cleanup)
            CLEANUP=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
done

# ── Cleanup ─────────────────────────────────────────────────────────────────
if [ "$CLEANUP" = true ]; then
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  DEV CLEANUP — this will permanently delete:                 │"
    echo "│    • log/*.log                                               │"
    echo "│    • data/sessions/*                                         │"
    echo "│    • data/uploads/*                                          │"
    echo "│    • data/blobs/*                                            │"
    echo "│    • data/chroma/*                                           │"
    echo "│    • ALL nodes and relationships in the Neo4j database       │"
    echo "└──────────────────────────────────────────────────────────────┘"
    printf "Continue? [y/N] "
    read -r confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Aborted."
        exit 0
    fi

    echo ""
    echo "Cleaning up log files..."
    find log/ -name "*.log" -not -name ".gitkeep" -delete 2>/dev/null && echo "  ✓ log/*.log deleted" || true

    echo "Cleaning up local data directories..."
    for dir in data/sessions data/uploads data/blobs data/chroma; do
        if [ -d "$dir" ]; then
            find "$dir" -mindepth 1 -delete 2>/dev/null && echo "  ✓ $dir cleared" || true
        fi
    done

    echo "Clearing Neo4j graph..."
    if [ -f .env ]; then
        uv run python -c "
import asyncio, os
from dotenv import load_dotenv
load_dotenv()

uri      = os.environ.get('NEO4J_URI', '')
username = os.environ.get('NEO4J_USERNAME', 'neo4j')
password = os.environ.get('NEO4J_PASSWORD', '')

if not uri or not password:
    print('  ! NEO4J_URI / NEO4J_PASSWORD not set in .env — skipping Neo4j cleanup.')
else:
    from neo4j import AsyncGraphDatabase
    async def wipe():
        driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
        try:
            async with driver.session() as session:
                result = await session.run('MATCH (n) DETACH DELETE n')
                summary = await result.consume()
                deleted = summary.counters.nodes_deleted
                print(f'  ✓ Neo4j: {deleted} node(s) deleted.')
            async with driver.session() as session:
                await session.run(\"DROP INDEX \`chunk-embeddings\` IF EXISTS\")
                await session.run(\"DROP INDEX \`memory-embeddings\` IF EXISTS\")
                await session.run(\"DROP INDEX \`embedding-config\` IF EXISTS\")
                print('  ✓ Neo4j: vector indexes dropped (will be recreated on next start).')
        finally:
            await driver.close()
    asyncio.run(wipe())
"
    else
        echo "  ! No .env file found — skipping Neo4j cleanup."
    fi

    echo ""
    echo "Cleanup complete."
    exit 0
fi

# ── Run tests ───────────────────────────────────────────────────────────────
if [ "$RUN_TESTS" = true ]; then
    mkdir -p log
    LOG_FILE="log/tests_$(date +%Y%m%d_%H%M%S).log"
    echo "Running test suite... (output also written to $LOG_FILE)"
    uv run pytest tests/ -v 2>&1 | tee "$LOG_FILE"
    exit "${PIPESTATUS[0]}"
fi

# ── Check for .env ─────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "ERROR: .env file not found."
    echo "Copy .env.example and fill in your API key:"
    echo "  cp .env.example .env"
    exit 1
fi

if ! grep -q "OPENROUTER_API_KEY=." .env 2>/dev/null; then
    echo "ERROR: OPENROUTER_API_KEY is not set in .env"
    exit 1
fi

# ── Open browser after a short delay ───────────────────────────────────────
(sleep 2 && open http://localhost:8000) &

# ── Start server ────────────────────────────────────────────────────────────
if [ "$DEBUG" = true ]; then
    echo "Starting Mentat at http://localhost:8000 (debug mode — Output Testing Agent active) ..."
    MENTAT_DEBUG=1 LOG_LEVEL=DEBUG uv run uvicorn mentat.api.app:app --reload --host 0.0.0.0 --port 8000
else
    echo "Starting Mentat at http://localhost:8000 ..."
    LOG_LEVEL=DEBUG uv run uvicorn mentat.api.app:app --reload --host 0.0.0.0 --port 8000
fi
