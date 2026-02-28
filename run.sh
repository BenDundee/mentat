#!/bin/bash
# run.sh — start Mentat or run the test suite.
#
# Usage:
#   ./run.sh                 Start the development server (default)
#   ./run.sh --debug         Start the server with the Output Testing Agent
#   ./run.sh --all-tests     Run the full pytest suite
#   ./run.sh -h | --help     Show this help and exit
#
# The server requires a .env file with OPENROUTER_API_KEY set.
# See .env.example for a template.
set -e

# ── Argument parsing ────────────────────────────────────────────────────────
RUN_TESTS=false
DEBUG=false

usage() {
    cat <<EOF
Usage: ./run.sh [OPTIONS]

Options:
  --debug        Start the server using the Output Testing Agent, which dumps
                 the full pipeline state to the chat window instead of a normal
                 response. Useful for inspecting agent output during development.
  --all-tests    Run the full pytest test suite instead of starting the server
  -h, --help     Show this help message and exit

Examples:
  ./run.sh               Start the Mentat server at http://localhost:8000
  ./run.sh --debug        Start the server in debug mode (state dump responses)
  ./run.sh --all-tests   Run all unit and graph tests (integration tests
                         require OPENROUTER_API_KEY and are skipped otherwise)
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
    MENTAT_DEBUG=1 uv run uvicorn mentat.api.app:app --reload --host 0.0.0.0 --port 8000
else
    echo "Starting Mentat at http://localhost:8000 ..."
    uv run uvicorn mentat.api.app:app --reload --host 0.0.0.0 --port 8000
fi
