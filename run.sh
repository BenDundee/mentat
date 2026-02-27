#!/bin/bash
# run.sh — start Mentat or run the test suite.
#
# Usage:
#   ./run.sh                 Start the development server (default)
#   ./run.sh --all-tests     Run the full pytest suite
#   ./run.sh -h | --help     Show this help and exit
#
# The server requires a .env file with OPENROUTER_API_KEY set.
# See .env.example for a template.
set -e

# ── Argument parsing ────────────────────────────────────────────────────────
RUN_TESTS=false

usage() {
    cat <<EOF
Usage: ./run.sh [OPTIONS]

Options:
  --all-tests    Run the full pytest test suite instead of starting the server
  -h, --help     Show this help message and exit

Examples:
  ./run.sh               Start the Mentat server at http://localhost:8000
  ./run.sh --all-tests   Run all unit and graph tests (integration tests
                         require OPENROUTER_API_KEY and are skipped otherwise)
EOF
}

for arg in "$@"; do
    case "$arg" in
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
    echo "Running test suite..."
    uv run pytest tests/ -v
    exit 0
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
echo "Starting Mentat at http://localhost:8000 ..."
uv run uvicorn mentat.api.app:app --reload --host 0.0.0.0 --port 8000
