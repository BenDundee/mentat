#!/bin/bash
set -e

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

# ── Start server ───────────────────────────────────────────────────────────
echo "Starting Mentat at http://localhost:8000 ..."
uv run uvicorn mentat.api.app:app --reload --host 0.0.0.0 --port 8000
