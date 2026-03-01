# Mentat Runbook

Operational guide for running and maintaining Mentat. Keep this file current as the deployment model evolves.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [First-time Setup](#first-time-setup)
3. [Local Development (no Docker)](#local-development-no-docker)
4. [Docker — Personal Background Use](#docker--personal-background-use)
5. [Docker — Dev Mode with Hot Reload](#docker--dev-mode-with-hot-reload)
6. [Common Operations](#common-operations)
7. [Troubleshooting](#troubleshooting)
8. [Environment Variables Reference](#environment-variables-reference)

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.13+ | `brew install python` or [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker Desktop | latest | [docker.com](https://www.docker.com/products/docker-desktop) |
| Git | any | `brew install git` |

You also need an [OpenRouter](https://openrouter.ai) API key.

---

## First-time Setup

```bash
# 1. Clone the repo
git clone git@github.com:BenDundee/mentat.git
cd mentat

# 2. Create your .env file
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=<your key>

# 3. Install Python dependencies (creates .venv automatically)
uv sync
```

The `.env` file is gitignored — never commit it. See [Environment Variables Reference](#environment-variables-reference) for all available settings.

---

## Local Development (no Docker)

Use this for day-to-day development. The server reloads on code changes.

### Start the server

```bash
./run.sh
```

Opens `http://localhost:8000` in your browser automatically.

### Debug mode

Activates the Output Testing Agent, which dumps the full pipeline state into the chat window instead of a normal coaching response. Useful when debugging agent wiring.

```bash
./run.sh --debug
```

### Run the test suite

```bash
./run.sh --all-tests
# or directly:
uv run pytest
```

Integration tests (requiring a live OpenRouter API key) are skipped automatically unless `OPENROUTER_API_KEY` is set in the environment.

### Code quality checks

Run these before pushing:

```bash
uv run ruff check src/mentat/ --fix && uv run ruff format src/mentat/
uv run pyrefly check
uv run pytest
```

---

## Docker — Personal Background Use

Use this to run Mentat as a persistent background service (starts on boot, survives terminal sessions).

### Start

```bash
docker compose up -d
```

The first run builds the image automatically. Subsequent starts reuse the cached image.

### Verify it's healthy

```bash
curl http://localhost:8000/api/health
# → {"status":"ok","version":"0.1.0"}

docker compose ps          # check status
docker compose logs -f     # tail logs
```

### Stop

```bash
docker compose down
```

Data is preserved in named Docker volumes (`mentat_data`, `mentat_logs`). `down` does not delete volumes.

### Rebuild the image after code changes

```bash
docker compose build
docker compose up -d
```

Or in one step:

```bash
docker compose up -d --build
```

### Wipe everything (including data)

```bash
docker compose down -v   # WARNING: deletes all volumes (conversations, uploads, ChromaDB)
```

---

## Docker — Dev Mode with Hot Reload

Use this when you want Docker's isolated environment but still need live code reloading during development.

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

The dev override mounts `src/`, `configs/`, and `frontend/` as bind mounts, so edits on the host are immediately reflected inside the container. `data/` is also bind-mounted to your local `./data/` directory so session data is easily inspectable.

Uvicorn runs with `--reload` in this mode. Stop with `Ctrl+C`.

---

## Common Operations

### View logs

```bash
# Docker logs (stdout/stderr)
docker compose logs -f

# Application log file (inside the volume — must shell into the container)
docker compose exec mentat cat /app/log/mentat.log

# Local dev logs
tail -f log/mentat.log
```

### Shell into the running container

```bash
docker compose exec mentat bash
```

### Inspect persisted data

In production mode, data lives in Docker named volumes. To inspect:

```bash
# List what's in the sessions directory
docker compose exec mentat ls /app/data/sessions/

# View a specific session file
docker compose exec mentat cat /app/data/sessions/<session-id>.json
```

In dev mode, `./data/` on your host is mounted directly — just browse it normally.

### Rebuild the ChromaDB vector store

Required after changing the embedding model in `configs/rag.yml`:

```bash
# Local dev
uv run python -m mentat.tools.rebuild_store

# Docker
docker compose exec mentat python -m mentat.tools.rebuild_store
```

### Update dependencies

```bash
uv add <package>                   # add a new runtime dependency
uv add --dev <package>             # add a dev dependency
uv sync                            # sync .venv after pyproject.toml changes
docker compose build               # rebuild the Docker image to pick up changes
```

---

## Troubleshooting

### Server won't start — missing API key

```
pydantic_settings.exceptions.SettingsError: validation error for Settings
openrouter_api_key: Field required
```

Your `.env` file is missing or `OPENROUTER_API_KEY` is not set. Check:

```bash
grep OPENROUTER_API_KEY .env
```

### Docker container exits immediately

```bash
docker compose logs mentat   # check the error
```

Common causes:
- `.env` file missing (Docker reads it via `env_file: .env` in compose)
- Port 8000 already in use — stop the local dev server first, or change the port mapping in `docker-compose.yml`

### ChromaDB embedding model mismatch

```
EmbeddingModelMismatchError: ...
```

The embedding model in `configs/rag.yml` changed after data was written. Rebuild the store (see above). This deletes all stored conversation history and uploaded documents.

### Hot reload not triggering in dev mode

Ensure you're using the dev compose override:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

If using plain `docker compose up`, source files are not bind-mounted and changes require a rebuild.

### Tests fail with import errors

Ensure your virtual environment is up to date:

```bash
uv sync
uv run pytest
```

---

## Environment Variables Reference

All variables are optional except `OPENROUTER_API_KEY`. Copy `.env.example` to `.env` and edit as needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | API key for OpenRouter LLM access |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MENTAT_DEBUG` | `false` | Activate the Output Testing Agent (dumps pipeline state to chat) |
| `ENVIRONMENT` | `development` | `development` or `production` (informational for now) |
| `DATA_DIR` | `data` | Root directory for sessions, uploads, and ChromaDB. Set to `/app/data` in Docker. |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port (used for documentation; pass to uvicorn explicitly if changed) |

The `Settings` class (`src/mentat/core/settings.py`) validates these at startup and will raise a clear error if a required value is missing.
