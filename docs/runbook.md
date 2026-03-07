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

You also need:
- An [OpenRouter](https://openrouter.ai) API key
- A [Neo4j AuraDB](https://neo4j.com/cloud/aura/) instance (free tier supported)
- A [Cohere](https://cohere.com) API key (for embeddings)

---

## First-time Setup

```bash
# 1. Clone the repo
git clone git@github.com:BenDundee/mentat.git
cd mentat

# 2. Create your .env file
cp .env.example .env
# Edit .env — required keys:
#   OPENROUTER_API_KEY=<your key>
#   NEO4J_URI=neo4j+s://<your-aura-instance>.databases.neo4j.io
#   NEO4J_USERNAME=neo4j
#   NEO4J_PASSWORD=<your password>
#   COHERE_API_KEY=<your key>

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
docker compose down -v   # WARNING: deletes all local volumes (sessions, uploads)
# Note: Neo4j data lives in AuraDB and is NOT affected by this command
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

### Re-index after schema changes

The graph schema is managed directly in Neo4j AuraDB. If you need to recreate the HNSW vector
indexes (e.g. after a data wipe), connect to your AuraDB instance and run the Cypher from
`docs/long-term-memory.md` under **Vector Index Setup**.

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

### Server isn't reachable right after `docker compose up -d`

The server connects to Neo4j AuraDB and verifies the connection on startup. If the Neo4j credentials
are missing or wrong, the server will fail to start. Check the logs:

```bash
docker compose logs -f mentat   # watch for "Mentat ready." or connection errors
curl http://localhost:8000/api/health
```

### Docker container exits immediately

```bash
docker compose logs mentat   # check the error
```

Common causes:
- `.env` file missing (Docker reads it via `env_file: .env` in compose)
- Port 8000 already in use — stop the local dev server first, or change the port mapping in `docker-compose.yml`
- Image built with an old Dockerfile — rebuild with `docker compose build --no-cache`

### Neo4j connection errors

```
neo4j.exceptions.ServiceUnavailable: ...
```

Check that `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` are set correctly in `.env`. The URI
must use the `neo4j+s://` scheme for AuraDB TLS connections. Verify your AuraDB instance is running
in the [Neo4j console](https://console.neo4j.io).

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

Required variables are marked. Copy `.env.example` to `.env` and edit as needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | API key for OpenRouter LLM access |
| `NEO4J_URI` | *(required)* | AuraDB connection URI, e.g. `neo4j+s://<id>.databases.neo4j.io` |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username (AuraDB default is `neo4j`) |
| `NEO4J_PASSWORD` | *(required)* | Neo4j password from the AuraDB console |
| `COHERE_API_KEY` | *(required)* | Cohere API key for `embed-english-v3.0` embeddings |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MENTAT_DEBUG` | `false` | Activate the Output Testing Agent (dumps pipeline state to chat) |
| `ENVIRONMENT` | `development` | `development` or `production` (informational for now) |
| `DATA_DIR` | `data` | Root directory for sessions and uploads. Set to `/app/data` in Docker. |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port (used for documentation; pass to uvicorn explicitly if changed) |

The `Settings` class (`src/mentat/core/settings.py`) validates these at startup and will raise a clear error if a required value is missing.
