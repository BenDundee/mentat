> Every hammer has the innate capacity to strike a nail. Every human mind has the innate capacity for greatness.
> But not every hammer is properly used, nor is every human mind.

---

# Mentat

**An AI executive coach that actually knows you.**

The best executive coaches charge over $1,000 an hour. Mentat is a $10/month chatbot that does the same thing — not because it cuts corners, but because it's built around the same principles that make great coaching great: deep listening, long-term memory, and a relentless focus on the client's growth.

Mentat looks like a familiar chatbot. Under the hood, it's a team of specialized AI agents working in concert to build context, ensure quality, and deliver personalized coaching that compounds over time. The fact that agents can improve conversation has been proven (see https://microsoft.ai/news/the-path-to-medical-superintelligence/, or Claude Code, for example). The future of better conversational AI is clearly in orchestration and collaboration. Mentat is designed around proven coaching techniques, leveraging an agentic back end to:
  - Build a personal relationship with its clients,
  - Understand short-, mid-, and long-term goals,
  - Identify patterns in behavior over time,
  - Drive structured conversations forward,
  - Ensure high quality responses.

---

## What Mentat Does

Mentat guides users through a structured coaching relationship modeled on best practices in executive coaching:

**Onboarding** — Mentat starts by building a 360° picture of you: your role, your current pressures, your near-term and long-term goals. It conducts a structured self-assessment to understand your strengths, your recurring patterns, and what you might be avoiding. From this, it develops a shared Coaching Plan — a living document that anchors the engagement.

**Regular Sessions** — Coaching sessions follow a proven arc: check-in, agenda setting, deep work, synthesis, and accountability. Mentat doesn't just respond to what you say — it probes for what's underneath, surfaces patterns across sessions, and holds you to what you've committed to.

**Ad-Hoc Coaching** — Between sessions, Mentat is available for quick check-ins, feedback on work products, evaluating opportunities, or anything else a trusted advisor would help with.

**Long-Term Memory** — Every conversation and uploaded document is stored. Mentat remembers what you've worked on, what you've committed to, and what's changed. The value of the relationship compounds over time.

---

## The Coaching Philosophy

Mentat is informed by three major coaching frameworks:

- **GROW** *(Goal, Reality, Options, Will)* — A structured framework for problem-focused conversations that keeps sessions from wandering and pushes toward commitment.

- **Immunity to Change** *(Kegan & Lahey)* — A psychologically sophisticated lens for persistent patterns. When smart, motivated people keep failing to change something important to them, it's usually because a hidden competing commitment is working against their stated goal. Surfacing that assumption is where real change begins.

- **Positive Intelligence** *(Shirzad Chamine)* — A mental fitness framework built on neuroscience and positive psychology. Mentat can identify your "Saboteurs" — the internalized mental patterns that undermine your performance — and help you build the habits to intercept them.

---

## Architecture

Mentat's friendly chat interface sits on top of a multi-agent pipeline. When you send a message, a coordinated team of agents assembles the right context, constructs a response, and checks its own work — before you ever see a reply.

```
User Message
     │
     ▼
Orchestration Agent          ← understands your intent; decides what's needed
     │
     ├──── Search Agent       ← fetches relevant information from the web
     ├──── RAG Agent          ← retrieves relevant past conversations and documents
     │
     ▼
Context Management Agent     ← finds the needles in the haystack; builds a coaching brief
     │
     ▼
Coaching Agent               ← constructs a personalized response
     │
     ▼
Quality Agent                ← rates the response 1–5; triggers rewrites if score ≤ 3
     │
     ▼
Assistant Reply
```

After the conversation ends, a second wave of agents runs in the background: updating the user's Persona, revising the Coaching Plan, and logging action items and takeaways.

---

## The Agent Team

| Agent | Role |
|-------|------|
| **Orchestration Agent** | Classifies user intent; decides which supporting agents to invoke |
| **Search Agent** | Generates targeted queries, fetches web results, and summarizes findings |
| **RAG Agent** | Performs semantic retrieval over past conversations and uploaded documents |
| **Context Management Agent** | Ranks all available context; produces a focused coaching brief for the Coaching Agent |
| **Coaching Agent** | Constructs the final coaching response using the brief as a guide |
| **Quality Agent** | Reviews and rates the response on five dimensions; forces a rewrite if the bar isn't met |
| **Session Update Agent** | Tracks session state and onboarding phase across conversation turns |
| *(Planned)* **Persona Agent** | Maintains a deep, evolving understanding of the user across all sessions |
| *(Planned)* **Plan Management Agent** | Tracks long-term coaching plan; updates goals and logs progress |
| *(Planned)* **Client Management Agent** | Summarizes sessions; logs action items; coordinates post-session updates |

The Quality Agent evaluates responses on five dimensions: hallucination check, conversation flow, suitability for the user, adherence to the coaching plan, and consistency with the coach's voice. If a response scores 3 or below, it loops back to the Coaching Agent with specific feedback — up to three times.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.13, FastAPI, LangGraph |
| **LLM Orchestration** | LangChain, OpenRouter (Claude Sonnet / Haiku via Anthropic) |
| **Vector Store** | ChromaDB + HuggingFace `all-MiniLM-L6-v2` embeddings |
| **Data Validation** | Pydantic (frozen models throughout) |
| **Frontend** | Vanilla HTML/CSS/JS — no build tooling |
| **Package Management** | `uv` |
| **Containerization** | Docker + Docker Compose (production + dev-with-hot-reload configs) |
| **Configuration** | Per-agent YAML files; environment variables via pydantic-settings |
| **Code Quality** | ruff, pyrefly, pytest + anyio |

---

## Getting Started

### Prerequisites

- Python 3.13+
- [`uv`](https://astral.sh/uv) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- An [OpenRouter](https://openrouter.ai) API key
- Docker Desktop *(optional, for containerized deployment)*

### Quick Start

```bash
# Clone the repo
git clone git@github.com:BenDundee/mentat.git
cd mentat

# Set up your environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# Install dependencies
uv sync

# Start the server (opens http://localhost:8000)
./run.sh
```

### Docker

```bash
# Production (background service, persists on restart)
docker compose up -d

# Dev mode (hot reload, bind-mounted source)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Development Commands

```bash
./run.sh --debug         # Enable debug mode (dumps full pipeline state to chat)
./run.sh --all-tests     # Run the full test suite

uv run ruff check src/mentat/ --fix && uv run ruff format src/mentat/
uv run pyrefly check
uv run pytest
```

---

## Project Structure

```
mentat/
├── src/mentat/
│   ├── agents/          # One file per agent (BaseAgent + specialized subclasses)
│   ├── api/             # FastAPI routes and app entrypoint
│   ├── core/            # Settings (pydantic-settings), logging
│   ├── graph/           # LangGraph workflow, state definition
│   └── session/         # Session models and persistence service
├── configs/             # Per-agent YAML configs (model, prompts, LLM params)
├── frontend/            # Static HTML/CSS/JS chat interface
├── docs/                # Architecture docs, coaching philosophy, runbook
└── tests/               # Unit + integration tests (anyio, pytest)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | API key for OpenRouter LLM access |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `MENTAT_DEBUG` | `false` | Dumps full pipeline state to the chat window |
| `ENVIRONMENT` | `development` | `development` or `production` |
| `DATA_DIR` | `data` | Root directory for sessions, uploads, and ChromaDB |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

See `docs/runbook.md` for full operational documentation, troubleshooting, and Docker details.

---

## Further Reading

- [`docs/coaching-philosophy.md`](docs/coaching-philosophy.md) — The coaching methodology in detail
- [`docs/agent-design.md`](docs/agent-design.md) — Agent responsibilities and design rationale
- [`docs/agent-interactions.md`](docs/agent-interactions.md) — Pipeline topology with Mermaid diagrams
- [`docs/front-end-design.md`](docs/front-end-design.md) — Frontend specification and UX decisions
- [`docs/runbook.md`](docs/runbook.md) — Operational runbook for development and deployment
