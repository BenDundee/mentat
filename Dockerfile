# Stage 1: install dependencies with uv
# Use /app as WORKDIR so the venv path matches the runtime stage exactly.
FROM python:3.13-slim AS builder
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
# --no-install-project: install only third-party deps, not the mentat package
# itself (source isn't present at this stage). PYTHONPATH handles it at runtime.
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: minimal runtime image
FROM python:3.13-slim
WORKDIR /app
# curl is needed for the Docker healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY configs/ ./configs/
COPY frontend/ ./frontend/
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
EXPOSE 8000
CMD ["uvicorn", "mentat.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
