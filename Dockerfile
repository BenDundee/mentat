# Stage 1: install dependencies with uv
FROM python:3.13-slim AS builder
WORKDIR /build
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 2: minimal runtime image
FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /build/.venv /app/.venv
COPY src/ ./src/
COPY configs/ ./configs/
COPY frontend/ ./frontend/
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
EXPOSE 8000
CMD ["uvicorn", "mentat.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
