FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app/src
ENV UV_LINK_MODE=copy

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md

COPY src/ src/
COPY configs/ configs/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn clickbait_classifier.api_gcp:app --host 0.0.0.0 --port ${PORT:-8000}"]
