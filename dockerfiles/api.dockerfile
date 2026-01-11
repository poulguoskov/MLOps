FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app/src
ENV UV_LINK_MODE=copy

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

COPY src/ src/

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "clickbait_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]