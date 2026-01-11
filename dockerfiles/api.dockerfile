FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

RUN uv sync --frozen --no-install-project

COPY src/ src/

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "src.clickbait_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]