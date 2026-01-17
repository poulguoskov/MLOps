FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app/src
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

EXPOSE 8501

ENTRYPOINT ["uv", "run", "streamlit", "run", "src/clickbait_classifier/frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
