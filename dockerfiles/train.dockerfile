FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY configs configs/
COPY src src/

# Pin Python version (onnxruntime doesn't support 3.14 yet)
RUN uv sync --frozen --no-cache --python 3.13

ENTRYPOINT ["uv", "run", "-m", "clickbait_classifier.train"]
