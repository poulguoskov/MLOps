# Lightweight ONNX inference image for GCP Cloud Run
# Loads model from GCS bucket at runtime

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV UV_LINK_MODE=copy

# Install ONNX runtime dependencies (no PyTorch = smaller image)
RUN uv pip install --system --no-cache \
    onnxruntime \
    transformers \
    fastapi \
    uvicorn \
    loguru \
    pydantic \
    google-cloud-storage

# Copy only what's needed for inference
COPY src/clickbait_classifier/api_onnx.py ./clickbait_classifier/api_onnx.py

# Create __init__.py
RUN touch clickbait_classifier/__init__.py

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn clickbait_classifier.api_onnx:app --host 0.0.0.0 --port ${PORT:-8080}"]
