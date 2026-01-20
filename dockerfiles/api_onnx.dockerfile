# Lightweight ONNX inference image
# Build with: docker build --build-arg CUDA=true for GPU support

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV UV_LINK_MODE=copy

# Only install ONNX runtime dependencies (no PyTorch)
# This keeps the image small
ARG CUDA=false

RUN if [ "$CUDA" = "true" ]; then \
        uv pip install --system --no-cache \
            onnxruntime-gpu \
            transformers \
            fastapi \
            uvicorn \
            loguru \
            pydantic; \
    else \
        uv pip install --system --no-cache \
            onnxruntime \
            transformers \
            fastapi \
            uvicorn \
            loguru \
            pydantic; \
    fi

# Copy only what's needed for inference
COPY src/clickbait_classifier/api_onnx.py ./clickbait_classifier/api_onnx.py

# Create __init__.py
RUN touch clickbait_classifier/__init__.py

# Set environment
ENV ONNX_MODEL_PATH=/app/models/clickbait_model.onnx

EXPOSE 8000

# Use dynamic PORT for Cloud Run compatibility
CMD ["sh", "-c", "uvicorn clickbait_classifier.api_onnx:app --host 0.0.0.0 --port ${PORT:-8080}"]
