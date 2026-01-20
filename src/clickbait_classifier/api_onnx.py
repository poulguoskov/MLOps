"""FastAPI application serving clickbait classifier using ONNX runtime."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from transformers import DistilBertTokenizer


# Pydantic models
class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    text: str
    is_clickbait: bool
    confidence: float


class BatchClassifyRequest(BaseModel):
    texts: list[str]


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


# Global variables
onnx_session: ort.InferenceSession | None = None
tokenizer: DistilBertTokenizer | None = None


def find_onnx_model() -> Path | None:
    """Find ONNX model file."""
    # Check environment variable first
    if model_path := os.getenv("ONNX_MODEL_PATH"):
        path = Path(model_path)
        if path.exists():
            return path

    # Check common locations
    search_paths = [
        Path("models/clickbait_model.onnx"),
        Path("/app/models/clickbait_model.onnx"),
        Path("/gcs/dtumlops-clickbait-data/models/clickbait_model.onnx"),
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global onnx_session, tokenizer

    logger.info("Starting ONNX API server...")

    # Find and load ONNX model
    model_path = find_onnx_model()
    if model_path is None:
        raise RuntimeError("No ONNX model found. Set ONNX_MODEL_PATH or place model in models/")

    logger.info(f"Loading ONNX model from {model_path}")

    # Select providers (prefer CUDA if available)
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Using CUDA execution provider")
    else:
        providers = ["CPUExecutionProvider"]
        logger.info("Using CPU execution provider")

    onnx_session = ort.InferenceSession(str(model_path), providers=providers)
    logger.info(f"ONNX model loaded. Providers: {onnx_session.get_providers()}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    logger.info("Tokenizer loaded")

    yield

    logger.info("Shutting down ONNX API server...")


app = FastAPI(
    title="Clickbait Classifier API (ONNX)",
    description="Lightweight ONNX-based clickbait detection API",
    version="1.0.0",
    lifespan=lifespan,
)


def predict(text: str) -> ClassifyResponse:
    """Run prediction using ONNX model."""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    # Run inference
    outputs = onnx_session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        },
    )

    logits = outputs[0][0]
    probabilities = np.exp(logits) / np.exp(logits).sum()  # softmax
    predicted_class = int(np.argmax(logits))
    confidence = float(probabilities[predicted_class])

    return ClassifyResponse(
        text=text,
        is_clickbait=bool(predicted_class == 1),
        confidence=confidence,
    )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Clickbait Classifier API (ONNX)",
        "status": "healthy",
        "runtime": "onnxruntime",
        "providers": onnx_session.get_providers() if onnx_session else [],
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """Classify a single text."""
    return predict(request.text)


@app.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(request: BatchClassifyRequest):
    """Classify multiple texts."""
    results = [predict(text) for text in request.texts]
    return BatchClassifyResponse(results=results)
