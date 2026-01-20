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

# Configuration
BUCKET_NAME = os.environ.get("MODEL_BUCKET", "dtumlops-clickbait-data")


def download_model_from_gcs() -> str:
    """Download the ONNX model (and external data) from GCS bucket."""
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    local_model_path = "/tmp/clickbait_model.onnx"
    local_data_path = "/tmp/clickbait_model.onnx.data"

    # Find ONNX files in bucket
    blobs = list(bucket.list_blobs(prefix="models/"))
    onnx_blobs = [b for b in blobs if b.name.endswith(".onnx") and not b.name.endswith(".onnx.data")]
    data_blobs = [b for b in blobs if b.name.endswith(".onnx.data")]

    if not onnx_blobs:
        raise FileNotFoundError(f"No ONNX model files found in bucket '{BUCKET_NAME}'")

    # Get the most recently updated ONNX model
    latest_onnx = max(onnx_blobs, key=lambda b: b.updated)
    logger.info(f"Downloading ONNX model from GCS: {latest_onnx.name}")
    latest_onnx.download_to_filename(local_model_path)
    logger.info("ONNX model download complete")

    # Download external data file if exists
    # Match the data file to the model file
    model_basename = latest_onnx.name  # e.g., models/clickbait_model.onnx
    expected_data_name = model_basename + ".data"  # e.g., models/clickbait_model.onnx.data

    matching_data = [b for b in data_blobs if b.name == expected_data_name]
    if matching_data:
        logger.info(f"Downloading external data file: {matching_data[0].name}")
        matching_data[0].download_to_filename(local_data_path)
        logger.info("External data download complete")
    else:
        logger.info("No external data file found (single-file model)")

    return local_model_path


def find_onnx_model() -> str:
    """Find ONNX model file - local or GCS."""
    # Check environment variable first
    if model_path := os.getenv("ONNX_MODEL_PATH"):
        path = Path(model_path)
        if path.exists():
            return str(path)

    # Check common local locations
    local_paths = [
        Path("models/clickbait_model.onnx"),
        Path("/app/models/clickbait_model.onnx"),
    ]

    for path in local_paths:
        if path.exists():
            return str(path)

    # Try downloading from GCS
    logger.info("No local ONNX model found, trying GCS...")
    return download_model_from_gcs()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global onnx_session, tokenizer

    logger.info("Starting ONNX API server...")

    # Find and load ONNX model
    model_path = find_onnx_model()
    logger.info(f"Loading ONNX model from {model_path}")

    # Select providers (prefer CUDA if available)
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Using CUDA execution provider")
    else:
        providers = ["CPUExecutionProvider"]
        logger.info("Using CPU execution provider")

    onnx_session = ort.InferenceSession(model_path, providers=providers)
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
