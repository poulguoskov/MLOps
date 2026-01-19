"""Clickbait Classifier API for GCP Cloud Run - Downloads model from GCS."""

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel
from transformers import AutoTokenizer

from clickbait_classifier.model import ClickbaitClassifier


# --- Request/Response Models ---
class TextInput(BaseModel):
    """Request body for single text classification."""

    text: str
    max_length: int = 128


class ClassificationResult(BaseModel):
    """Response for classification."""

    text: str
    is_clickbait: bool
    confidence: float


class BatchTextInput(BaseModel):
    """Request body for batch text classification."""

    texts: list[str]
    max_length: int = 128


class BatchClassificationResult(BaseModel):
    """Response for batch classification."""

    results: list[ClassificationResult]


# --- Configuration ---
BUCKET_NAME = os.environ.get("MODEL_BUCKET", "dtumlops-clickbait-data")
MODEL_NAME = "distilbert-base-uncased"

# --- App setup ---
model = None
tokenizer = None


def download_model_from_gcs() -> str:
    """Download the latest model checkpoint from GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    local_model_path = "/tmp/model.ckpt"

    # Find all checkpoint files in bucket
    blobs = list(bucket.list_blobs(prefix="models/"))
    ckpt_blobs = [b for b in blobs if b.name.endswith(".ckpt") or b.name.endswith(".pt")]

    if not ckpt_blobs:
        raise FileNotFoundError(f"No model checkpoint files found in bucket '{BUCKET_NAME}'")

    # Get the most recently updated checkpoint
    latest_blob = max(ckpt_blobs, key=lambda b: b.updated)
    print(f"Downloading model from GCS: {latest_blob.name}")

    latest_blob.download_to_filename(local_model_path)
    print("Model download complete")

    return local_model_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup from GCS, cleanup on shutdown."""
    global model, tokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ClickbaitClassifier(model_name=MODEL_NAME)

    # Download model from GCS
    checkpoint_path = download_model_from_gcs()
    print(f"Loading weights from: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        weights = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
        model.load_state_dict(weights)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    print("Model loaded successfully")

    yield
    print("Shutting down API")


app = FastAPI(
    title="Clickbait Classifier API",
    description="Classify headlines as clickbait or not (Cloud Run deployment)",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Endpoints ---
@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Welcome to the Clickbait Classifier API!", "status": "healthy"}


@app.post("/classify", response_model=ClassificationResult)
def classify_text(input_data: TextInput):
    """Classify a single text as clickbait or not."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = tokenizer(
        input_data.text,
        return_tensors="pt",
        truncation=True,
        max_length=input_data.max_length,
    )

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        prediction = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1)[0].tolist()

    return ClassificationResult(
        text=input_data.text,
        is_clickbait=bool(prediction),
        confidence=probabilities[prediction],
    )


@app.post("/classify/batch", response_model=BatchClassificationResult)
def classify_batch(input_data: BatchTextInput):
    """Classify multiple texts at once (more efficient for bulk processing)."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = tokenizer(
        input_data.texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=input_data.max_length,
    )

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        predictions = torch.argmax(logits, dim=1).tolist()
        probabilities = torch.softmax(logits, dim=1).tolist()

    results = [
        ClassificationResult(
            text=text,
            is_clickbait=bool(pred),
            confidence=probs[pred],
        )
        for text, pred, probs in zip(input_data.texts, predictions, probabilities)
    ]

    return BatchClassificationResult(results=results)


# --- Backward compatibility endpoints ---
@app.post("/predict")
def predict_legacy(text: str):
    """Legacy endpoint for backward compatibility."""
    input_data = TextInput(text=text)
    result = classify_text(input_data)
    return {"text": result.text, "is_clickbait": result.is_clickbait}
