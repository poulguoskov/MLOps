from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
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


# --- App setup ---
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, tokenizer

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ClickbaitClassifier(model_name=model_name)

    checkpoint_path = "models/2026-01-17_12-21-53/clickbait_model.ckpt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        weights = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
        model.load_state_dict(weights)
    else:
        model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    yield
    print("Shutting down API")


app = FastAPI(
    title="Clickbait Classifier API",
    description="Classify headlines as clickbait or not",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Endpoints ---
@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Welcome to the Clickbait Classifier API!", "status": "health"}


@app.post("/classify", response_model=ClassificationResult)
def classify_text(input_data: TextInput):
    """Classify text as clickbait or not."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = tokenizer(
        input_data.text,
        return_tensors="pt",
        truncation=True,
        max_length=input_data.max_length,
    )

    # Inference
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
    """Classify multiple texts at once."""
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
