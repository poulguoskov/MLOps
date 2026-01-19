from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from clickbait_classifier.model import ClickbaitClassifier


class TextInput(BaseModel):
    """Request body schema for text classification."""

    text: str
    max_length: int = 128


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, tokenizer

    # Startup: load model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ClickbaitClassifier(model_name=model_name)

    # Load the checkpoint
    checkpoint_path = "models/2026-01-17_12-21-53/clickbait_model.ckpt"
    state_dict = torch.load(checkpoint_path, map_location="mps")
    if "state_dict" in state_dict:
        weights = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
        model.load_state_dict(weights)
    else:
        model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    yield  # App runs here

    # Shutdown: cleanup (optional)
    print("Shutting down API")


app = FastAPI(lifespan=lifespan)

# Global variables for model and tokenizer
model = None
tokenizer = None


@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Welcome to the Clickbait Classifier API!"}


@app.post("/classify")
def classify_text(input_data: TextInput):
    """Classify text as clickbait or not."""
    # Tokenize
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

    return {
        "text": input_data.text,
        "is_clickbait": bool(prediction),
        "confidence": probabilities[prediction],
    }
