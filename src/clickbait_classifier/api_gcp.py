import os
from http import HTTPStatus

import torch
from fastapi import FastAPI
from google.cloud import storage  # Ny import
from transformers import AutoTokenizer

from clickbait_classifier.model import ClickbaitClassifier

app = FastAPI()

model = None
tokenizer = None
BUCKET_NAME = "dtumlops-clickbait-data"
MODEL_FILE = "models/clickbait_model.ckpt"  # Path to file inside bucket (can be overridden)


def download_model_from_gcs():
    """Downloads the model file from GCS bucket to local folder."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    local_model_path = "models/downloaded_model.ckpt"
    os.makedirs("models", exist_ok=True)

    blobs = list(bucket.list_blobs(prefix="models/"))
    ckpt_blobs = [b for b in blobs if b.name.endswith(".ckpt") or b.name.endswith(".pt")]

    if not ckpt_blobs:
        raise FileNotFoundError("Did not find any model checkpoint files in the bucket.")

    latest_blob = max(ckpt_blobs, key=lambda b: b.updated)
    print(f"Downloading model from bucket: {latest_blob.name}...")

    latest_blob.download_to_filename(local_model_path)
    print("Download complete!")
    return local_model_path


@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ClickbaitClassifier(model_name=model_name)

    try:
        model_path = download_model_from_gcs()
    except Exception as e:
        print(f"Could not download from Cloud Storage: {e}")
        raise e

    print(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")

    if "state_dict" in state_dict:
        raw_weights = state_dict["state_dict"]
        clean_weights = {k.replace("model.", ""): v for k, v in raw_weights.items()}
        model.load_state_dict(clean_weights)
    else:
        model.load_state_dict(state_dict)

    model.eval()


@app.post("/predict")
async def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        prediction = torch.argmax(logits, dim=1).item()
    return {"text": text, "is_clickbait": bool(prediction)}


@app.get("/")
def root():
    return {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK}
