import glob
import os
from http import HTTPStatus

import torch
from fastapi import FastAPI
from transformers import AutoTokenizer

from clickbait_classifier.model import ClickbaitClassifier  # Importerer din klasse

app = FastAPI()

# Vi definerer globale variabler som lastes ved oppstart
model = None
tokenizer = None


@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ClickbaitClassifier(model_name=model_name)

    # 1. Finn alle modellfiler i alle undermapper av 'models'
    # Denne leter etter både .pt og .ckpt filer
    list_of_files = glob.glob("models/**/*.ckpt", recursive=True) + glob.glob("models/**/*.pt", recursive=True)

    if not list_of_files:
        raise FileNotFoundError("Ingen modellfiler funnet i 'models/' mappen!")

    # 2. Finn den nyeste filen basert på endringstidspunkt (mtime)
    latest_file = max(list_of_files, key=os.path.getmtime)
    print(f"Laster nyeste modell: {latest_file}")

    # 3. Last vektene
    # ... (resten av koden din før lasting)

    state_dict = torch.load(latest_file, map_location="cpu")

    if "state_dict" in state_dict:
        raw_weights = state_dict["state_dict"]
        # Vi lager en ny dictionary der vi fjerner "model." fra starten av alle nøkler
        clean_weights = {k.replace("model.", ""): v for k, v in raw_weights.items()}
        model.load_state_dict(clean_weights)
    else:
        model.load_state_dict(state_dict)

    model.eval()


@app.post("/predict")
async def predict(text: str):
    # 1. Tokenisering (tekst -> tall)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    # 2. Inference (tall -> prediksjon)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        prediction = torch.argmax(logits, dim=1).item()

    return {"text": text, "is_clickbait": bool(prediction)}


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response
