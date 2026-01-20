"""BentoML service for clickbait classification using ONNX with adaptive batching.

BentoML collects multiple incoming requests into a batch (adaptive batching),
but since our ONNX model only supports batch_size=1, we process them sequentially.
This still provides benefits: reduced HTTP overhead and better request management.
"""

import bentoml
import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizer


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 30},
)
class ClickbaitClassifier:
    """Clickbait classification service using ONNX runtime."""

    def __init__(self):
        """Load model and tokenizer on startup."""
        from pathlib import Path

        model_paths = [
            Path("models/clickbait_model.onnx"),
            Path("/app/models/clickbait_model.onnx"),
        ]

        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = str(path)
                break

        if model_path is None:
            raise FileNotFoundError("No ONNX model found")

        print(f"Loading ONNX model from {model_path}")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        print("Model and tokenizer loaded")

    def _predict_single(self, text: str) -> dict:
        """Predict for a single text (batch_size=1 for ONNX)."""
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            max_length=128,
            truncation=True,
        )

        outputs = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )

        logits = outputs[0][0]
        probs = np.exp(logits) / np.exp(logits).sum()
        pred_class = int(np.argmax(logits))

        return {
            "text": text,
            "is_clickbait": pred_class == 1,
            "confidence": float(probs[pred_class]),
        }

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=500,
    )
    def classify(self, texts: list[str]) -> list[dict]:
        """Classify texts with adaptive batching.

        BentoML collects multiple requests and passes them as a list.
        We process each one individually since ONNX model has batch_size=1.
        """
        return [self._predict_single(text) for text in texts]
