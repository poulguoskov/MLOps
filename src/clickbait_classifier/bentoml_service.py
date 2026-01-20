"""BentoML service for clickbait classification using ONNX with adaptive batching."""

import bentoml
import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizer


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 30},
)
class ClickbaitClassifier:
    """Clickbait classification service using ONNX runtime with adaptive batching."""

    def __init__(self):
        """Load model and tokenizer on startup."""
        from pathlib import Path

        # Find ONNX model
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

    def _predict_batch(self, texts: list[str]) -> list[dict]:
        """Internal batch prediction."""
        # Tokenize all texts
        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            padding="max_length",
            max_length=128,
            truncation=True,
        )

        # Run batch inference
        outputs = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )

        logits = outputs[0]
        results = []
        for i, text in enumerate(texts):
            probs = np.exp(logits[i]) / np.exp(logits[i]).sum()
            pred_class = int(np.argmax(logits[i]))
            results.append(
                {
                    "text": text,
                    "is_clickbait": pred_class == 1,
                    "confidence": float(probs[pred_class]),
                }
            )
        return results

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=32,
        max_latency_ms=100,
    )
    def classify(self, texts: list[str]) -> list[dict]:
        """Classify texts with adaptive batching.

        BentoML will automatically batch incoming requests together
        up to max_batch_size or max_latency_ms, whichever comes first.
        """
        return self._predict_batch(texts)
