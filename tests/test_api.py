from unittest.mock import MagicMock, patch

import torch
from fastapi.testclient import TestClient

from src.clickbait_classifier.api import app

# Vi lager en TestClient uten 'with'-blokk her for de enkle testene
client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "OK", "status-code": 200}


def test_predict_endpoint():
    # Vi må "fylle" de globale variablene i api.py med mocks
    # slik at de ikke er 'None' når testen kjører
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.1, 0.9]])  # Simulerer clickbait

    # Vi patcher tokenizer og model inne i api.py modulen
    with (
        patch("src.clickbait_classifier.api.tokenizer", mock_tokenizer),
        patch("src.clickbait_classifier.api.model", mock_model),
    ):
        response = client.post("/predict?text=Dette er en test")

        assert response.status_code == 200
        assert "is_clickbait" in response.json()
        assert response.json()["is_clickbait"] is True
