from fastapi.testclient import TestClient
from src.clickbait_classifier.api import app
import pytest
from unittest.mock import patch

client = TestClient(app)

def test_read_main():
    """Sjekker at root-endepunktet svarer."""
    response = client.get("/")
    assert response.status_code == 200
    # Tilpass denne meldingen til det API-et ditt faktisk returnerer
    assert response.json() == {"message": "OK", "status-code": 200}


@patch("src.clickbait_classifier.api.tokenizer") # Patcher tokenizer i api.py
@patch("src.clickbait_classifier.api.model")

def test_predict_endpoint(mock_model, mock_tokenizer):
    """Denne tester API-logikken uten å laste tunge filer."""
    # Vi simulerer hva modellen skal returnere
    mock_model.return_value = "mocked_prediction" 
    
    with TestClient(app) as client:
        response = client.post("/predict?text=Dette er en test")
        assert response.status_code == 200
        # Vi sjekker at API-et sender ut de riktige nøklene
        assert "is_clickbait" in response.json()