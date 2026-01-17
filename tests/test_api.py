from fastapi.testclient import TestClient
from src.clickbait_classifier.api import app
import pytest

client = TestClient(app)

def test_read_main():
    """Sjekker at root-endepunktet svarer."""
    response = client.get("/")
    assert response.status_code == 200
    # Tilpass denne meldingen til det API-et ditt faktisk returnerer
    assert response.json() == {"message": "OK", "status-code": 200}

def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post("/predict?text=Dette er en test")
        assert response.status_code == 200
        json_data = response.json()
        assert "is_clickbait" in json_data
        assert json_data["text"] == "Dette er en test"