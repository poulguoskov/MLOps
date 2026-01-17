from unittest.mock import patch

from fastapi.testclient import TestClient

from src.clickbait_classifier.api import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "OK", "status-code": 200}


# Vi må "mocke" glob slik at den returnerer en liksom-fil,
# og vi må "mocke" ClickbaitClassifier så den ikke prøver å laste vekter.
@patch("src.clickbait_classifier.api.glob.glob")
@patch("src.clickbait_classifier.api.ClickbaitClassifier")
@patch("src.clickbait_classifier.api.AutoTokenizer")
def test_predict_endpoint(mock_tokenizer, mock_classifier, mock_glob):
    # 1. Vi later som glob finner en fil så FileNotFoundError ikke kastes
    mock_glob.return_value = ["models/fake_model.ckpt"]

    # 2. Vi later som modellen og tokenizeren lastes fint
    mock_classifier.return_value = lambda x: x  # Enkel dummy-funksjon

    with TestClient(app) as client:
        # Nå vil startup_event kjøre uten å krasje!
        response = client.post("/predict?text=Dette er en test")
        assert response.status_code == 200
        assert "is_clickbait" in response.json()
