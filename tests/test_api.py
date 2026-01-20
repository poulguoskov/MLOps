import glob

import pytest
from fastapi.testclient import TestClient

from src.clickbait_classifier.api import app

# All tests need model because API loads it on startup
_has_model = glob.glob("models/**/*.ckpt", recursive=True) or glob.glob("models/**/*.pt", recursive=True)


@pytest.mark.skipif(not _has_model, reason="No model files found - API requires model on startup")
def test_read_main():
    """Check that root endpoint responds."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        json_data = response.json()
        assert "message" in json_data
        assert "status" in json_data


@pytest.mark.skipif(not _has_model, reason="No model files found - API requires model on startup")
def test_classify_endpoint():
    """Test the classify endpoint with a non-clickbait headline."""
    with TestClient(app) as client:
        response = client.post("/classify", json={"text": "Scientists publish research findings"})
        assert response.status_code == 200
        json_data = response.json()
        assert "is_clickbait" in json_data
        assert "confidence" in json_data
        assert json_data["text"] == "Scientists publish research findings"
        assert isinstance(json_data["is_clickbait"], bool)
        assert 0 <= json_data["confidence"] <= 1


@pytest.mark.skipif(not _has_model, reason="No model files found - API requires model on startup")
def test_classify_clickbait():
    """Test the classify endpoint with a clickbait headline."""
    with TestClient(app) as client:
        response = client.post("/classify", json={"text": "You Will NEVER Believe What Happened Next!"})
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["is_clickbait"] is True
        assert json_data["confidence"] > 0.5


@pytest.mark.skipif(not _has_model, reason="No model files found - API requires model on startup")
def test_classify_batch_endpoint():
    """Test the batch classify endpoint."""
    with TestClient(app) as client:
        response = client.post(
            "/classify/batch",
            json={
                "texts": [
                    "Scientists discover new species",
                    "This ONE Trick Will Change Your Life Forever",
                ]
            },
        )
        assert response.status_code == 200
        json_data = response.json()
        assert "results" in json_data
        assert len(json_data["results"]) == 2
        # First should be not clickbait, second should be clickbait
        assert json_data["results"][0]["is_clickbait"] is False
        assert json_data["results"][1]["is_clickbait"] is True


@pytest.mark.skipif(not _has_model, reason="No model files found - API requires model on startup")
def test_classify_missing_text():
    """Test that missing text returns 422 validation error."""
    with TestClient(app) as client:
        response = client.post("/classify", json={})
        assert response.status_code == 422  # Validation error
