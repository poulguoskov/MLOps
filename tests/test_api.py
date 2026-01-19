import glob

import pytest
from fastapi.testclient import TestClient

from src.clickbait_classifier.api import app

client = TestClient(app)


def test_read_main():
    """Check that root endpoint responds."""
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "message" in json_data
    assert "status" in json_data


@pytest.mark.skipif(
    not (glob.glob("models/**/*.ckpt", recursive=True) or glob.glob("models/**/*.pt", recursive=True)),
    reason="No model files found - skipping predict endpoint test",
)
def test_classify_endpoint():
    """Test the classify endpoint."""
    with TestClient(app) as client:
        response = client.post("/classify", json={"text": "This is a test headline"})
        assert response.status_code == 200
        json_data = response.json()
        assert "is_clickbait" in json_data
        assert "confidence" in json_data
        assert json_data["text"] == "This is a test headline"
