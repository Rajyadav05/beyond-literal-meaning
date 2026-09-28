"""
tests/test_sentiment_api.py
============================
Integration unit tests for POST /api/sentiment endpoint.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app

client = TestClient(app)


def test_api_sentiment_positive():
    response = client.post("/api/sentiment", json={
        "text": "I absolutely love this product!"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["positive", "neutral", "negative"]
    assert "probabilities" in data
    assert set(data["probabilities"].keys()) == {"negative", "neutral", "positive"}


def test_api_sentiment_empty():
    response = client.post("/api/sentiment", json={
        "text": "   "
    })
    assert response.status_code == 422
