"""
tests/test_emotion_api.py
==========================
Integration unit tests for POST /api/emotion endpoint.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app

client = TestClient(app)


def test_api_emotion_success():
    response = client.post("/api/emotion", json={
        "text": "I am so happy and grateful!",
        "top_n": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["probabilities"]) == 28
    assert len(data["top_emotions"]) == 5


def test_api_emotion_invalid_top_n():
    response = client.post("/api/emotion", json={
        "text": "Hello world",
        "top_n": 50
    })
    assert response.status_code == 422


def test_api_emotion_empty_text():
    response = client.post("/api/emotion", json={
        "text": "   "
    })
    assert response.status_code == 422
