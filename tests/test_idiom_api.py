"""
tests/test_idiom_api.py
=======================
Integration unit tests for POST /api/idiom endpoint.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app

client = TestClient(app)


def test_api_idiom_piece_of_cake():
    response = client.post("/api/idiom", json={
        "sentence": "The exam was a piece of cake."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["sentence"] == "The exam was a piece of cake."
    assert data["total_found"] >= 1
    found = data["idioms_found"][0]
    assert found["detected"] is True
    assert found["phrase"] == "piece of cake"
    assert found["span_start"] == 15
    assert found["span_end"] == 28
    assert found["usage"] in ["idiomatic", "literal"]


def test_api_idiom_spilled_the_beans():
    response = client.post("/api/idiom", json={
        "sentence": "Rahul finally spilled the beans."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total_found"] >= 1
    found = data["idioms_found"][0]
    assert found["detected"] is True
    assert found["phrase"] == "spilled the beans"
    assert found["span_start"] == 14
    assert found["span_end"] == 31


def test_api_idiom_no_idiom():
    response = client.post("/api/idiom", json={
        "sentence": "The sky is blue and sunny today."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total_found"] == 0
    assert len(data["idioms_found"]) == 0


def test_api_idiom_empty_sentence():
    response = client.post("/api/idiom", json={
        "sentence": ""
    })
    assert response.status_code == 422
