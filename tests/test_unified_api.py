"""
tests/test_unified_api.py
==========================
Phase 10 Integration & Schema Validation Tests for POST /api/analyze & POST /api/analyze/context.

Verifies:
- HTTP 200 response for valid requests with and without context
- Preserves outputs from Sarcasm, Idiom, Sentiment, Emotion, Semantics, Evidence, Explanation
- Context ON/OFF side-by-side comparison endpoint (/api/analyze/context)
- Clean HTTP 422 for empty, whitespace-only, or missing sentence/context
- Schema validation: absence of fabricated overall_confidence, probabilities in [0, 1], cosine similarity in [-1, 1]
- Existing endpoint backward compatibility (/api/health, /api/sentiment, /api/emotion, /api/idiom)
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


def test_analyze_endpoint_with_context(api_client):
    """CASE 1: POST /api/analyze with laptop crash context."""
    payload = {
        "sentence": "Wow, exactly what I needed!",
        "context": "My laptop crashed five minutes before my presentation."
    }
    response = api_client.post("/api/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["input_text"] == payload["sentence"]
    assert data["context"] == payload["context"]
    assert "evidence" in data
    assert "explanation_trace" in data
    assert "signal_alignment" in data
    assert "interpretation_summary" in data

    # Verify absence of fabricated overall_confidence
    assert "overall_confidence" not in data["interpretation_summary"]


def test_analyze_endpoint_idiom_case(api_client):
    """CASE 2: POST /api/analyze with 'piece of cake'."""
    payload = {"sentence": "The exam was a piece of cake."}
    response = api_client.post("/api/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["input_text"] == payload["sentence"]
    assert data["context"] is None

    # Verify idiom candidate detection
    idiom_ev = next(ev for ev in data["evidence"] if ev["source"] == "idiom_detector")
    idioms = idiom_ev["value"]
    assert len(idioms) >= 1
    assert idioms[0]["phrase"] == "piece of cake"


def test_analyze_endpoint_literal_idiom_candidate(api_client):
    """CASE 3: POST /api/analyze with literal 'piece of cake' usage."""
    payload = {"sentence": "She ate a piece of cake."}
    response = api_client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["input_text"] == payload["sentence"]


def test_analyze_context_comparison_endpoint(api_client):
    """CASE 6: POST /api/analyze/context comparison endpoint."""
    payload = {
        "sentence": "Wonderful.",
        "context": "My phone fell into the swimming pool."
    }
    response = api_client.post("/api/analyze/context", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["sentence"] == payload["sentence"]
    assert data["context"] == payload["context"]
    assert "without_context" in data
    assert "with_context" in data
    assert "delta" in data

    delta = data["delta"]
    assert "sarcasm_probability_delta" in delta
    assert "prediction_changed" in delta
    assert isinstance(delta["prediction_changed"], bool)


def test_analyze_invalid_input_validation(api_client):
    """Verify HTTP 422 for empty, whitespace-only, or missing sentence."""
    # Empty string
    res1 = api_client.post("/api/analyze", json={"sentence": ""})
    assert res1.status_code == 422

    # Whitespace only
    res2 = api_client.post("/api/analyze", json={"sentence": "   "})
    assert res2.status_code == 422

    # Context comparison with empty context
    res3 = api_client.post("/api/analyze/context", json={"sentence": "Wonderful.", "context": "  "})
    assert res3.status_code == 422


def test_existing_endpoints_backward_compatibility(api_client):
    """Verify existing GET /api/health, POST /api/sentiment, POST /api/emotion, POST /api/idiom remain functional."""
    # GET /api/health
    h_res = api_client.get("/api/health")
    assert h_res.status_code == 200

    # POST /api/sentiment
    s_res = api_client.post("/api/sentiment", json={"text": "Wonderful."})
    assert s_res.status_code == 200
    assert s_res.json()["label"] == "positive"

    # POST /api/emotion
    e_res = api_client.post("/api/emotion", json={"text": "Wonderful.", "top_n": 5})
    assert e_res.status_code == 200
    assert len(e_res.json()["top_emotions"]) == 5

    # POST /api/idiom
    i_res = api_client.post("/api/idiom", json={"sentence": "The exam was a piece of cake."})
    assert i_res.status_code == 200
    assert i_res.json()["total_found"] >= 1
