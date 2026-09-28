"""
tests/test_challenge_sqlite.py
===============================
Phase 12 unit and API tests for SQLite Challenge feedback persistence.

Verifies:
1. Database table initialization
2. Valid 'correct' submission
3. Valid 'incorrect' submission
4. Optional feedback
5. Empty sentence rejection (422 / ValueError)
6. Invalid verdict rejection (422 / ValueError)
7. Stats with zero submissions (total=0, correct=0, incorrect=0, accuracy=None)
8. Stats after 1 submission
9. Stats after multiple submissions
10. Accuracy calculation
11. Persistence across database connections
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from backend.database import (

    init_db,
    record_challenge_evaluation,
    get_challenge_stats,
    get_recent_challenge_evaluations,
    get_db_connection,
)
from backend.main import app


@pytest.fixture
def temp_db_path():
    """Create a temporary SQLite database file for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def client(temp_db_path, monkeypatch):
    """FastAPI TestClient with monkeypatched database path."""
    import backend.database as db_mod
    monkeypatch.setattr(db_mod, "DEFAULT_DB_PATH", temp_db_path)
    with TestClient(app) as c:
        yield c


# ===========================================================================
# Database Unit Tests
# ===========================================================================

def test_init_db_creates_table(temp_db_path):
    conn = get_db_connection(temp_db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='challenge_evaluations';")
        row = cursor.fetchone()
        assert row is not None, "Table challenge_evaluations should exist"
    finally:
        conn.close()


def test_record_correct_submission(temp_db_path):
    res = record_challenge_evaluation(
        sentence="The exam was a piece of cake.",
        context="I studied hard.",
        model_prediction="not_sarcastic",
        model_interpretation="Literal assertion of ease.",
        user_verdict="correct",
        user_feedback="Accurate prediction.",
        db_path=temp_db_path,
    )
    assert res["submission_id"] > 0
    assert res["user_verdict"] == "correct"
    assert "created_at" in res


def test_record_incorrect_submission(temp_db_path):
    res = record_challenge_evaluation(
        sentence="Wonderful.",
        context="Lost luggage.",
        model_prediction="not_sarcastic",
        model_interpretation="Literal positive expression.",
        user_verdict="incorrect",
        user_feedback="Missed sarcasm!",
        db_path=temp_db_path,
    )
    assert res["submission_id"] > 0
    assert res["user_verdict"] == "incorrect"


def test_empty_sentence_rejected(temp_db_path):
    with pytest.raises(ValueError, match="Sentence must not be empty"):
        record_challenge_evaluation(
            sentence="   ",
            model_prediction="sarcastic",
            user_verdict="correct",
            db_path=temp_db_path,
        )


def test_invalid_verdict_rejected(temp_db_path):
    with pytest.raises(ValueError, match="user_verdict must be either 'correct' or 'incorrect'"):
        record_challenge_evaluation(
            sentence="Valid sentence.",
            model_prediction="sarcastic",
            user_verdict="maybe",
            db_path=temp_db_path,
        )


def test_stats_zero_submissions(temp_db_path):
    stats = get_challenge_stats(temp_db_path)
    assert stats["total_reviews"] == 0
    assert stats["correct"] == 0
    assert stats["incorrect"] == 0
    assert stats["accuracy"] is None


def test_stats_after_one_submission(temp_db_path):
    record_challenge_evaluation(
        sentence="Test 1",
        model_prediction="sarcastic",
        user_verdict="correct",
        db_path=temp_db_path,
    )
    stats = get_challenge_stats(temp_db_path)
    assert stats["total_reviews"] == 1
    assert stats["correct"] == 1
    assert stats["incorrect"] == 0
    assert stats["accuracy"] == 1.0


def test_stats_after_multiple_submissions_and_accuracy(temp_db_path):
    record_challenge_evaluation(sentence="S1", model_prediction="sarcastic", user_verdict="correct", db_path=temp_db_path)
    record_challenge_evaluation(sentence="S2", model_prediction="sarcastic", user_verdict="incorrect", db_path=temp_db_path)
    record_challenge_evaluation(sentence="S3", model_prediction="sarcastic", user_verdict="correct", db_path=temp_db_path)
    record_challenge_evaluation(sentence="S4", model_prediction="sarcastic", user_verdict="correct", db_path=temp_db_path)

    stats = get_challenge_stats(temp_db_path)
    assert stats["total_reviews"] == 4
    assert stats["correct"] == 3
    assert stats["incorrect"] == 1
    assert stats["accuracy"] == 0.75


def test_persistence_across_connections(temp_db_path):
    res = record_challenge_evaluation(sentence="Persist test", model_prediction="not_sarcastic", user_verdict="correct", db_path=temp_db_path)
    sub_id = res["submission_id"]

    # Re-open connection
    recent = get_recent_challenge_evaluations(limit=10, db_path=temp_db_path)
    assert len(recent) == 1
    assert recent[0]["id"] == sub_id
    assert recent[0]["sentence"] == "Persist test"


# ===========================================================================
# FastAPI Endpoint Integration Tests
# ===========================================================================

def test_api_post_challenge_success(client):
    payload = {
        "sentence": "I absolutely LOVE waiting three hours for the bus.",
        "context": "Standing in the freezing cold after a long shift.",
        "model_prediction": "sarcastic",
        "model_interpretation": "Extreme frustration with transport delay.",
        "user_verdict": "correct",
        "user_feedback": "Spot on evaluation."
    }
    resp = client.post("/api/challenge", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["submission_id"] > 0
    assert data["user_verdict"] == "correct"
    assert "created_at" in data


def test_api_post_challenge_invalid_verdict(client):
    payload = {
        "sentence": "Sample text",
        "model_prediction": "sarcastic",
        "user_verdict": "invalid_choice"
    }
    resp = client.post("/api/challenge", json=payload)
    assert resp.status_code == 422


def test_api_post_challenge_empty_sentence(client):
    payload = {
        "sentence": "   ",
        "model_prediction": "sarcastic",
        "user_verdict": "correct"
    }
    resp = client.post("/api/challenge", json=payload)
    assert resp.status_code == 422


def test_api_get_challenge_stats(client):
    # Initial stats check
    resp = client.get("/api/challenge/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "stats" in data
    assert data["stats"]["total_reviews"] == 0
    assert data["stats"]["accuracy"] is None
    assert data["recent"] == []

    # Submit one item
    client.post("/api/challenge", json={
        "sentence": "Test sentence",
        "model_prediction": "sarcastic",
        "user_verdict": "correct"
    })

    resp2 = client.get("/api/challenge/stats")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["stats"]["total_reviews"] == 1
    assert data2["stats"]["correct"] == 1
    assert data2["stats"]["accuracy"] == 1.0
    assert len(data2["recent"]) == 1
    assert data2["recent"][0]["sentence"] == "Test sentence"


def test_api_get_challenge_recent(client):
    client.post("/api/challenge", json={
        "sentence": "Recent 1",
        "model_prediction": "sarcastic",
        "user_verdict": "correct"
    })
    resp = client.get("/api/challenge/recent")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["sentence"] == "Recent 1"
