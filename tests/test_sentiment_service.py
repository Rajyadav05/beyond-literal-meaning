"""
tests/test_sentiment_service.py
================================
Unit tests for SentimentService (`src/sentiment/sentiment_service.py`).
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment.sentiment_service import SentimentService


def test_sentiment_service_loading():
    service = SentimentService()
    assert service.is_loaded is True
    assert len(service.id2label) == 3


def test_sentiment_service_predict():
    service = SentimentService()
    res = service.predict_sentiment("Wonderful product, highly recommend!")

    assert res["label"] == "positive"
    assert "probabilities" in res
    assert set(res["probabilities"].keys()) == {"negative", "neutral", "positive"}

    prob_sum = sum(res["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-4


def test_sentiment_service_empty_input():
    service = SentimentService()
    with pytest.raises(ValueError):
        service.predict_sentiment("   ")
