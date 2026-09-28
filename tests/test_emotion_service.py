"""
tests/test_emotion_service.py
==============================
Unit tests for EmotionService (`src/emotion/emotion_service.py`).
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.emotion.emotion_service import EmotionService


def test_emotion_service_loading():
    service = EmotionService()
    assert service.is_loaded is True
    assert len(service.id2label) == 28


def test_emotion_service_predict():
    service = EmotionService()
    res = service.predict_emotion("I am so happy and excited today!", top_n=5)

    assert len(res["probabilities"]) == 28
    assert len(res["top_emotions"]) == 5

    # Check sorted order descending
    top_probs = [item["probability"] for item in res["top_emotions"]]
    assert top_probs == sorted(top_probs, reverse=True)

    # All probabilities in [0, 1]
    for val in res["probabilities"].values():
        assert 0.0 <= val <= 1.0


def test_emotion_service_invalid_top_n():
    service = EmotionService()
    with pytest.raises(ValueError):
        service.predict_emotion("Test text", top_n=0)
    with pytest.raises(ValueError):
        service.predict_emotion("Test text", top_n=30)


def test_emotion_service_empty_input():
    service = EmotionService()
    with pytest.raises(ValueError):
        service.predict_emotion("  ")
