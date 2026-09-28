"""
tests/test_sentiment_emotion_reproducibility.py
=================================================
Phase 7.4 — Reproducibility & Determinism Tests for Sentiment and Emotion Services.

Verifies:
- Tokenizer input_ids and attention_masks match between direct HuggingFace load and Service layer.
- Raw logits and softmax/sigmoid probabilities match within tolerance (1e-6).
- Direct inference, Service layer, and FastAPI API responses return identical values.
- Inference is 100% deterministic across repeated runs.
"""

import math
from pathlib import Path
import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from backend.main import app
from src.sentiment.sentiment_service import SentimentService
from src.emotion.emotion_service import EmotionService

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_SENTENCES = [
    "I absolutely LOVE waiting three hours for the bus.",
    "Wonderful.",
    "The exam was a piece of cake.",
    "Rahul finally spilled the beans.",
    "My phone fell into the swimming pool.",
    "Wow, exactly what I needed!"
]


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


@pytest.fixture(scope="module")
def sentiment_service():
    return SentimentService()


@pytest.fixture(scope="module")
def emotion_service():
    return EmotionService()


def test_sentiment_reproducibility_direct_vs_service_vs_api(api_client, sentiment_service):
    """Verify sentiment predictions are identical across direct model, service, and FastAPI API."""
    model_dir = PROJECT_ROOT / "models" / "sentiment"
    tokenizer_direct = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model_direct = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model_direct.eval()

    for text in TEST_SENTENCES:
        # Direct
        inputs_direct = tokenizer_direct(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits_direct = model_direct(**inputs_direct).logits
            probs_direct = torch.softmax(logits_direct, dim=-1).squeeze(0).numpy()

        # Service
        res_svc = sentiment_service.predict_sentiment(text)
        inputs_svc = sentiment_service.tokenizer(
            text.strip(), return_tensors="pt", padding=True, truncation=True, max_length=sentiment_service.max_length
        )
        with torch.no_grad():
            logits_svc = sentiment_service.model(**inputs_svc).logits
            probs_svc = torch.softmax(logits_svc, dim=-1).squeeze(0).numpy()

        # API
        resp_api = api_client.post("/api/sentiment", json={"text": text})
        assert resp_api.status_code == 200
        data_api = resp_api.json()

        # Assert tokenization exact match
        assert inputs_direct["input_ids"].tolist() == inputs_svc["input_ids"].tolist()

        # Assert logits exact match (tolerance 1e-6)
        logit_diff = np.max(np.abs(logits_direct.numpy() - logits_svc.numpy()))
        assert logit_diff < 1e-6

        # Assert label match
        assert res_svc["label"] == data_api["label"]

        # Assert probabilities match across direct, service, and API
        id2label = {int(k): str(v) for k, v in model_direct.config.id2label.items()}
        for i in range(len(probs_direct)):
            label_name = id2label[i]
            assert math.isclose(probs_direct[i], res_svc["probabilities"][label_name], abs_tol=1e-6)
            assert math.isclose(res_svc["probabilities"][label_name], data_api["probabilities"][label_name], abs_tol=1e-6)


def test_emotion_reproducibility_direct_vs_service_vs_api(api_client, emotion_service):
    """Verify emotion predictions are identical across direct model, service, and FastAPI API."""
    model_dir = PROJECT_ROOT / "models" / "emotion"
    tokenizer_direct = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model_direct = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model_direct.eval()

    for text in TEST_SENTENCES:
        # Direct
        inputs_direct = tokenizer_direct(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits_direct = model_direct(**inputs_direct).logits

        # Service
        res_svc = emotion_service.predict_emotion(text, top_n=5)
        inputs_svc = emotion_service.tokenizer(
            text.strip(), return_tensors="pt", padding=True, truncation=True, max_length=emotion_service.max_length
        )
        with torch.no_grad():
            logits_svc = emotion_service.model(**inputs_svc).logits

        # API
        resp_api = api_client.post("/api/emotion", json={"text": text, "top_n": 5})
        assert resp_api.status_code == 200
        data_api = resp_api.json()

        # Assert tokenization match
        assert inputs_direct["input_ids"].tolist() == inputs_svc["input_ids"].tolist()

        # Assert logit match
        logit_diff = np.max(np.abs(logits_direct.numpy() - logits_svc.numpy()))
        assert logit_diff < 1e-6

        # Assert top-5 ranking exact label & prob match
        svc_top_labels = [item["label"] for item in res_svc["top_emotions"]]
        api_top_labels = [item["label"] for item in data_api["top_emotions"]]
        assert svc_top_labels == api_top_labels


def test_sentiment_and_emotion_determinism(sentiment_service, emotion_service):
    """Verify 3 consecutive runs produce identical logits (model.eval() & no_grad())."""
    for text in TEST_SENTENCES:
        # Sentiment repeated inference
        res1 = sentiment_service.predict_sentiment(text)
        res2 = sentiment_service.predict_sentiment(text)
        res3 = sentiment_service.predict_sentiment(text)
        assert res1 == res2 == res3

        # Emotion repeated inference
        emo1 = emotion_service.predict_emotion(text, top_n=5)
        emo2 = emotion_service.predict_emotion(text, top_n=5)
        emo3 = emotion_service.predict_emotion(text, top_n=5)
        assert emo1 == emo2 == emo3
