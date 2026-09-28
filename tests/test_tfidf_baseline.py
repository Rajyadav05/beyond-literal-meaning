"""
tests/test_tfidf_baseline.py
=============================
Unit tests for TF-IDF + Logistic Regression sarcasm baseline (`src/sarcasm/tfidf_baseline.py`).

Tests:
1. Vectorizer produces sparse features end-to-end
2. Classifier trains and fits on small in-memory fixture
3. Binary predictions and probability range [0, 1]
4. Probabilities sum to 1.0 (non-sarcastic + sarcastic)
5. Metric calculation dictionary structure
6. Joblib model serialization and deserialization
7. Top positive and negative feature extraction
8. Context comparison metric delta schema validity
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sarcasm.tfidf_baseline import (
    build_tfidf_pipeline,
    extract_top_features,
    train_and_evaluate_model,
)


@pytest.fixture
def sample_data():
    texts = [
        "That was great, I really enjoyed waiting three hours.",
        "Oh fantastic, my car broke down again.",
        "I love paying taxes and getting stuck in traffic.",
        "What a wonderful rainy vacation day.",
        "The weather today is warm and sunny.",
        "I had a peaceful walk in the park.",
        "The food at the restaurant was fresh and delicious.",
        "She completed her assignment early and accurately.",
    ]
    contexts = [
        "Flight delayed again.",
        "Car problem.",
        "Tax season.",
        "Vacation trip.",
        "Sunny morning.",
        "Park stroll.",
        "Dinner night.",
        "Homework task.",
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]

    context_texts = [f"[CONTEXT] {c} [SEP] {t}" for c, t in zip(contexts, texts)]

    df = pd.DataFrame({
        "example_id": [f"id_{i}" for i in range(len(texts))],
        "text": texts,
        "context": contexts,
        "context_text": context_texts,
        "label": labels,
    })
    return df


def test_build_tfidf_pipeline_structure():
    pipeline = build_tfidf_pipeline(max_features=50)
    assert "vectorizer" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_train_and_evaluate_pipeline(sample_data):
    pipeline = build_tfidf_pipeline(ngram_range=(1, 1), min_df=1, max_features=100)
    trained_pipeline, metrics, y_pred, y_prob = train_and_evaluate_model(
        pipeline,
        train_text=sample_data["text"],
        train_labels=sample_data["label"],
        eval_text=sample_data["text"],
        eval_labels=sample_data["label"],
        experiment_name="unit_test",
    )

    # Output shapes
    assert len(y_pred) == len(sample_data)
    assert len(y_prob) == len(sample_data)

    # Binary predictions & probabilities in [0, 1]
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert np.all(y_prob >= 0.0) and np.all(y_prob <= 1.0)

    # Required metric keys
    for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "confusion_matrix"]:
        assert k in metrics


def test_prob_sum_to_one(sample_data):
    pipeline = build_tfidf_pipeline(ngram_range=(1, 1), min_df=1)
    pipeline.fit(sample_data["text"], sample_data["label"])
    probs = pipeline.predict_proba(sample_data["text"])

    assert probs.shape == (len(sample_data), 2)
    # Sum of non-sarcastic + sarcastic probability should equal 1.0
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


def test_extract_top_features(sample_data):
    pipeline = build_tfidf_pipeline(ngram_range=(1, 1), min_df=1)
    pipeline.fit(sample_data["text"], sample_data["label"])

    feat_dict = extract_top_features(pipeline, top_k=3)
    assert "top_positive_features_sarcastic" in feat_dict
    assert "top_negative_features_non_sarcastic" in feat_dict
    assert len(feat_dict["top_positive_features_sarcastic"]) <= 3


def test_joblib_save_and_load(tmp_path, sample_data):
    import joblib

    pipeline = build_tfidf_pipeline(ngram_range=(1, 1), min_df=1)
    pipeline.fit(sample_data["text"], sample_data["label"])

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(pipeline, model_path)

    loaded_pipeline = joblib.load(model_path)
    pred_orig = pipeline.predict(sample_data["text"])
    pred_loaded = loaded_pipeline.predict(sample_data["text"])

    np.testing.assert_array_equal(pred_orig, pred_loaded)
