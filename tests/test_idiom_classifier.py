"""
tests/test_idiom_classifier.py
===============================
Unit tests for idiom baseline classifier (`src/idiom/baseline.py`).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.idiom.baseline import IdiomTfidfBaseline


def test_idiom_baseline_fit_predict():
    train_texts = pd.Series([
        "The exam was a piece of cake.",
        "She ate a piece of cake.",
        "He spilled the beans about the party.",
        "He cooked the beans in a pot."
    ])
    train_labels = pd.Series([1, 0, 1, 0])

    model = IdiomTfidfBaseline(random_seed=42)
    model.fit(train_texts, train_labels)

    preds = model.predict(train_texts)
    probs = model.predict_proba(train_texts)

    assert len(preds) == 4
    assert probs.shape == (4, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    metrics = model.evaluate(train_texts, train_labels)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
