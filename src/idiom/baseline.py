"""
src/idiom/baseline.py
======================
Phase 6 Baseline Classifier for Idiom Contextual Usage (Literal vs. Idiomatic).

Uses TF-IDF feature extraction + Logistic Regression to classify whether a candidate
idiom expression is used literally (0) or idiomatically (1) in context.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("idiom_baseline")


class IdiomTfidfBaseline:
    def __init__(self, random_seed: int = 42, max_features: int = 10000):
        self.random_seed = random_seed
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(
            random_state=self.random_seed,
            max_iter=1000,
            solver="liblinear"
        )
        self.is_fitted = False

    def fit(self, sentences: pd.Series, labels: pd.Series):
        """Fits TF-IDF vectorizer and Logistic Regression classifier."""
        X = self.vectorizer.fit_transform(sentences)
        self.classifier.fit(X, labels)
        self.is_fitted = True

    def predict(self, sentences: pd.Series) -> np.ndarray:
        """Predicts binary labels (0 = literal, 1 = idiomatic)."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        X = self.vectorizer.transform(sentences)
        return self.classifier.predict(X)

    def predict_proba(self, sentences: pd.Series) -> np.ndarray:
        """Predicts probability distribution [p_literal, p_idiomatic]."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        X = self.vectorizer.transform(sentences)
        return self.classifier.predict_proba(X)

    def evaluate(self, sentences: pd.Series, labels: pd.Series) -> Dict[str, Any]:
        """Evaluates model performance and returns clean metric dictionary."""
        preds = self.predict(sentences)
        probs = self.predict_proba(sentences)[:, 1]

        acc = accuracy_score(labels, preds)
        prec_macro = precision_score(labels, preds, average="macro", zero_division=0)
        rec_macro = recall_score(labels, preds, average="macro", zero_division=0)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

        try:
            roc_auc = roc_auc_score(labels, probs)
        except Exception:
            roc_auc = 0.5

        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        report = classification_report(labels, preds, output_dict=True, zero_division=0)

        return {
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "roc_auc": float(roc_auc),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "classification_report": report
        }

    def save(self, model_dir: Path, model_name: str = "tfidf_idiom_baseline"):
        """Saves fitted model and vectorizer to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, model_dir / f"{model_name}_vectorizer.joblib")
        joblib.dump(self.classifier, model_dir / f"{model_name}_classifier.joblib")
        logger.info("Saved baseline model to: %s", model_dir)

    def load(self, model_dir: Path, model_name: str = "tfidf_idiom_baseline"):
        """Loads fitted model and vectorizer from disk."""
        model_dir = Path(model_dir)
        self.vectorizer = joblib.load(model_dir / f"{model_name}_vectorizer.joblib")
        self.classifier = joblib.load(model_dir / f"{model_name}_classifier.joblib")
        self.is_fitted = True
        logger.info("Loaded baseline model from: %s", model_dir)
