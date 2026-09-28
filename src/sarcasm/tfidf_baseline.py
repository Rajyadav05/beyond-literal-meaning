"""
src/sarcasm/tfidf_baseline.py
==============================
TF-IDF + Logistic Regression Sarcasm Baseline Classifier.

Provides modular methods for:
- Building scikit-learn TF-IDF + LogisticRegression pipelines
- Training and evaluating sentence-only and context-aware models
- Extracting top positive and negative features (coefficients)
- Generating confusion matrix plots and test prediction DataFrames
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
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
from sklearn.pipeline import Pipeline

logger = logging.getLogger("tfidf_baseline")


def build_tfidf_pipeline(
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 3,
    max_df: float = 0.95,
    max_features: int = 300000,
    sublinear_tf: bool = True,
    max_iter: int = 1000,
    random_state: int = 42,
    C: float = 1.0,
) -> Pipeline:
    """
    Constructs a memory-safe scikit-learn Pipeline with TfidfVectorizer + LogisticRegression.
    Maintains sparse matrices end-to-end to prevent memory exhaustion.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
        dtype=np.float32,
    )

    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )

    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def train_and_evaluate_model(
    pipeline: Pipeline,
    train_text: pd.Series,
    train_labels: pd.Series,
    eval_text: pd.Series,
    eval_labels: pd.Series,
    experiment_name: str = "sentence",
) -> Tuple[Pipeline, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Trains the TF-IDF + LogisticRegression pipeline and evaluates on eval split.

    Returns:
    (trained_pipeline, metrics_dict, y_pred, y_prob)
    """
    logger.info("Training TF-IDF + Logistic Regression [%s]...", experiment_name)
    pipeline.fit(train_text, train_labels)

    logger.info("Evaluating [%s] model...", experiment_name)
    y_pred = pipeline.predict(eval_text)
    y_prob = pipeline.predict_proba(eval_text)[:, 1]

    acc = float(accuracy_score(eval_labels, y_pred))
    prec_macro = float(precision_score(eval_labels, y_pred, average="macro"))
    rec_macro = float(recall_score(eval_labels, y_pred, average="macro"))
    f1_macro = float(f1_score(eval_labels, y_pred, average="macro"))

    prec_0 = float(precision_score(eval_labels, y_pred, pos_label=0))
    rec_0 = float(recall_score(eval_labels, y_pred, pos_label=0))
    f1_0 = float(f1_score(eval_labels, y_pred, pos_label=0))

    prec_1 = float(precision_score(eval_labels, y_pred, pos_label=1))
    rec_1 = float(recall_score(eval_labels, y_pred, pos_label=1))
    f1_1 = float(f1_score(eval_labels, y_pred, pos_label=1))

    roc_auc = float(roc_auc_score(eval_labels, y_prob))
    cm = confusion_matrix(eval_labels, y_pred)

    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "per_class": {
            "non_sarcastic_0": {"precision": prec_0, "recall": rec_0, "f1": f1_0, "support": int((eval_labels == 0).sum())},
            "sarcastic_1": {"precision": prec_1, "recall": rec_1, "f1": f1_1, "support": int((eval_labels == 1).sum())},
        },
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "classification_report": classification_report(eval_labels, y_pred, output_dict=True),
    }

    return pipeline, metrics, y_pred, y_prob


def extract_top_features(
    pipeline: Pipeline, top_k: int = 20
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extracts top positive (sarcasm indicator) and negative (non-sarcasm indicator)
    n-gram features along with their logistic regression coefficients.
    """
    vectorizer: TfidfVectorizer = pipeline.named_steps["vectorizer"]
    classifier: LogisticRegression = pipeline.named_steps["classifier"]

    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]

    # Sort feature indices by coefficient
    top_pos_indices = np.argsort(coefs)[-top_k:][::-1]
    top_neg_indices = np.argsort(coefs)[:top_k]

    top_positive = [
        {"feature": str(feature_names[idx]), "coefficient": float(coefs[idx])}
        for idx in top_pos_indices
    ]

    top_negative = [
        {"feature": str(feature_names[idx]), "coefficient": float(coefs[idx])}
        for idx in top_neg_indices
    ]

    return {
        "top_positive_features_sarcastic": top_positive,
        "top_negative_features_non_sarcastic": top_negative,
    }


def save_confusion_matrix_plot(
    cm_dict: Dict[str, int], title: str, save_path: Path
) -> None:
    """
    Renders and saves a confusion matrix plot to disk.
    cm_dict format: {'tn': ..., 'fp': ..., 'fn': ..., 'tp': ...}
    """
    cm = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Non-Sarcastic (0)", "Sarcastic (1)"],
        yticklabels=["Non-Sarcastic (0)", "Sarcastic (1)"],
        title=title,
        ylabel="Actual Label",
        xlabel="Predicted Label",
    )

    # Format text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
