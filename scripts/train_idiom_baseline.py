"""
scripts/train_idiom_baseline.py
================================
Phase 6 Step 6 & 7 — Trains and evaluates TF-IDF + Logistic Regression Idiom Classifier.

Evaluates two representations:
1. Sentence-Only (target sentence)
2. Context + Sentence (surrounding context)

Saves baseline models, metrics JSON (`results/metrics/idiom_baseline_metrics.json`),
and confusion matrix visualization.

Usage:
    python scripts/train_idiom_baseline.py
"""

import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.idiom.baseline import IdiomTfidfBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_idiom_baseline")


def save_confusion_matrix_plot(cm_dict: dict, title: str, save_path: Path):
    plt.figure(figsize=(5, 4))
    cm_arr = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]]
    ])
    plt.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Literal (0)", "Idiomatic (1)"])
    plt.yticks(tick_marks, ["Literal (0)", "Idiomatic (1)"])
    
    thresh = cm_arr.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm_arr[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm_arr[i, j] > thresh else "black")
            
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix plot to: %s", save_path)


def run_baseline_experiment():
    semeval_dir = PROJECT_ROOT / "datasets" / "processed" / "semeval_idiom"
    models_dir = PROJECT_ROOT / "models" / "idiom"
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    cm_dir = PROJECT_ROOT / "results" / "confusion_matrices"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(semeval_dir / "train.parquet")
    test_df = pd.read_parquet(semeval_dir / "test.parquet")

    logger.info("Loaded SemEval dataset -> Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    # --- EXPERIMENT A: Sentence-Only ---
    logger.info("\n--- Training TF-IDF Baseline (Sentence Only) ---")
    model_sent = IdiomTfidfBaseline(random_seed=42)
    model_sent.fit(train_df["sentence"], train_df["label"])
    metrics_sent = model_sent.evaluate(test_df["sentence"], test_df["label"])
    model_sent.save(models_dir, model_name="tfidf_idiom_sentence")

    logger.info("Sentence Baseline -> Acc: %.4f | Macro F1: %.4f | ROC-AUC: %.4f",
                metrics_sent["accuracy"], metrics_sent["f1_macro"], metrics_sent["roc_auc"])

    cm_path_sent = cm_dir / "idiom_baseline_sentence_confusion_matrix.png"
    save_confusion_matrix_plot(metrics_sent["confusion_matrix"],
                                "SemEval Idiom Baseline (Sentence Only)",
                                cm_path_sent)

    # --- EXPERIMENT B: Context + Sentence ---
    logger.info("\n--- Training TF-IDF Baseline (Context + Sentence) ---")
    model_ctx = IdiomTfidfBaseline(random_seed=42)
    model_ctx.fit(train_df["context"], train_df["label"])
    metrics_ctx = model_ctx.evaluate(test_df["context"], test_df["label"])
    model_ctx.save(models_dir, model_name="tfidf_idiom_context")

    logger.info("Context Baseline -> Acc: %.4f | Macro F1: %.4f | ROC-AUC: %.4f",
                metrics_ctx["accuracy"], metrics_ctx["f1_macro"], metrics_ctx["roc_auc"])

    cm_path_ctx = cm_dir / "idiom_baseline_context_confusion_matrix.png"
    save_confusion_matrix_plot(metrics_ctx["confusion_matrix"],
                                "SemEval Idiom Baseline (Context + Sentence)",
                                cm_path_ctx)

    # Calculate Context Delta
    delta = {
        "accuracy": metrics_ctx["accuracy"] - metrics_sent["accuracy"],
        "precision_macro": metrics_ctx["precision_macro"] - metrics_sent["precision_macro"],
        "recall_macro": metrics_ctx["recall_macro"] - metrics_sent["recall_macro"],
        "f1_macro": metrics_ctx["f1_macro"] - metrics_sent["f1_macro"],
        "roc_auc": metrics_ctx["roc_auc"] - metrics_sent["roc_auc"]
    }

    baseline_metrics = {
        "dataset": "SemEval-2022 Task 2 (EN)",
        "train_size": len(train_df),
        "test_size": len(test_df),
        "seed": 42,
        "sentence_only": metrics_sent,
        "context_plus_sentence": metrics_ctx,
        "context_delta": delta,
        "notes": "Full-data TF-IDF + Logistic Regression baseline on SemEval-2022 Task 2 English dataset."
    }

    metrics_out = metrics_dir / "idiom_baseline_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2)

    logger.info("\nSaved baseline metrics JSON to: %s", metrics_out)


if __name__ == "__main__":
    run_baseline_experiment()
