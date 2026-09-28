"""
scripts/train_idiom_context.py
===============================
Phase 6 Steps 8 & 9 — Contextual DistilBERT Fine-Tuning & Context Ablation for Idiom Classification.

Fine-tunes DistilBERT for:
1. Experiment A: Sentence-Only
2. Experiment B: Context + Sentence

Measures accuracy, macro F1, ROC-AUC, confusion matrices, and exact context deltas.
Saves models to `models/idiom/distilbert_sentence/` and `models/idiom/distilbert_context/`,
and outputs metrics to `results/metrics/idiom_context_metrics.json`.

Usage:
    python scripts/train_idiom_context.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.idiom.classifier import (
    IdiomDataset,
    evaluate_idiom_model,
    get_device,
    load_idiom_model,
    train_idiom_epoch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_idiom_context")


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


def run_idiom_context_experiment(
    batch_size: int = 16,
    epochs: int = 1,
    learning_rate: float = 2e-5,
    subset_train_size: int = 1000,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    logger.info("Device: %s", device)

    semeval_dir = PROJECT_ROOT / "datasets" / "processed" / "semeval_idiom"
    models_dir = PROJECT_ROOT / "models" / "idiom"
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    cm_dir = PROJECT_ROOT / "results" / "confusion_matrices"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(semeval_dir / "train.parquet")
    test_df = pd.read_parquet(semeval_dir / "test.parquet")

    # Sample deterministic subset for CPU training safety
    if subset_train_size and subset_train_size < len(train_df):
        train_sub = train_df.sample(n=subset_train_size, random_state=seed).reset_index(drop=True)
        is_subset = True
    else:
        train_sub = train_df.copy()
        is_subset = False

    logger.info("Training subset size: %d rows (Full Train: %d rows)", len(train_sub), len(train_df))
    logger.info("Test size: %d rows", len(test_df))

    # --- EXPERIMENT A: DistilBERT Sentence-Only ---
    logger.info("\n=== Starting DistilBERT Idiom Experiment A: Sentence-Only ===")
    start_time_sent = time.time()
    tokenizer_sent, model_sent, _ = load_idiom_model("distilbert-base-uncased", num_labels=2, device=device)

    train_ds_sent = IdiomDataset(train_sub["sentence"].tolist(), train_sub["label"].tolist())
    test_ds_sent = IdiomDataset(test_df["sentence"].tolist(), test_df["label"].tolist())

    train_loader_sent = DataLoader(train_ds_sent, batch_size=batch_size, shuffle=True)
    test_loader_sent = DataLoader(test_ds_sent, batch_size=batch_size, shuffle=False)

    optimizer_sent = AdamW(model_sent.parameters(), lr=learning_rate)

    for ep in range(epochs):
        loss_sent = train_idiom_epoch(model_sent, train_loader_sent, optimizer_sent, device)
        logger.info("Epoch %d/%d - Loss: %.4f", ep + 1, epochs, loss_sent)

    metrics_sent, y_pred_sent, y_prob_sent = evaluate_idiom_model(model_sent, test_loader_sent, device)
    time_sent = time.time() - start_time_sent
    metrics_sent["training_duration_seconds"] = round(time_sent, 2)

    logger.info("Sentence DistilBERT -> Acc: %.4f | Macro F1: %.4f | ROC-AUC: %.4f",
                metrics_sent["accuracy"], metrics_sent["f1_macro"], metrics_sent["roc_auc"])

    # Save Sentence Model
    model_sent_dir = models_dir / "distilbert_sentence"
    model_sent_dir.mkdir(parents=True, exist_ok=True)
    model_sent.save_pretrained(model_sent_dir)
    tokenizer_sent.save_pretrained(model_sent_dir)
    logger.info("Saved sentence DistilBERT model to: %s", model_sent_dir)

    cm_path_sent = cm_dir / "distilbert_idiom_sentence_confusion_matrix.png"
    save_confusion_matrix_plot(metrics_sent["confusion_matrix"], "DistilBERT Idiom (Sentence Only)", cm_path_sent)

    # --- EXPERIMENT B: DistilBERT Context + Sentence ---
    logger.info("\n=== Starting DistilBERT Idiom Experiment B: Context + Sentence ===")
    start_time_ctx = time.time()
    tokenizer_ctx, model_ctx, _ = load_idiom_model("distilbert-base-uncased", num_labels=2, device=device)

    train_ds_ctx = IdiomDataset(train_sub["context"].tolist(), train_sub["label"].tolist())
    test_ds_ctx = IdiomDataset(test_df["context"].tolist(), test_df["label"].tolist())

    train_loader_ctx = DataLoader(train_ds_ctx, batch_size=batch_size, shuffle=True)
    test_loader_ctx = DataLoader(test_ds_ctx, batch_size=batch_size, shuffle=False)

    optimizer_ctx = AdamW(model_ctx.parameters(), lr=learning_rate)

    for ep in range(epochs):
        loss_ctx = train_idiom_epoch(model_ctx, train_loader_ctx, optimizer_ctx, device)
        logger.info("Epoch %d/%d - Loss: %.4f", ep + 1, epochs, loss_ctx)

    metrics_ctx, y_pred_ctx, y_prob_ctx = evaluate_idiom_model(model_ctx, test_loader_ctx, device)
    time_ctx = time.time() - start_time_ctx
    metrics_ctx["training_duration_seconds"] = round(time_ctx, 2)

    logger.info("Context DistilBERT -> Acc: %.4f | Macro F1: %.4f | ROC-AUC: %.4f",
                metrics_ctx["accuracy"], metrics_ctx["f1_macro"], metrics_ctx["roc_auc"])

    # Save Context Model
    model_ctx_dir = models_dir / "distilbert_context"
    model_ctx_dir.mkdir(parents=True, exist_ok=True)
    model_ctx.save_pretrained(model_ctx_dir)
    tokenizer_ctx.save_pretrained(model_ctx_dir)
    logger.info("Saved context DistilBERT model to: %s", model_ctx_dir)

    cm_path_ctx = cm_dir / "distilbert_idiom_context_confusion_matrix.png"
    save_confusion_matrix_plot(metrics_ctx["confusion_matrix"], "DistilBERT Idiom (Context + Sentence)", cm_path_ctx)

    # Context Ablation Delta
    delta = {
        "accuracy": metrics_ctx["accuracy"] - metrics_sent["accuracy"],
        "precision_macro": metrics_ctx["precision_macro"] - metrics_sent["precision_macro"],
        "recall_macro": metrics_ctx["recall_macro"] - metrics_sent["recall_macro"],
        "f1_macro": metrics_ctx["f1_macro"] - metrics_sent["f1_macro"],
        "roc_auc": metrics_ctx["roc_auc"] - metrics_sent["roc_auc"]
    }

    results = {
        "dataset": "SemEval-2022 Task 2 (EN)",
        "model": "distilbert-base-uncased",
        "training_subset_size": len(train_sub),
        "test_subset_size": len(test_df),
        "random_seed": seed,
        "device": str(device),
        "experiment_type": "exploratory_subset" if is_subset else "full_data",
        "sentence_only": metrics_sent,
        "context_plus_sentence": metrics_ctx,
        "difference_context_minus_sentence": delta,
        "comparison_warning": "Evaluated on SemEval-2022 Task 2 test set. Sentence-only and Context+Sentence use identical split and seed."
    }

    out_metrics = metrics_dir / "idiom_context_metrics.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save metadata
    meta = {
        "base_model": "distilbert-base-uncased",
        "seed": seed,
        "device": str(device),
        "training_subset_size": len(train_sub),
        "test_subset_size": len(test_df),
        "experiment_type": "exploratory_subset" if is_subset else "full_data"
    }
    with open(models_dir / "idiom_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("\nSaved idiom context comparison metrics to: %s", out_metrics)


if __name__ == "__main__":
    run_idiom_context_experiment()
