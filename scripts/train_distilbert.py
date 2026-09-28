"""
scripts/train_distilbert.py
============================
Phase 5B — DistilBERT Fine-Tuning & Evaluation Script.

Trains and evaluates sentence-only (Experiment A) and context-aware (Experiment B)
DistilBERT sarcasm classification models on SARC Parquet datasets.

CPU Resource Safety:
- Uses a deterministic 10,000-example stratified training subset and 5,000-example test subset
  to ensure reproducible CPU execution (~3.5 ex/sec throughput).
- Exact same example IDs used for sentence and context experiments.

Outputs:
- Saved models: `models/sarcasm/distilbert_sentence/`, `models/sarcasm/distilbert_context/`
- Metadata: `models/sarcasm/distilbert_metadata.json`
- Metrics & comparison: `results/metrics/distilbert_context_comparison.json`
- Test predictions: `results/predictions/distilbert_sentence_test.parquet`, `results/predictions/distilbert_context_test.parquet`
- Confusion matrices: `results/confusion_matrices/distilbert_sentence_confusion_matrix.png`, `distilbert_context_confusion_matrix.png`
- Context flip analysis: `results/experiments/distilbert_context_flip_examples.json`
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sarcasm.distilbert_model import (
    SarcasmDataset,
    evaluate_distilbert,
    get_device,
    load_tokenizer_and_model,
    train_distilbert_epoch,
)
from src.sarcasm.tfidf_baseline import save_confusion_matrix_plot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_distilbert")


def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT Sarcasm Classifier.")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["both", "sentence", "context"],
        default="both",
        help="Which DistilBERT experiment to run.",
    )
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=5000,
        help="Deterministic training subset size for CPU feasibility.",
    )
    parser.add_argument(
        "--test-subset-size",
        type=int,
        default=2000,
        help="Deterministic test evaluation subset size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset selection and training.",
    )

    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "datasets" / "processed" / "sarc"
    models_dir = PROJECT_ROOT / "models" / "sarcasm"
    results_dir = PROJECT_ROOT / "results"

    metrics_dir = results_dir / "metrics"
    predictions_dir = results_dir / "predictions"
    cm_dir = results_dir / "confusion_matrices"
    experiments_dir = results_dir / "experiments"

    for d in [models_dir, metrics_dir, predictions_dir, cm_dir, experiments_dir]:
        d.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    logger.info("=== Phase 5B: DistilBERT Training & Evaluation ===")
    logger.info("Device:            %s", device)
    logger.info("Experiment:        %s", args.experiment)
    logger.info("Train Subset Size: %d", args.train_subset_size)
    logger.info("Test Subset Size:  %d", args.test_subset_size)
    logger.info("Batch Size:        %d", args.batch_size)
    logger.info("Epochs:            %d", args.epochs)
    logger.info("Learning Rate:     %e", args.lr)
    logger.info("Seed:              %d", seed)

    # Load SARC processed splits
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    # Select deterministic stratified subsets for CPU feasibility
    if args.train_subset_size < len(train_df):
        train_sub, _ = train_test_split(
            train_df,
            train_size=args.train_subset_size,
            random_state=seed,
            stratify=train_df["label"],
        )
    else:
        train_sub = train_df

    if args.test_subset_size < len(test_df):
        test_sub, _ = train_test_split(
            test_df,
            train_size=args.test_subset_size,
            random_state=seed,
            stratify=test_df["label"],
        )
    else:
        test_sub = test_df

    train_sub = train_sub.reset_index(drop=True)
    test_sub = test_sub.reset_index(drop=True)

    logger.info("Deterministic Subsets Selected:")
    logger.info("  Train Subset: %d rows (%s)", len(train_sub), dict(train_sub["label"].value_counts()))
    logger.info("  Test Subset:  %d rows (%s)", len(test_sub), dict(test_sub["label"].value_counts()))

    # Verify ID alignment
    assert set(train_sub["example_id"]).isdisjoint(set(test_sub["example_id"])), "Train and Test IDs overlap!"

    run_sentence = args.experiment in ["both", "sentence"]
    run_context = args.experiment in ["both", "context"]

    sentence_metrics = None
    context_metrics = None
    pred_sent_df = None
    pred_ctx_df = None

    # -----------------------------------------------------------------------
    # EXPERIMENT A: DistilBERT Sentence-Only ('text')
    # -----------------------------------------------------------------------
    if run_sentence:
        logger.info("\n--- Starting DistilBERT Experiment A: Sentence-Only ('text') ---")
        tokenizer_sent, model_sent, _ = load_tokenizer_and_model("distilbert-base-uncased", num_labels=2, device=device)

        train_ds_sent = SarcasmDataset(train_sub["text"].tolist(), train_sub["label"].tolist(), tokenizer=tokenizer_sent, max_length=128)
        test_ds_sent = SarcasmDataset(test_sub["text"].tolist(), test_sub["label"].tolist(), tokenizer=tokenizer_sent, max_length=128)

        train_loader_sent = DataLoader(train_ds_sent, batch_size=args.batch_size, shuffle=True)
        test_loader_sent = DataLoader(test_ds_sent, batch_size=args.batch_size, shuffle=False)

        optimizer_sent = torch.optim.AdamW(model_sent.parameters(), lr=args.lr, weight_decay=0.01)

        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            loss = train_distilbert_epoch(model_sent, train_loader_sent, optimizer_sent, device)
            logger.info("  Sentence Epoch %d/%d - Train Loss: %.4f", epoch, args.epochs, loss)

        sentence_metrics, y_pred_sent, y_prob_sent = evaluate_distilbert(model_sent, test_loader_sent, device)
        sent_duration = time.time() - t0
        sentence_metrics["training_duration_seconds"] = sent_duration

        logger.info("Experiment A Test Acc: %.4f | Macro F1: %.4f | ROC-AUC: %.4f (took %.2fs)",
                    sentence_metrics["accuracy"], sentence_metrics["f1_macro"], sentence_metrics["roc_auc"], sent_duration)

        # Save Model Artifact
        model_sent_dir = models_dir / "distilbert_sentence"
        model_sent_dir.mkdir(parents=True, exist_ok=True)
        model_sent.save_pretrained(model_sent_dir)
        tokenizer_sent.save_pretrained(model_sent_dir)
        logger.info("Saved sentence DistilBERT model to: %s", model_sent_dir)

        # Save Test Predictions Parquet
        pred_sent_df = pd.DataFrame({
            "example_id": test_sub["example_id"],
            "label": test_sub["label"],
            "prediction": y_pred_sent,
            "probability_non_sarcastic": 1.0 - y_prob_sent,
            "probability_sarcastic": y_prob_sent,
        })
        pred_sent_path = predictions_dir / "distilbert_sentence_test.parquet"
        pred_sent_df.to_parquet(pred_sent_path, index=False)
        logger.info("Saved sentence test predictions to: %s", pred_sent_path)

        # Save Confusion Matrix Plot
        cm_sent_path = cm_dir / "distilbert_sentence_confusion_matrix.png"
        save_confusion_matrix_plot(
            sentence_metrics["confusion_matrix"],
            title="DistilBERT Confusion Matrix (Sentence Only)",
            save_path=cm_sent_path,
        )

    # -----------------------------------------------------------------------
    # EXPERIMENT B: DistilBERT Context + Sentence ('context_text')
    # -----------------------------------------------------------------------
    if run_context:
        logger.info("\n--- Starting DistilBERT Experiment B: Context + Sentence ('context_text') ---")
        tokenizer_ctx, model_ctx, _ = load_tokenizer_and_model("distilbert-base-uncased", num_labels=2, device=device)

        train_ds_ctx = SarcasmDataset(train_sub["context_text"].tolist(), train_sub["label"].tolist(), tokenizer=tokenizer_ctx, max_length=128)
        test_ds_ctx = SarcasmDataset(test_sub["context_text"].tolist(), test_sub["label"].tolist(), tokenizer=tokenizer_ctx, max_length=128)

        train_loader_ctx = DataLoader(train_ds_ctx, batch_size=args.batch_size, shuffle=True)
        test_loader_ctx = DataLoader(test_ds_ctx, batch_size=args.batch_size, shuffle=False)

        optimizer_ctx = torch.optim.AdamW(model_ctx.parameters(), lr=args.lr, weight_decay=0.01)

        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            loss = train_distilbert_epoch(model_ctx, train_loader_ctx, optimizer_ctx, device)
            logger.info("  Context Epoch %d/%d - Train Loss: %.4f", epoch, args.epochs, loss)

        context_metrics, y_pred_ctx, y_prob_ctx = evaluate_distilbert(model_ctx, test_loader_ctx, device)
        ctx_duration = time.time() - t0
        context_metrics["training_duration_seconds"] = ctx_duration

        logger.info("Experiment B Test Acc: %.4f | Macro F1: %.4f | ROC-AUC: %.4f (took %.2fs)",
                    context_metrics["accuracy"], context_metrics["f1_macro"], context_metrics["roc_auc"], ctx_duration)

        # Save Model Artifact
        model_ctx_dir = models_dir / "distilbert_context"
        model_ctx_dir.mkdir(parents=True, exist_ok=True)
        model_ctx.save_pretrained(model_ctx_dir)
        tokenizer_ctx.save_pretrained(model_ctx_dir)
        logger.info("Saved context DistilBERT model to: %s", model_ctx_dir)

        # Save Test Predictions Parquet
        pred_ctx_df = pd.DataFrame({
            "example_id": test_sub["example_id"],
            "label": test_sub["label"],
            "prediction": y_pred_ctx,
            "probability_non_sarcastic": 1.0 - y_prob_ctx,
            "probability_sarcastic": y_prob_ctx,
        })
        pred_ctx_path = predictions_dir / "distilbert_context_test.parquet"
        pred_ctx_df.to_parquet(pred_ctx_path, index=False)
        logger.info("Saved context test predictions to: %s", pred_ctx_path)

        # Save Confusion Matrix Plot
        cm_ctx_path = cm_dir / "distilbert_context_confusion_matrix.png"
        save_confusion_matrix_plot(
            context_metrics["confusion_matrix"],
            title="DistilBERT Confusion Matrix (Context + Sentence)",
            save_path=cm_ctx_path,
        )

    # -----------------------------------------------------------------------
    # 3. Side-by-Side Comparison & Context Flip Analysis
    # -----------------------------------------------------------------------
    if sentence_metrics and context_metrics:
        delta = {
            "accuracy": context_metrics["accuracy"] - sentence_metrics["accuracy"],
            "precision_macro": context_metrics["precision_macro"] - sentence_metrics["precision_macro"],
            "recall_macro": context_metrics["recall_macro"] - sentence_metrics["recall_macro"],
            "f1_macro": context_metrics["f1_macro"] - sentence_metrics["f1_macro"],
            "roc_auc": context_metrics["roc_auc"] - sentence_metrics["roc_auc"],
        }
        comp_data = {
            "sentence_only": sentence_metrics,
            "context_plus_sentence": context_metrics,
            "difference_context_minus_sentence": delta,
        }
        comp_path = metrics_dir / "distilbert_context_comparison.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comp_data, f, indent=2)

        # Context Flip Analysis
        flip_cases = []
        for i in range(len(test_sub)):
            sent_p = int(pred_sent_df.iloc[i]["prediction"])
            ctx_p = int(pred_ctx_df.iloc[i]["prediction"])
            if sent_p != ctx_p:
                flip_cases.append({
                    "example_id": test_sub.iloc[i]["example_id"],
                    "true_label": int(test_sub.iloc[i]["label"]),
                    "text": test_sub.iloc[i]["text"],
                    "context": test_sub.iloc[i]["context"],
                    "sentence_prediction": sent_p,
                    "sentence_probability_sarcastic": float(pred_sent_df.iloc[i]["probability_sarcastic"]),
                    "context_prediction": ctx_p,
                    "context_probability_sarcastic": float(pred_ctx_df.iloc[i]["probability_sarcastic"]),
                })

        flip_path = experiments_dir / "distilbert_context_flip_examples.json"
        with open(flip_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_test_examples": len(test_sub),
                "total_flipped_predictions": len(flip_cases),
                "flip_percentage": float(len(flip_cases) / len(test_sub) * 100.0),
                "flip_examples": flip_cases[:25],
            }, f, indent=2)
        logger.info("Saved context flip analysis (%d flips) to: %s", len(flip_cases), flip_path)

    # 4. Save Metadata JSON
    metadata = {
        "base_model": "distilbert-base-uncased",
        "max_length": 128,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": seed,
        "device": str(device),
        "training_subset_size": len(train_sub),
        "test_subset_size": len(test_sub),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_cpu_subset_experiment": True,
        "full_dataset_training_impractical_on_cpu": "64.66 hours estimated for full 807k epoch on CPU",
    }
    meta_path = models_dir / "distilbert_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved DistilBERT metadata to: %s", meta_path)

    logger.info("=== Phase 5B DistilBERT Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
