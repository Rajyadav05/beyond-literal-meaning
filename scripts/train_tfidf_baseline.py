"""
scripts/train_tfidf_baseline.py
================================
Phase 4 — TF-IDF + Logistic Regression Baseline Training & Evaluation.

Trains and evaluates sentence-only (Experiment A) and context-aware (Experiment B)
sarcasm classification baselines on processed SARC Parquet datasets.

Generates:
- Joblib model weights in `models/sarcasm/`
- Metadata JSON in `models/sarcasm/tfidf_baseline_metadata.json`
- Metrics & side-by-side comparison in `results/metrics/tfidf_context_comparison.json`
- Test predictions Parquet files in `results/predictions/`
- Confusion matrix plots in `results/confusion_matrices/`
- Feature importance JSON in `results/experiments/tfidf_feature_importance.json`

Usage:
    python scripts/train_tfidf_baseline.py [--experiment both|sentence|context] [--seed 42]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sarcasm.tfidf_baseline import (
    build_tfidf_pipeline,
    extract_top_features,
    save_confusion_matrix_plot,
    train_and_evaluate_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_tfidf_baseline")


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression Sarcasm Baselines.")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["both", "sentence", "context"],
        default="both",
        help="Which baseline experiment to run (sentence-only, context-aware, or both).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "sarc",
        help="Path to processed SARC parquet datasets directory.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "sarcasm",
        help="Directory to save joblib model artifacts.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Root results directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model training.",
    )

    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    models_dir = args.models_dir.resolve()
    results_dir = args.results_dir.resolve()
    seed = args.seed
    experiment_mode = args.experiment

    metrics_dir = results_dir / "metrics"
    predictions_dir = results_dir / "predictions"
    cm_dir = results_dir / "confusion_matrices"
    experiments_dir = results_dir / "experiments"

    for d in [models_dir, metrics_dir, predictions_dir, cm_dir, experiments_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 4: TF-IDF + Logistic Regression Baseline Training ===")
    logger.info("Data Dir:    %s", data_dir)
    logger.info("Models Dir:  %s", models_dir)
    logger.info("Results Dir: %s", results_dir)
    logger.info("Experiment:  %s", experiment_mode)
    logger.info("Random Seed: %d", seed)

    # 1. Load Processed Parquet Datasets
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "validation.parquet"
    test_path = data_dir / "test.parquet"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            logger.error("Required dataset file missing: %s", p)
            logger.error("Run 'scripts/prepare_sarc.py' first.")
            sys.exit(1)

    logger.info("Loading Parquet files...")
    start_load = time.time()
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    load_time = time.time() - start_load
    logger.info("Loaded train (%d), val (%d), test (%d) in %.2fs", len(train_df), len(val_df), len(test_df), load_time)

    # 2. Verify Disjointness Across Splits
    train_ids = set(train_df["example_id"])
    val_ids = set(val_df["example_id"])
    test_ids = set(test_df["example_id"])

    assert train_ids.isdisjoint(val_ids), "Train and Validation example IDs overlap!"
    assert train_ids.isdisjoint(test_ids), "Train and Test example IDs overlap!"
    assert val_ids.isdisjoint(test_ids), "Validation and Test example IDs overlap!"
    logger.info("Disjointness check passed: 100%% unique example IDs across splits.")

    results_summary = {}
    feature_importance_data = {}

    run_sentence = experiment_mode in ["both", "sentence"]
    run_context = experiment_mode in ["both", "context"]

    sentence_metrics = None
    context_metrics = None

    # -----------------------------------------------------------------------
    # EXPERIMENT A: Sentence-Only (text)
    # -----------------------------------------------------------------------
    if run_sentence:
        logger.info("\n--- Starting Experiment A: Sentence-Only ('text') ---")
        pipeline_sentence = build_tfidf_pipeline(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            max_features=300000,
            sublinear_tf=True,
            random_state=seed,
        )

        t0 = time.time()
        pipeline_sentence, sentence_metrics, y_pred_sent, y_prob_sent = train_and_evaluate_model(
            pipeline_sentence,
            train_text=train_df["text"],
            train_labels=train_df["label"],
            eval_text=test_df["text"],
            eval_labels=test_df["label"],
            experiment_name="sentence_only",
        )
        sent_duration = time.time() - t0
        sentence_metrics["training_duration_seconds"] = sent_duration
        logger.info("Experiment A Test Accuracy: %.4f | Macro F1: %.4f | ROC-AUC: %.4f (took %.2fs)",
                    sentence_metrics["accuracy"], sentence_metrics["f1_macro"], sentence_metrics["roc_auc"], sent_duration)

        # Save Sentence Model Artifact
        model_sent_path = models_dir / "tfidf_logreg_sentence.joblib"
        joblib.dump(pipeline_sentence, model_sent_path)
        logger.info("Saved sentence model to: %s", model_sent_path)

        # Save Sentence Test Predictions
        pred_sent_df = pd.DataFrame({
            "example_id": test_df["example_id"],
            "label": test_df["label"],
            "prediction": y_pred_sent,
            "probability_non_sarcastic": 1.0 - y_prob_sent,
            "probability_sarcastic": y_prob_sent,
        })
        pred_sent_path = predictions_dir / "tfidf_sentence_test.parquet"
        pred_sent_df.to_parquet(pred_sent_path, index=False)
        logger.info("Saved sentence test predictions to: %s", pred_sent_path)

        # Save Confusion Matrix Plot
        cm_sent_path = cm_dir / "tfidf_sentence_confusion_matrix.png"
        save_confusion_matrix_plot(
            sentence_metrics["confusion_matrix"],
            title="TF-IDF Baseline Confusion Matrix (Sentence Only)",
            save_path=cm_sent_path,
        )
        logger.info("Saved sentence confusion matrix plot to: %s", cm_sent_path)

        # Extract Top Features
        feature_importance_data["sentence_only"] = extract_top_features(pipeline_sentence, top_k=25)
        results_summary["sentence_only"] = sentence_metrics

    # -----------------------------------------------------------------------
    # EXPERIMENT B: Context + Sentence (context_text)
    # -----------------------------------------------------------------------
    if run_context:
        logger.info("\n--- Starting Experiment B: Context + Sentence ('context_text') ---")
        pipeline_context = build_tfidf_pipeline(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            max_features=300000,
            sublinear_tf=True,
            random_state=seed,
        )

        t0 = time.time()
        pipeline_context, context_metrics, y_pred_ctx, y_prob_ctx = train_and_evaluate_model(
            pipeline_context,
            train_text=train_df["context_text"],
            train_labels=train_df["label"],
            eval_text=test_df["context_text"],
            eval_labels=test_df["label"],
            experiment_name="context_plus_sentence",
        )
        ctx_duration = time.time() - t0
        context_metrics["training_duration_seconds"] = ctx_duration
        logger.info("Experiment B Test Accuracy: %.4f | Macro F1: %.4f | ROC-AUC: %.4f (took %.2fs)",
                    context_metrics["accuracy"], context_metrics["f1_macro"], context_metrics["roc_auc"], ctx_duration)

        # Save Context Model Artifact
        model_ctx_path = models_dir / "tfidf_logreg_context.joblib"
        joblib.dump(pipeline_context, model_ctx_path)
        logger.info("Saved context model to: %s", model_ctx_path)

        # Save Context Test Predictions
        pred_ctx_df = pd.DataFrame({
            "example_id": test_df["example_id"],
            "label": test_df["label"],
            "prediction": y_pred_ctx,
            "probability_non_sarcastic": 1.0 - y_prob_ctx,
            "probability_sarcastic": y_prob_ctx,
        })
        pred_ctx_path = predictions_dir / "tfidf_context_test.parquet"
        pred_ctx_df.to_parquet(pred_ctx_path, index=False)
        logger.info("Saved context test predictions to: %s", pred_ctx_path)

        # Save Confusion Matrix Plot
        cm_ctx_path = cm_dir / "tfidf_context_confusion_matrix.png"
        save_confusion_matrix_plot(
            context_metrics["confusion_matrix"],
            title="TF-IDF Baseline Confusion Matrix (Context + Sentence)",
            save_path=cm_ctx_path,
        )
        logger.info("Saved context confusion matrix plot to: %s", cm_ctx_path)

        # Extract Top Features
        feature_importance_data["context_plus_sentence"] = extract_top_features(pipeline_context, top_k=25)
        results_summary["context_plus_sentence"] = context_metrics

    # -----------------------------------------------------------------------
    # 3. Compute Side-by-Side Context Comparison & Difference
    # -----------------------------------------------------------------------
    comparison_report = {}
    if sentence_metrics and context_metrics:
        delta = {
            "accuracy": context_metrics["accuracy"] - sentence_metrics["accuracy"],
            "precision_macro": context_metrics["precision_macro"] - sentence_metrics["precision_macro"],
            "recall_macro": context_metrics["recall_macro"] - sentence_metrics["recall_macro"],
            "f1_macro": context_metrics["f1_macro"] - sentence_metrics["f1_macro"],
            "roc_auc": context_metrics["roc_auc"] - sentence_metrics["roc_auc"],
        }
        comparison_report = {
            "sentence_only": sentence_metrics,
            "context_plus_sentence": context_metrics,
            "difference_context_minus_sentence": delta,
        }
        comp_path = metrics_dir / "tfidf_context_comparison.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison_report, f, indent=2)
        logger.info("\nSaved side-by-side context comparison to: %s", comp_path)

    # 4. Save Feature Importance JSON
    feat_path = experiments_dir / "tfidf_feature_importance.json"
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feature_importance_data, f, indent=2)
    logger.info("Saved feature coefficients to: %s", feat_path)

    # 5. Save Model Baseline Metadata JSON
    metadata = {
        "model_type": "TF-IDF + Logistic Regression Baseline",
        "random_seed": seed,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_row_count": len(train_df),
        "validation_row_count": len(val_df),
        "test_row_count": len(test_df),
        "vectorizer_configuration": {
            "lowercase": True,
            "ngram_range": [1, 2],
            "min_df": 3,
            "max_df": 0.95,
            "sublinear_tf": True,
            "max_features": 300000,
        },
        "classifier_configuration": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": None,
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "experiments_run": list(results_summary.keys()),
    }
    meta_path = models_dir / "tfidf_baseline_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved baseline metadata to: %s", meta_path)

    logger.info("=== Phase 4 TF-IDF Baseline Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
