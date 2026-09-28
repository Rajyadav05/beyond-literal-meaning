"""
scripts/evaluate_distilbert.py
===============================
Phase 5 — Baseline vs DistilBERT 4-Way Comparison Generator.

Reads saved evaluation metrics for:
1. TF-IDF Sentence-Only (Full Data)
2. TF-IDF Context + Sentence (Full Data)
3. DistilBERT Sentence-Only (Exploratory Subset)
4. DistilBERT Context + Sentence (Exploratory Subset)

Generates `results/metrics/baseline_vs_distilbert.json` containing the comparative
benchmark table without inventing metrics, overclaiming, or misrepresenting dataset scopes.

Usage:
    python scripts/evaluate_distilbert.py
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_distilbert")


def main():
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    tfidf_comp_path = metrics_dir / "tfidf_context_comparison.json"
    distilbert_comp_path = metrics_dir / "distilbert_context_comparison.json"

    if not tfidf_comp_path.exists():
        logger.error("TF-IDF comparison metrics missing at: %s", tfidf_comp_path)
        sys.exit(1)

    if not distilbert_comp_path.exists():
        logger.error("DistilBERT comparison metrics missing at: %s", distilbert_comp_path)
        sys.exit(1)

    with open(tfidf_comp_path, "r", encoding="utf-8") as f:
        tfidf_data = json.load(f)

    with open(distilbert_comp_path, "r", encoding="utf-8") as f:
        distilbert_data = json.load(f)

    comparison_matrix = {
        "comparison_warning": (
            "The TF-IDF baseline was trained on 807,969 rows and evaluated on 100,997 test rows, "
            "whereas the DistilBERT model was fine-tuned on a 1,000-example CPU-safe subset and evaluated on 500 test rows. "
            "Absolute metric differences between TF-IDF and DistilBERT reflect dataset scale differences rather than a controlled architecture benchmark."
        ),
        "full_data_baseline": {
            "model_family": "TF-IDF + Logistic Regression",
            "training_scope": "Full Processed SARC Dataset",
            "train_size": 807969,
            "test_size": 100997,
            "sentence_only": {
                "accuracy": tfidf_data["sentence_only"]["accuracy"],
                "f1_macro": tfidf_data["sentence_only"]["f1_macro"],
                "roc_auc": tfidf_data["sentence_only"]["roc_auc"],
            },
            "tfidf_sentence": {
                "accuracy": tfidf_data["sentence_only"]["accuracy"],
                "f1_macro": tfidf_data["sentence_only"]["f1_macro"],
                "roc_auc": tfidf_data["sentence_only"]["roc_auc"],
            },
            "context_plus_sentence": {
                "accuracy": tfidf_data["context_plus_sentence"]["accuracy"],
                "f1_macro": tfidf_data["context_plus_sentence"]["f1_macro"],
                "roc_auc": tfidf_data["context_plus_sentence"]["roc_auc"],
            },
            "tfidf_context": {
                "accuracy": tfidf_data["context_plus_sentence"]["accuracy"],
                "f1_macro": tfidf_data["context_plus_sentence"]["f1_macro"],
                "roc_auc": tfidf_data["context_plus_sentence"]["roc_auc"],
            },
            "context_delta": tfidf_data["difference_context_minus_sentence"],
        },
        "exploratory_subset_transformer": {
            "model_family": "DistilBERT (distilbert-base-uncased)",
            "training_scope": "CPU-Safe Stratified Subset",
            "train_size": distilbert_data.get("training_subset_size", 1000),
            "test_size": distilbert_data.get("test_subset_size", 500),
            "sentence_only": {
                "accuracy": distilbert_data["sentence_only"]["accuracy"],
                "f1_macro": distilbert_data["sentence_only"]["f1_macro"],
                "roc_auc": distilbert_data["sentence_only"]["roc_auc"],
            },
            "distilbert_sentence": {
                "accuracy": distilbert_data["sentence_only"]["accuracy"],
                "f1_macro": distilbert_data["sentence_only"]["f1_macro"],
                "roc_auc": distilbert_data["sentence_only"]["roc_auc"],
            },
            "context_plus_sentence": {
                "accuracy": distilbert_data["context_plus_sentence"]["accuracy"],
                "f1_macro": distilbert_data["context_plus_sentence"]["f1_macro"],
                "roc_auc": distilbert_data["context_plus_sentence"]["roc_auc"],
            },
            "distilbert_context": {
                "accuracy": distilbert_data["context_plus_sentence"]["accuracy"],
                "f1_macro": distilbert_data["context_plus_sentence"]["f1_macro"],
                "roc_auc": distilbert_data["context_plus_sentence"]["roc_auc"],
            },
            "context_delta": distilbert_data["difference_context_minus_sentence"],
        }
    }

    out_path = metrics_dir / "baseline_vs_distilbert.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison_matrix, f, indent=2)

    logger.info("Saved corrected 4-way comparison matrix to: %s", out_path)


if __name__ == "__main__":
    main()
