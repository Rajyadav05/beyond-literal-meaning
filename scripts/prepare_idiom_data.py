"""
scripts/prepare_idiom_data.py
=============================
Phase 6 Step 3 — Prepares processed Parquet datasets for MAGPIE and SemEval Task 2.

Usage:
    python scripts/prepare_idiom_data.py
"""

import json
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.idiom.idiom_preprocessor import load_and_clean_magpie, load_and_clean_semeval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_idiom_data")


def prepare_magpie():
    logger.info("=== Preprocessing MAGPIE Dataset ===")
    raw_file = PROJECT_ROOT / "datasets" / "raw" / "magpie" / "MAGPIE_filtered_split_random.jsonl"
    out_dir = PROJECT_ROOT / "datasets" / "processed" / "magpie"
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_df = load_and_clean_magpie(raw_file)
    logger.info("Total cleaned MAGPIE rows: %d", len(cleaned_df))

    train_df = cleaned_df[cleaned_df["split"] == "training"].copy()
    dev_df = cleaned_df[cleaned_df["split"] == "development"].copy()
    test_df = cleaned_df[cleaned_df["split"] == "test"].copy()

    logger.info("MAGPIE Splits -> Train: %d, Dev: %d, Test: %d", len(train_df), len(dev_df), len(test_df))

    train_df.to_parquet(out_dir / "train.parquet", index=False)
    dev_df.to_parquet(out_dir / "dev.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)
    logger.info("Saved MAGPIE Parquet files to: %s", out_dir)

    return {
        "train_rows": len(train_df),
        "dev_rows": len(dev_df),
        "test_rows": len(test_df),
        "label_distribution_train": train_df["label"].value_counts().to_dict(),
        "label_distribution_test": test_df["label"].value_counts().to_dict()
    }


def prepare_semeval():
    logger.info("\n=== Preprocessing SemEval-2022 Task 2 Dataset ===")
    raw_file = PROJECT_ROOT / "datasets" / "raw" / "semeval_idiom" / "train_zero_shot.csv"
    out_dir = PROJECT_ROOT / "datasets" / "processed" / "semeval_idiom"
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_df = load_and_clean_semeval(raw_file, language="EN")
    logger.info("Total cleaned SemEval EN rows: %d", len(cleaned_df))

    # Perform stratified 80/10/10 split with seed 42
    train_df, temp_df = train_test_split(
        cleaned_df, test_size=0.20, random_state=42, stratify=cleaned_df["label"]
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )

    logger.info("SemEval Splits (80/10/10) -> Train: %d, Dev: %d, Test: %d", len(train_df), len(dev_df), len(test_df))

    train_df.to_parquet(out_dir / "train.parquet", index=False)
    dev_df.to_parquet(out_dir / "dev.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)
    logger.info("Saved SemEval Parquet files to: %s", out_dir)

    return {
        "train_rows": len(train_df),
        "dev_rows": len(dev_df),
        "test_rows": len(test_df),
        "label_distribution_train": train_df["label"].value_counts().to_dict(),
        "label_distribution_test": test_df["label"].value_counts().to_dict()
    }


def main():
    magpie_stats = prepare_magpie()
    semeval_stats = prepare_semeval()

    manifest_path = PROJECT_ROOT / "datasets" / "processed" / "idiom_preprocessing_manifest.json"
    manifest = {
        "seed": 42,
        "magpie": magpie_stats,
        "semeval2022_task2": semeval_stats
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("\nSaved preprocessing manifest to: %s", manifest_path)


if __name__ == "__main__":
    main()
