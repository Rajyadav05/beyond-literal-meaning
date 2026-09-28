"""
scripts/prepare_sarc.py
=======================
Phase 3 — SARC Dataset Preprocessing Script.

CLI script to execute SARC preprocessing pipeline:
- Loads `datasets/raw/sarc/train-balanced-sarcasm.csv`
- Runs deterministic text normalization & context formatting
- Deduplicates identical pairs
- Performs 80/10/10 stratified split (seed 42)
- Saves `train.parquet`, `validation.parquet`, `test.parquet`
- Generates `datasets/processed/sarc/sarc_preprocessing_manifest.json`

Usage:
    python scripts/prepare_sarc.py [--input PATH] [--output-dir PATH] [--seed INT]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.sarc_preprocessor import process_sarc_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_sarc")


def main():
    parser = argparse.ArgumentParser(description="Preprocess SARC dataset for Non-Literal Language Lab.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw" / "sarc" / "train-balanced-sarcasm.csv",
        help="Path to raw SARC train-balanced-sarcasm.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "sarc",
        help="Directory where processed parquet files and manifest will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val/test split.",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    seed = args.seed

    logger.info("=== Phase 3: SARC Preprocessing Pipeline ===")
    logger.info("Input file:  %s", input_path)
    logger.info("Output dir:  %s", output_dir)
    logger.info("Random seed: %d", seed)

    if not input_path.exists():
        logger.error("Raw input file not found: %s", input_path)
        logger.error("Please place 'train-balanced-sarcasm.csv' in datasets/raw/sarc/ first.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Memory-efficient reading of required columns only
    use_cols = ["label", "comment", "parent_comment"]
    logger.info("Loading required columns %s from raw SARC CSV...", use_cols)
    raw_df = pd.read_csv(input_path, usecols=use_cols)
    logger.info("Raw SARC CSV loaded successfully (%d rows).", len(raw_df))

    # Process dataframe
    train_df, val_df, test_df, manifest = process_sarc_dataframe(
        raw_df,
        random_seed=seed,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    # Save Parquet splits
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "validation.parquet"
    test_path = output_dir / "test.parquet"

    logger.info("Saving processed Parquet files to %s ...", output_dir)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    manifest["output_files"] = {
        "train": {"path": str(train_path), "size_bytes": train_path.stat().st_size, "rows": len(train_df)},
        "validation": {"path": str(val_path), "size_bytes": val_path.stat().st_size, "rows": len(val_df)},
        "test": {"path": str(test_path), "size_bytes": test_path.stat().st_size, "rows": len(test_df)},
    }

    manifest_path = output_dir / "sarc_preprocessing_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest saved: %s", manifest_path)
    logger.info("=== SARC Preprocessing Complete ===")


if __name__ == "__main__":
    main()
