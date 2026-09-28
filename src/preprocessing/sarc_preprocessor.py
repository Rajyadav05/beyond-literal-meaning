"""
src/preprocessing/sarc_preprocessor.py
=======================================
Deterministic, leakage-aware preprocessing pipeline for the Self-Annotated
Reddit Corpus (SARC) dataset.

This module loads the raw SARC CSV (`train-balanced-sarcasm.csv`), performs
text normalization, constructs sentence-only and context+sentence representations,
analyzes duplicate pairs, generates deterministic example IDs, and splits the data
into stratified train/validation/test sets (80%/10%/10%) using seed 42.
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("sarc_preprocessor")


def normalize_text(text: Any) -> str:
    """
    Normalizes input text while preserving surface sarcasm signals (punctuation, casing, emojis).
    - Converts null/NaN to empty string
    - Performs Unicode NFKC normalization
    - Strips leading/trailing whitespace
    - Collapses repeated internal whitespace (newlines/tabs -> single spaces)
    """
    if pd.isna(text) or text is None:
        return ""

    s = str(text)
    # Unicode normalization
    s = unicodedata.normalize("NFKC", s)
    # Remove null / control characters except space
    s = "".join(ch for ch in s if ch == " " or not unicodedata.category(ch).startswith("C"))
    # Collapse multiple whitespace (spaces, tabs, newlines) into single space
    s = " ".join(s.split())
    return s.strip()


def generate_example_id(parent_comment: str, comment: str, original_idx: int) -> str:
    """
    Generates a deterministic 16-character hex hash ID for an example.
    Uses SHA256 of (parent_comment, comment, original_idx).
    """
    raw_key = f"{parent_comment}|||{comment}|||{original_idx}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]


def format_context_text(parent_comment: str, comment: str) -> str:
    """
    Formats combined context and comment text deterministically.
    If context is missing, returns only comment.
    Format: [CONTEXT] <parent_comment> [SEP] <comment>
    """
    if parent_comment:
        return f"[CONTEXT] {parent_comment} [SEP] {comment}"
    return comment


def process_sarc_dataframe(
    df: pd.DataFrame,
    random_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Main preprocessing pipeline for SARC dataframe.

    Steps:
    1. Validate required columns ('label', 'comment', 'parent_comment').
    2. Normalize text fields.
    3. Filter out rows where comment becomes empty after cleaning.
    4. Compute duplicate statistics and perform deduplication on exact (parent_comment, comment) pairs.
    5. Construct example_id, text, context, context_text, and label.
    6. Perform stratified 80/10/10 train/val/test split using fixed random seed.
    7. Perform leakage checks to verify zero overlap across splits.

    Returns:
    (train_df, val_df, test_df, metadata_manifest)
    """
    # 1. Column validation
    required_cols = {"label", "comment", "parent_comment"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    initial_row_count = len(df)
    logger.info("Processing SARC dataset with %d initial rows...", initial_row_count)

    # 2. Text normalization
    logger.info("Normalizing text fields...")
    df["comment_clean"] = df["comment"].apply(normalize_text)
    df["parent_clean"] = df["parent_comment"].apply(normalize_text)

    # Convert label to integer
    df["label"] = df["label"].astype(int)
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Invalid label set found: {unique_labels}. Expected only {{0, 1}}.")

    # 3. Handle missing/empty comments
    empty_comments_mask = df["comment_clean"] == ""
    empty_comments_count = int(empty_comments_mask.sum())
    if empty_comments_count > 0:
        logger.warning("Removing %d rows with empty comment text.", empty_comments_count)
        df = df[~empty_comments_mask].copy()

    # Reset index for clean tracking
    df = df.reset_index(drop=True)

    # 4. Duplicate and Leakage Analysis
    comment_duplicates = int(df.duplicated(subset=["comment_clean"]).sum())
    pair_duplicates = int(df.duplicated(subset=["parent_clean", "comment_clean"]).sum())

    logger.info("Duplicate analysis:")
    logger.info("  Exact duplicate comments: %d", comment_duplicates)
    logger.info("  Exact duplicate (parent, comment) pairs: %d", pair_duplicates)

    # Deduplicate on exact (parent_clean, comment_clean) pairs to prevent leakage
    df_dedup = df.drop_duplicates(subset=["parent_clean", "comment_clean"], keep="first").copy()
    dedup_removed_count = len(df) - len(df_dedup)
    if dedup_removed_count > 0:
        logger.info("Deduplicated dataset: removed %d duplicate (parent, comment) pairs.", dedup_removed_count)

    df_dedup = df_dedup.reset_index(drop=True)

    # 5. Build final features & example IDs
    df_dedup["example_id"] = [
        generate_example_id(parent, comm, idx)
        for idx, (parent, comm) in enumerate(zip(df_dedup["parent_clean"], df_dedup["comment_clean"]))
    ]
    df_dedup["text"] = df_dedup["comment_clean"]
    df_dedup["context"] = df_dedup["parent_clean"]
    df_dedup["context_text"] = [
        format_context_text(p, c) for p, c in zip(df_dedup["parent_clean"], df_dedup["comment_clean"])
    ]

    final_cols = ["example_id", "text", "context", "context_text", "label"]
    df_final = df_dedup[final_cols].copy()

    # 6. Stratified Train / Validation / Test Split (80 / 10 / 10)
    val_test_ratio = val_ratio + test_ratio  # 0.20
    test_portion_of_val_test = test_ratio / val_test_ratio  # 0.50

    train_df, val_test_df = train_test_split(
        df_final,
        test_size=val_test_ratio,
        random_state=random_seed,
        stratify=df_final["label"],
    )

    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_portion_of_val_test,
        random_state=random_seed,
        stratify=val_test_df["label"],
    )

    # Reset split indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # 7. Leakage Verification
    train_ids = set(train_df["example_id"])
    val_ids = set(val_df["example_id"])
    test_ids = set(test_df["example_id"])

    train_val_overlap = len(train_ids.intersection(val_ids))
    train_test_overlap = len(train_ids.intersection(test_ids))
    val_test_overlap = len(val_ids.intersection(test_ids))

    if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
        raise RuntimeError("Leakage verification failed! Example IDs overlap across splits.")

    logger.info("Split completion:")
    logger.info("  Train rows: %d (%s)", len(train_df), dict(train_df["label"].value_counts()))
    logger.info("  Val rows:   %d (%s)", len(val_df), dict(val_df["label"].value_counts()))
    logger.info("  Test rows:  %d (%s)", len(test_df), dict(test_df["label"].value_counts()))
    logger.info("  Leakage check: Train/Val/Test are 100%% disjoint.")

    manifest = {
        "dataset": "SARC",
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "source_row_count": initial_row_count,
        "processed_row_count": len(df_final),
        "removed_row_count": initial_row_count - len(df_final),
        "removal_reasons": {
            "empty_comments": empty_comments_count,
            "duplicate_pairs": dedup_removed_count,
        },
        "columns_used": ["label", "comment", "parent_comment"],
        "split_ratios": {"train": train_ratio, "validation": val_ratio, "test": test_ratio},
        "split_counts": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "label_distribution": {
            "total": {str(k): int(v) for k, v in df_final["label"].value_counts().to_dict().items()},
            "train": {str(k): int(v) for k, v in train_df["label"].value_counts().to_dict().items()},
            "validation": {str(k): int(v) for k, v in val_df["label"].value_counts().to_dict().items()},
            "test": {str(k): int(v) for k, v in test_df["label"].value_counts().to_dict().items()},
        },
        "duplicate_statistics": {
            "comment_duplicates": comment_duplicates,
            "pair_duplicates": pair_duplicates,
        },
        "leakage_checks": {
            "train_val_overlap": train_val_overlap,
            "train_test_overlap": train_test_overlap,
            "val_test_overlap": val_test_overlap,
            "passed": True,
        },
        "cleaning_rules": [
            "Unicode NFKC normalization",
            "Preserved punctuation, casing, contractions, and emojis",
            "Collapsed internal whitespace and stripped outer whitespace",
            "Removed rows with empty comment text",
            "Deduplicated identical (parent_comment, comment) pairs",
        ],
    }

    return train_df, val_df, test_df, manifest
