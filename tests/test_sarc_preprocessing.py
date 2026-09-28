"""
tests/test_sarc_preprocessing.py
=================================
Unit tests for SARC preprocessing pipeline (`src/preprocessing/sarc_preprocessor.py`).

Tests:
1. Text normalization (whitespace, unicode, casing/punctuation preservation)
2. Missing-value and empty comment handling
3. Context text formatting
4. Label validation
5. Deterministic example ID generation
6. Train/validation/test split ratios and disjointness (leakage check)
7. Deterministic seed reproducibility
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.sarc_preprocessor import (
    format_context_text,
    generate_example_id,
    normalize_text,
    process_sarc_dataframe,
)


def test_normalize_text():
    # Whitespace collapsing & stripping
    assert normalize_text("  hello   world! \n ") == "hello world!"
    # Preserve punctuation, casing, emojis
    assert normalize_text("  WOW, really?! 😊 ") == "WOW, really?! 😊"
    # Null / NaN handling
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""


def test_format_context_text():
    ctx = "What a great day!"
    comment = "Yeah, truly amazing."
    formatted = format_context_text(ctx, comment)
    assert formatted == "[CONTEXT] What a great day! [SEP] Yeah, truly amazing."

    # Missing context fallback
    assert format_context_text("", comment) == comment


def test_generate_example_id():
    id1 = generate_example_id("Parent comment", "Sarcastic response", 0)
    id2 = generate_example_id("Parent comment", "Sarcastic response", 0)
    id3 = generate_example_id("Parent comment", "Sarcastic response", 1)

    assert len(id1) == 16
    assert id1 == id2, "ID generation must be deterministic for identical inputs."
    assert id1 != id3, "Different index must produce a different ID."


def test_process_sarc_dataframe_pipeline():
    # Create a synthetic in-memory fixture with 100 rows
    data = []
    for i in range(100):
        label = i % 2
        parent = f"Parent post number {i}"
        comment = f"Comment response text {i}!"
        data.append({"label": label, "comment": comment, "parent_comment": parent})

    # Add duplicate row to test deduplication
    data.append({"label": 0, "comment": "Comment response text 0!", "parent_comment": "Parent post number 0"})

    df = pd.DataFrame(data)

    train_df, val_df, test_df, manifest = process_sarc_dataframe(
        df, random_seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    # 1. Total processed rows check (100 unique rows out of 101)
    assert manifest["processed_row_count"] == 100
    assert manifest["removed_row_count"] == 1

    # 2. Split counts (80 / 10 / 10)
    assert len(train_df) == 80
    assert len(val_df) == 10
    assert len(test_df) == 10

    # 3. Disjointness / Leakage check
    train_ids = set(train_df["example_id"])
    val_ids = set(val_df["example_id"])
    test_ids = set(test_df["example_id"])

    assert train_ids.isdisjoint(val_ids), "Train and validation sets must be disjoint!"
    assert train_ids.isdisjoint(test_ids), "Train and test sets must be disjoint!"
    assert val_ids.isdisjoint(test_ids), "Validation and test sets must be disjoint!"

    # 4. Same example IDs exist in sentence-only and context+sentence representations
    assert "text" in train_df.columns
    assert "context" in train_df.columns
    assert "context_text" in train_df.columns
    assert "label" in train_df.columns

    # 5. Label validation
    assert set(train_df["label"].unique()).issubset({0, 1})
    assert set(val_df["label"].unique()).issubset({0, 1})
    assert set(test_df["label"].unique()).issubset({0, 1})


def test_deterministic_split_seed():
    data = [
        {"label": i % 2, "comment": f"Comment {i}", "parent_comment": f"Parent {i}"}
        for i in range(50)
    ]
    df = pd.DataFrame(data)

    train1, val1, test1, _ = process_sarc_dataframe(df, random_seed=42)
    train2, val2, test2, _ = process_sarc_dataframe(df, random_seed=42)

    assert train1["example_id"].tolist() == train2["example_id"].tolist()
    assert val1["example_id"].tolist() == val2["example_id"].tolist()
    assert test1["example_id"].tolist() == test2["example_id"].tolist()


def test_invalid_labels_raise_error():
    data = [{"label": 5, "comment": "Valid comment", "parent_comment": "Valid parent"}]
    df = pd.DataFrame(data)

    with pytest.raises(ValueError, match="Invalid label set found"):
        process_sarc_dataframe(df)
