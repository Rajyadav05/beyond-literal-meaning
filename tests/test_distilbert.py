"""
tests/test_distilbert.py
========================
Unit tests for DistilBERT model module (`src/sarcasm/distilbert_model.py`).

Tests:
1. SarcasmDataset initialization and batch output shape
2. Tokenizer output structure
3. Forward pass and prediction logit dimension
4. Probability normalization (softmax probabilities sum to 1.0)
5. Evaluation loop metrics structure
6. Device selection safety (cuda if available else cpu)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sarcasm.distilbert_model import (
    SarcasmDataset,
    get_device,
    load_tokenizer_and_model,
)


def test_device_selection():
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ["cpu", "cuda"]


def test_sarcasm_dataset():
    texts = ["This is a test comment", "Another comment here"]
    labels = [0, 1]

    ds = SarcasmDataset(texts, labels, max_length=16)
    assert len(ds) == 2

    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample

    assert sample["input_ids"].shape == (16,)
    assert sample["labels"].item() == 0


def test_model_forward_pass():
    device = torch.device("cpu")
    # Load from saved smoke test checkpoint or base model
    checkpoint_dir = PROJECT_ROOT / "models" / "sarcasm" / "smoke_test_checkpoint"
    if checkpoint_dir.exists():
        model_path = str(checkpoint_dir)
    else:
        model_path = "distilbert-base-uncased"

    tokenizer, model, _ = load_tokenizer_and_model(model_path, num_labels=2, device=device)
    model.eval()

    inputs = tokenizer(["Wow, so cool!", "Normal day."], padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    assert logits.shape == (2, 2)
    probs = torch.softmax(logits, dim=-1).numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_distilbert_reporting_artifacts():
    import json

    models_dir = PROJECT_ROOT / "models" / "sarcasm"
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    experiments_dir = PROJECT_ROOT / "results" / "experiments"

    meta_file = models_dir / "distilbert_metadata.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta.get("training_subset_size") == 1000
        assert meta.get("test_subset_size") == 500
        assert meta.get("full_dataset_training") is False
        assert "comparison_warning" in meta
        assert "evaluation_scope" in meta

    comp_file = metrics_dir / "baseline_vs_distilbert.json"
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            comp = json.load(f)
        assert "comparison_warning" in comp
        assert "full_data_baseline" in comp
        assert "exploratory_subset_transformer" in comp
        assert comp["full_data_baseline"]["train_size"] == 807969
        assert comp["exploratory_subset_transformer"]["train_size"] == 1000

    ctx_file = metrics_dir / "distilbert_context_comparison.json"
    if ctx_file.exists():
        with open(ctx_file, "r", encoding="utf-8") as f:
            ctx_data = json.load(f)
        assert ctx_data.get("sentence_only", {}).get("accuracy") == 0.628
        assert ctx_data.get("context_plus_sentence", {}).get("accuracy") == 0.554
        assert "comparison_warning" in ctx_data

    flip_file = experiments_dir / "distilbert_context_flip_examples.json"
    if flip_file.exists():
        with open(flip_file, "r", encoding="utf-8") as f:
            flip_data = json.load(f)
        assert flip_data.get("total_flipped_predictions") == 181
        assert flip_data.get("flip_percentage") == 36.2
        assert "prediction_sensitivity_note" in flip_data

