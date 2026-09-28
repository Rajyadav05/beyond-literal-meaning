"""
scripts/smoke_test_distilbert.py
================================
Phase 5A — DistilBERT Pipeline Smoke Test & CPU Feasibility Benchmark.

Verifies end-to-end functionality of DistilBERT:
1. Tokenizer download & loading
2. Model download & loading (`distilbert-base-uncased`)
3. Tokenization
4. Dataset formatting (SarcasmDataset)
5. Forward pass
6. Loss calculation
7. Backward pass
8. Optimizer step
9. Checkpoint saving
10. Model reload
11. Evaluation
12. Prediction & Probability check
13. CPU Throughput Feasibility Benchmark (examples/sec)

Usage:
    python scripts/smoke_test_distilbert.py
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sarcasm.distilbert_model import (
    SarcasmDataset,
    evaluate_distilbert,
    get_device,
    load_tokenizer_and_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smoke_test_distilbert")


def main():
    logger.info("=== Phase 5A: DistilBERT Pipeline Smoke Test ===")
    device = get_device()
    logger.info("Environment Device Check:")
    logger.info("  CUDA Available: %s", torch.cuda.is_available())
    logger.info("  Active Device:  %s", device)

    # 1. Load Tokenizer & Model
    model_name = "distilbert-base-uncased"
    tokenizer, model, _ = load_tokenizer_and_model(model_name, num_labels=2, device=device)
    logger.info("  [1/12 PASS] Tokenizer loaded.")
    logger.info("  [2/12 PASS] Model loaded (%s).", model_name)

    # 2. Tokenization Check
    sample_text = "[CONTEXT] Flight was delayed by 5 hours. [SEP] Wow, wonderful service!"
    tokens = tokenizer(sample_text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
    assert "input_ids" in tokens and "attention_mask" in tokens
    logger.info("  [3/12 PASS] Tokenization verified.")

    # 3. Dataset Formatting
    test_texts = [
        "This is a fantastic product.",
        "Oh awesome, another flat tire on the highway.",
        "What a great day for a picnic.",
        "Exactly what I needed right now.",
    ]
    test_labels = [0, 1, 0, 1]
    dataset = SarcasmDataset(test_texts, test_labels, tokenizer=tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    assert len(dataset) == 4
    logger.info("  [4/12 PASS] SarcasmDataset formatting verified.")

    # 4. Forward Pass & Loss Calculation
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    logits = outputs.logits

    assert loss is not None and logits.shape == (2, 2)
    logger.info("  [5/12 PASS] Forward pass verified.")
    logger.info("  [6/12 PASS] Loss calculation verified (loss = %.4f).", loss.item())

    # 5. Backward Pass & Optimizer Step
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logger.info("  [7/12 PASS] Backward pass verified.")
    logger.info("  [8/12 PASS] Optimizer step verified.")

    # 6. Checkpoint Save & Reload
    checkpoint_dir = PROJECT_ROOT / "models" / "sarcasm" / "smoke_test_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info("  [9/12 PASS] Checkpoint saved to: %s", checkpoint_dir)

    reloaded_tok, reloaded_model, _ = load_tokenizer_and_model(str(checkpoint_dir), num_labels=2, device=device)
    logger.info("  [10/12 PASS] Checkpoint reloaded successfully.")

    # 7. Evaluation & Prediction Check
    metrics, y_pred, y_prob = evaluate_distilbert(reloaded_model, dataloader, device)
    assert "accuracy" in metrics and "f1_macro" in metrics
    assert len(y_pred) == 4 and len(y_prob) == 4
    logger.info("  [11/12 PASS] Evaluation loop verified.")
    logger.info("  [12/12 PASS] Predictions & probabilities verified.")

    # 8. Throughput Feasibility Benchmark (CPU Speed Measurement)
    logger.info("\n--- Running CPU Throughput Benchmark ---")
    benchmark_texts = [f"Sample sentence text number {i} for benchmarking" for i in range(100)]
    benchmark_labels = [i % 2 for i in range(100)]
    bench_ds = SarcasmDataset(benchmark_texts, benchmark_labels, tokenizer=tokenizer, max_length=128)
    bench_loader = DataLoader(bench_ds, batch_size=4, shuffle=True)

    t0 = time.time()
    model.train()
    for b in bench_loader:
        b_ids = b["input_ids"].to(device)
        b_mask = b["attention_mask"].to(device)
        b_lbls = b["labels"].to(device)
        optimizer.zero_grad()
        out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_lbls)
        out.loss.backward()
        optimizer.step()
    elapsed = time.time() - t0

    throughput = len(benchmark_texts) / max(0.001, elapsed)
    est_full_epoch_hours = (807969 / throughput) / 3600.0

    logger.info("Benchmark Results:")
    logger.info("  Processed 100 examples in %.2f seconds.", elapsed)
    logger.info("  CPU Training Throughput: %.2f examples/second", throughput)
    logger.info("  Estimated time for 1 full epoch on 807k SARC rows: %.2f hours", est_full_epoch_hours)

    print("\n" + "=" * 56)
    print("  DistilBERT Smoke Test Result: ALL 12 CHECKS PASSED")
    print(f"  CPU Training Speed: {throughput:.2f} ex/sec")
    print(f"  Estimated Full Epoch Time: {est_full_epoch_hours:.2f} hours")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
