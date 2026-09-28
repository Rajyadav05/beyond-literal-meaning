"""
src/sarcasm/distilbert_model.py
===============================
PyTorch & Hugging Face Transformers module for DistilBERT Sarcasm Classification.

Includes:
- SarcasmDataset (PyTorch Dataset for text & context_text tokenization)
- Model & Tokenizer initializer helpers (`distilbert-base-uncased`)
- Custom training & evaluation loops optimized for CPU/GPU device safety
- Prediction and probability extraction functions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("distilbert_model")

DEFAULT_MODEL_NAME = "distilbert-base-uncased"


def get_device() -> torch.device:
    """Returns 'cuda' if GPU is available, else 'cpu'."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SarcasmDataset(Dataset):
    """
    PyTorch Dataset for tokenizing SARC text/context sequences for DistilBERT.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: int = 128,
    ):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.max_length = max_length

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        else:
            self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding: BatchEncoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def load_tokenizer_and_model(
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    num_labels: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel, torch.device]:
    """
    Loads pretrained or fine-tuned DistilBERT tokenizer and classification model.
    """
    if device is None:
        device = get_device()

    logger.info("Loading tokenizer and model '%s' onto device: %s...", model_name_or_path, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
    )
    model.to(device)
    return tokenizer, model, device


def train_distilbert_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Runs one training epoch over DataLoader.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def evaluate_distilbert(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluates DistilBERT model on a DataLoader split.

    Returns:
    (metrics_dict, y_pred, y_prob)
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = np.argmax(logits.cpu().numpy(), axis=-1)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro"))
    rec_macro = float(recall_score(y_true, y_pred, average="macro"))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))

    prec_0 = float(precision_score(y_true, y_pred, pos_label=0))
    rec_0 = float(recall_score(y_true, y_pred, pos_label=0))
    f1_0 = float(f1_score(y_true, y_pred, pos_label=0))

    prec_1 = float(precision_score(y_true, y_pred, pos_label=1))
    rec_1 = float(recall_score(y_true, y_pred, pos_label=1))
    f1_1 = float(f1_score(y_true, y_pred, pos_label=1))

    roc_auc = float(roc_auc_score(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "per_class": {
            "non_sarcastic_0": {"precision": prec_0, "recall": rec_0, "f1": f1_0, "support": int((y_true == 0).sum())},
            "sarcastic_1": {"precision": prec_1, "recall": rec_1, "f1": f1_1, "support": int((y_true == 1).sum())},
        },
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

    return metrics, y_pred, y_prob
