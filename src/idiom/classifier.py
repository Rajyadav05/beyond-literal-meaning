"""
src/idiom/classifier.py
========================
Phase 6 Contextual Idiom Classifier using DistilBERT.

Fine-tunes a local `distilbert-base-uncased` model to classify whether a target sentence
(or context + target sentence) contains an idiomatic usage (1) vs literal usage (0).
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("idiom_classifier")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IdiomDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_idiom_model(model_name_or_path: str = "distilbert-base-uncased", num_labels: int = 2, device: Optional[torch.device] = None):
    if device is None:
        device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    model.to(device)
    return tokenizer, model, device


def train_idiom_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


def evaluate_idiom_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_probs.extend(probs[:, 1])
            all_labels.extend(labels)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "classification_report": report
    }

    return metrics, y_pred, y_prob
