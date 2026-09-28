"""
src/sentiment/sentiment_service.py
===================================
Phase 7.3 — Local Sentiment Analysis Service using cardiffnlp/twitter-roberta-base-sentiment-latest.

Loads the local pretrained model from `models/sentiment/` offline on CPU.
Computes 3-class softmax probabilities for negative, neutral, and positive classes.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("sentiment_service")


class SentimentService:
    def __init__(self, model_dir: Optional[Path] = None, max_length: int = 256):
        if model_dir is None:
            model_dir = PROJECT_ROOT / "models" / "sentiment"
        self.model_dir = model_dir
        self.max_length = max_length
        self.device = torch.device("cpu")

        self.tokenizer = None
        self.model = None
        self.id2label: Dict[int, str] = {}
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if not self.model_dir.exists():
            logger.warning("Sentiment model directory missing at: %s", self.model_dir)
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir), local_files_only=True)
            self.model.to(self.device)
            self.model.eval()

            self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
            self.is_loaded = True
            logger.info("Loaded SentimentService successfully from %s (Labels: %s)", self.model_dir, self.id2label)
        except Exception as e:
            logger.error("Failed to load SentimentService from %s: %s", self.model_dir, e)
            self.is_loaded = False

    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Performs local CPU sentiment inference.
        Returns label and probability distribution across negative, neutral, and positive classes.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only.")

        if not self.is_loaded:
            raise RuntimeError("Sentiment model is not loaded or local weights are missing.")

        inputs = self.tokenizer(
            text.strip(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_id = int(probs.argmax())
        pred_label = self.id2label[pred_id]
        prob_dict = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return {
            "text": text.strip(),
            "label": pred_label,
            "probabilities": prob_dict
        }
