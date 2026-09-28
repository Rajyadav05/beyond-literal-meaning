"""
src/emotion/emotion_service.py
===============================
Phase 7.3 — Local Emotion Analysis Service using SamLowe/roberta-base-go_emotions.

Loads local pretrained model from `models/emotion/` offline on CPU.
Computes independent multi-label Sigmoid probabilities over 28 native GoEmotions labels.
Returns full probability distribution and top-N sorted emotion rankings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("emotion_service")


class EmotionService:
    def __init__(self, model_dir: Optional[Path] = None, max_length: int = 256):
        if model_dir is None:
            model_dir = PROJECT_ROOT / "models" / "emotion"
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
            logger.warning("Emotion model directory missing at: %s", self.model_dir)
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir), local_files_only=True)
            self.model.to(self.device)
            self.model.eval()

            self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
            self.is_loaded = True
            logger.info("Loaded EmotionService successfully from %s (Labels: %d)", self.model_dir, len(self.id2label))
        except Exception as e:
            logger.error("Failed to load EmotionService from %s: %s", self.model_dir, e)
            self.is_loaded = False

    def predict_emotion(self, text: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Performs local CPU emotion inference.
        Returns complete 28-label probability distribution and sorted top-N emotions.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only.")

        if not (1 <= top_n <= 28):
            raise ValueError(f"top_n must be between 1 and 28, got {top_n}.")

        if not self.is_loaded:
            raise RuntimeError("Emotion model is not loaded or local weights are missing.")

        inputs = self.tokenizer(
            text.strip(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        full_probs = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}
        
        # Sort top-N emotions by probability descending
        sorted_emotions = sorted(full_probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_emotions = [{"label": k, "probability": v} for k, v in sorted_emotions]

        return {
            "text": text.strip(),
            "probabilities": full_probs,
            "top_emotions": top_emotions
        }
