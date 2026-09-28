"""
scripts/smoke_test_sentiment_emotion.py
========================================
Phase 7.1 — Inspection & Smoke Test Script for Sentiment and Emotion Analysis.

Verifies local availability of:
1. Sentiment model (`models/sentiment/`)
2. Emotion model (`models/emotion/`)
3. Raw GoEmotions dataset (`datasets/raw/goemotions/`)

Usage:
    python scripts/smoke_test_sentiment_emotion.py
"""

import json
import logging
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smoke_test_sentiment_emotion")


def check_sentiment_model():
    model_dir = PROJECT_ROOT / "models" / "sentiment"
    logger.info("=== Checking Sentiment Model in %s ===", model_dir)

    has_weights = any(model_dir.glob("*.bin")) or any(model_dir.glob("*.safetensors"))
    has_config = (model_dir / "config.json").exists()

    if has_weights and has_config:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            logger.info("  [SUCCESS] Local Sentiment Model loadable offline!")

            # Run forward pass on demo sentence
            demo_text = "I absolutely LOVE waiting three hours for the bus."
            inputs = tokenizer(demo_text, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
            logger.info("  Demo forward pass probabilities: %s", probs)

            return {"available": True, "loadable": True, "path": str(model_dir)}
        except Exception as e:
            logger.warning("  [FAILED TO LOAD] %s", e)
            return {"available": False, "error": str(e)}
    else:
        logger.info("  [NOT FOUND] Sentiment model weights missing in %s", model_dir)
        return {"available": False, "download_required": True, "model": "cardiffnlp/twitter-roberta-base-sentiment-latest"}


def check_emotion_model():
    model_dir = PROJECT_ROOT / "models" / "emotion"
    logger.info("\n=== Checking Emotion Model in %s ===", model_dir)

    has_weights = any(model_dir.glob("*.bin")) or any(model_dir.glob("*.safetensors"))
    has_config = (model_dir / "config.json").exists()

    if has_weights and has_config:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            id2label = model.config.id2label
            logger.info("  [SUCCESS] Local Emotion Model loadable offline! Labels (%d): %s", len(id2label), list(id2label.values())[:5])

            demo_text = "I absolutely LOVE waiting three hours for the bus."
            inputs = tokenizer(demo_text, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
            logger.info("  Demo forward pass output dimension: %d", len(probs))

            return {"available": True, "loadable": True, "path": str(model_dir), "label_count": len(id2label)}
        except Exception as e:
            logger.warning("  [FAILED TO LOAD] %s", e)
            return {"available": False, "error": str(e)}
    else:
        logger.info("  [NOT FOUND] Emotion model weights missing in %s", model_dir)
        return {"available": False, "download_required": True, "model": "SamLowe/roberta-base-go_emotions"}


def check_goemotions_dataset():
    go_dir = PROJECT_ROOT / "datasets" / "raw" / "goemotions"
    logger.info("\n=== Checking GoEmotions Dataset in %s ===", go_dir)

    train_tsv = go_dir / "train.tsv"
    dev_tsv = go_dir / "dev.tsv"
    test_tsv = go_dir / "test.tsv"
    emotions_txt = go_dir / "emotions.txt"

    if train_tsv.exists() and dev_tsv.exists() and test_tsv.exists() and emotions_txt.exists():
        labels = [line.strip() for line in open(emotions_txt, encoding="utf-8") if line.strip()]
        logger.info("  [SUCCESS] GoEmotions dataset present! Labels (%d): %s", len(labels), labels[:5])
        return {"available": True, "label_count": len(labels)}
    else:
        logger.warning("  [MISSING] GoEmotions dataset files missing.")
        return {"available": False}


def main():
    sent_res = check_sentiment_model()
    emo_res = check_emotion_model()
    go_res = check_goemotions_dataset()

    logger.info("\n=== Inspection Summary ===")
    logger.info("Sentiment Model Available: %s", sent_res.get("available"))
    logger.info("Emotion Model Available:   %s", emo_res.get("available"))
    logger.info("GoEmotions Dataset:       %s", go_res.get("available"))


if __name__ == "__main__":
    main()
