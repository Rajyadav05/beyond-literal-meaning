"""
scripts/download_sentiment_emotion_models.py
=============================================
Phase 7.2 — Idempotent Local Model Downloader for Sentiment & Emotion Models.

Downloads and saves pretrained Hugging Face models into project directories:
- `models/sentiment/` (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- `models/emotion/` (`SamLowe/roberta-base-go_emotions`)

Usage:
    python scripts/download_sentiment_emotion_models.py
"""

import json
import logging
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_sentiment_emotion_models")


def is_model_downloaded(model_dir: Path) -> bool:
    """Checks if model directory contains config.json and weight files."""
    if not model_dir.exists():
        return False
    has_config = (model_dir / "config.json").exists()
    has_weights = (
        any(model_dir.glob("*.bin")) or 
        any(model_dir.glob("*.safetensors")) or
        any(model_dir.glob("model.safetensors*"))
    )
    return has_config and has_weights


def download_sentiment_model():
    target_dir = PROJECT_ROOT / "models" / "sentiment"
    repo_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    logger.info("=== Downloading Sentiment Model (%s) ===", repo_id)

    if is_model_downloaded(target_dir):
        logger.info("  [EXISTS] Sentiment model files already present at %s. Skipping download.", target_dir)
        return True

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Downloading from Hugging Face Hub: %s ...", repo_id)
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)

    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)

    logger.info("  [SUCCESS] Saved sentiment model locally to %s", target_dir)
    return True


def download_emotion_model():
    target_dir = PROJECT_ROOT / "models" / "emotion"
    repo_id = "SamLowe/roberta-base-go_emotions"
    logger.info("\n=== Downloading Emotion Model (%s) ===", repo_id)

    if is_model_downloaded(target_dir):
        logger.info("  [EXISTS] Emotion model files already present at %s. Skipping download.", target_dir)
        return True

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Downloading from Hugging Face Hub: %s ...", repo_id)

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)

    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)

    logger.info("  [SUCCESS] Saved emotion model locally to %s", target_dir)
    return True


def update_model_manifest():
    manifest_path = PROJECT_ROOT / "models" / "model_manifest.json"

    sent_dir = PROJECT_ROOT / "models" / "sentiment"
    emo_dir = PROJECT_ROOT / "models" / "emotion"

    sent_config_path = sent_dir / "config.json"
    emo_config_path = emo_dir / "config.json"

    manifest = {}

    if sent_config_path.exists():
        with open(sent_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        id2label = cfg.get("id2label", {})
        label2id = cfg.get("label2id", {})
        # Convert numeric keys to int strings if needed
        id2label = {str(k): str(v) for k, v in id2label.items()}
        manifest["sentiment"] = {
            "repository": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "local_path": "models/sentiment",
            "available": True,
            "num_labels": len(id2label),
            "id2label": id2label,
            "label2id": label2id,
            "verified": True
        }

    if emo_config_path.exists():
        with open(emo_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        id2label = cfg.get("id2label", {})
        label2id = cfg.get("label2id", {})
        id2label = {str(k): str(v) for k, v in id2label.items()}
        manifest["emotion"] = {
            "repository": "SamLowe/roberta-base-go_emotions",
            "local_path": "models/emotion",
            "available": True,
            "num_labels": len(id2label),
            "id2label": id2label,
            "label2id": label2id,
            "verified": True
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("\nSaved model manifest to: %s", manifest_path)


def main():
    download_sentiment_model()
    download_emotion_model()
    update_model_manifest()


if __name__ == "__main__":
    main()
