"""
scripts/smoke_test_sentiment_emotion_models.py
================================================
Phase 7.2 — Local Offline Load Test & Forward Pass Verification for Sentiment & Emotion Models.

Loads models using strictly `local_files_only=True` from:
- `models/sentiment/`
- `models/emotion/`

Runs CPU inference on 4 project demonstration sentences and saves predictions to:
- `results/predictions/sentiment_emotion_smoke_test.json`

Usage:
    python scripts/smoke_test_sentiment_emotion_models.py
"""

import json
import logging
import time
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smoke_test_sentiment_emotion_models")

DEMO_SENTENCES = [
    "I absolutely LOVE waiting three hours for the bus.",
    "Wonderful.",
    "The exam was a piece of cake.",
    "Rahul finally spilled the beans."
]


def test_sentiment_offline():
    model_dir = PROJECT_ROOT / "models" / "sentiment"
    logger.info("=== Offline Load Test: Sentiment Model (%s) ===", model_dir)

    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    load_time = time.time() - start_load

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    logger.info("  Loaded in %.2fs on %s. Labels (%d): %s", load_time, device, len(id2label), id2label)

    results = []
    start_infer = time.time()
    for sentence in DEMO_SENTENCES:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_id = int(probs.argmax())
        pred_label = id2label[pred_id]
        prob_dict = {id2label[i]: float(probs[i]) for i in range(len(probs))}

        results.append({
            "sentence": sentence,
            "predicted_label": pred_label,
            "probabilities": prob_dict
        })

    infer_time = time.time() - start_infer
    logger.info("  Inference completed in %.3fs for %d sentences.", infer_time, len(DEMO_SENTENCES))
    return results, id2label, load_time, infer_time


def test_emotion_offline():
    model_dir = PROJECT_ROOT / "models" / "emotion"
    logger.info("\n=== Offline Load Test: Emotion Model (%s) ===", model_dir)

    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    load_time = time.time() - start_load

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    logger.info("  Loaded in %.2fs on %s. Labels (%d): %s", load_time, device, len(id2label), list(id2label.values())[:5])

    results = []
    start_infer = time.time()
    for sentence in DEMO_SENTENCES:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            # GoEmotions roberta-base uses sigmoid for multi-label emotion scoring
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # Map full 28-label probability distribution
        full_probs = {id2label[i]: float(probs[i]) for i in range(len(probs))}
        
        # Sort top 5 emotions for readability
        top_emotions = sorted(full_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_dict = {k: v for k, v in top_emotions}

        results.append({
            "sentence": sentence,
            "top_emotions": top_dict,
            "full_emotion_distribution": full_probs
        })

    infer_time = time.time() - start_infer
    logger.info("  Inference completed in %.3fs for %d sentences.", infer_time, len(DEMO_SENTENCES))
    return results, id2label, load_time, infer_time


def main():
    sent_results, sent_id2label, sent_load_t, sent_infer_t = test_sentiment_offline()
    emo_results, emo_id2label, emo_load_t, emo_infer_t = test_emotion_offline()

    out_dir = PROJECT_ROOT / "results" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sentiment_emotion_smoke_test.json"

    payload = {
        "metadata": {
            "environment": "offline",
            "device": "cpu",
            "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "sentiment_id2label": {str(k): v for k, v in sent_id2label.items()},
            "sentiment_load_seconds": round(sent_load_t, 3),
            "sentiment_infer_seconds": round(sent_infer_t, 3),
            "emotion_model": "SamLowe/roberta-base-go_emotions",
            "emotion_id2label": {str(k): v for k, v in emo_id2label.items()},
            "emotion_load_seconds": round(emo_load_t, 3),
            "emotion_infer_seconds": round(emo_infer_t, 3)
        },
        "sentiment_demo_predictions": sent_results,
        "emotion_demo_predictions": emo_results
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("\nSaved offline smoke test predictions to: %s", out_path)


if __name__ == "__main__":
    main()
