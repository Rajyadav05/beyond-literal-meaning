"""
scratch/audit.py
================
Detailed reproducibility audit script for Phase 7.4.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import hashlib
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.sentiment.sentiment_service import SentimentService
from src.emotion.emotion_service import EmotionService

TEST_SENTENCES = [
    "I absolutely LOVE waiting three hours for the bus.",
    "Wonderful.",
    "The exam was a piece of cake.",
    "Rahul finally spilled the beans.",
    "My phone fell into the swimming pool.",
    "Wow, exactly what I needed!"
]

def hash_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def audit_model_files(model_dir: Path):
    file_hashes = {}
    for f in sorted(model_dir.glob("*")):
        if f.is_file():
            file_hashes[f.name] = {
                "size_bytes": f.stat().st_size,
                "sha256_prefix": hash_file(f)[:16]
            }
    return file_hashes

def run_audit():
    print("=== PHASE 7.4 AUDIT START ===")
    
    # 1. Model file checks
    sent_hashes = audit_model_files(PROJECT_ROOT / "models" / "sentiment")
    emo_hashes = audit_model_files(PROJECT_ROOT / "models" / "emotion")
    
    print("\n--- Sentiment Model Files ---")
    for k, v in sent_hashes.items():
        print(f"  {k}: {v['sha256_prefix']} ({v['size_bytes']} bytes)")

    print("\n--- Emotion Model Files ---")
    for k, v in emo_hashes.items():
        print(f"  {k}: {v['sha256_prefix']} ({v['size_bytes']} bytes)")

    # 2. Instantiate Direct Models & Services
    sent_svc = SentimentService()
    emo_svc = EmotionService()
    
    sent_tok_direct = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "models" / "sentiment"), local_files_only=True)
    sent_mod_direct = AutoModelForSequenceClassification.from_pretrained(str(PROJECT_ROOT / "models" / "sentiment"), local_files_only=True)
    sent_mod_direct.eval()

    emo_tok_direct = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "models" / "emotion"), local_files_only=True)
    emo_mod_direct = AutoModelForSequenceClassification.from_pretrained(str(PROJECT_ROOT / "models" / "emotion"), local_files_only=True)
    emo_mod_direct.eval()

    # Load FastAPI test client
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)

    # Sentiment Audit Loop
    print("\n=== SENTIMENT AUDIT Across Direct, Service, API ===")
    sent_cases = []
    sent_direct_vs_svc_match = True
    sent_svc_vs_api_match = True
    sent_deterministic = True

    for text in TEST_SENTENCES:
        # A. Direct Inference
        inputs_direct = sent_tok_direct(text, return_tensors="pt", padding=True, truncation=True)
        inputs_direct_max256 = sent_tok_direct(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        
        with torch.inference_mode():
            logits_direct1 = sent_mod_direct(**inputs_direct).logits
            logits_direct2 = sent_mod_direct(**inputs_direct).logits
            logits_direct3 = sent_mod_direct(**inputs_direct).logits
        
        # Determinism check
        max_diff_det = max(
            np.max(np.abs((logits_direct1 - logits_direct2).numpy())),
            np.max(np.abs((logits_direct1 - logits_direct3).numpy()))
        )
        if max_diff_det > 1e-6:
            sent_deterministic = False

        probs_direct = torch.softmax(logits_direct1, dim=-1).squeeze(0).numpy()
        id2label_sent = {int(k): str(v) for k, v in sent_mod_direct.config.id2label.items()}
        label_direct = id2label_sent[int(probs_direct.argmax())]
        probs_direct_dict = {id2label_sent[i]: float(probs_direct[i]) for i in range(len(probs_direct))}

        # B. SentimentService
        svc_res = sent_svc.predict_sentiment(text)
        inputs_svc = sent_svc.tokenizer(text.strip(), return_tensors="pt", padding=True, truncation=True, max_length=sent_svc.max_length)
        with torch.no_grad():
            logits_svc = sent_svc.model(**inputs_svc).logits
        probs_svc = torch.softmax(logits_svc, dim=-1).squeeze(0).numpy()

        # Tokenizer compare
        tok_match = (
            inputs_direct["input_ids"].tolist() == inputs_svc["input_ids"].tolist() and
            inputs_direct["attention_mask"].tolist() == inputs_svc["attention_mask"].tolist()
        )

        # Logit compare (direct vs service)
        logit_diff = np.max(np.abs(logits_direct1.numpy() - logits_svc.numpy()))
        if logit_diff > 1e-6:
            sent_direct_vs_svc_match = False

        # C. FastAPI Endpoint
        api_resp = client.post("/api/sentiment", json={"text": text})
        api_data = api_resp.json()
        
        # Compare service vs API
        svc_vs_api_diff = max(
            abs(svc_res["probabilities"]["negative"] - api_data["probabilities"]["negative"]),
            abs(svc_res["probabilities"]["neutral"] - api_data["probabilities"]["neutral"]),
            abs(svc_res["probabilities"]["positive"] - api_data["probabilities"]["positive"])
        )
        if svc_vs_api_diff > 1e-6 or svc_res["label"] != api_data["label"]:
            sent_svc_vs_api_match = False

        print(f"\nSentence: '{text}'")
        print(f"  Token match (Direct vs Svc): {tok_match}")
        print(f"  Direct Logits: {logits_direct1.squeeze(0).tolist()}")
        print(f"  Service Logits: {logits_svc.squeeze(0).tolist()}")
        print(f"  Logit max diff: {logit_diff:.8f}")
        print(f"  Direct Probs:  {probs_direct_dict}")
        print(f"  Service Probs: {svc_res['probabilities']}")
        print(f"  API Response:  {api_data['probabilities']} (label: {api_data['label']})")

        sent_cases.append({
            "text": text,
            "token_ids_match": tok_match,
            "direct_logits": [float(x) for x in logits_direct1.squeeze(0).numpy()],
            "service_logits": [float(x) for x in logits_svc.squeeze(0).numpy()],
            "logit_max_diff": float(logit_diff),
            "direct_probabilities": probs_direct_dict,
            "service_probabilities": svc_res["probabilities"],
            "api_probabilities": api_data["probabilities"],
            "predicted_label": api_data["label"]
        })

    # Emotion Audit Loop
    print("\n=== EMOTION AUDIT Across Direct, Service, API ===")
    emo_cases = []
    emo_direct_vs_svc_match = True
    emo_svc_vs_api_match = True
    emo_deterministic = True

    for text in TEST_SENTENCES:
        inputs_direct = emo_tok_direct(text, return_tensors="pt", padding=True, truncation=True)
        with torch.inference_mode():
            logits_direct1 = emo_mod_direct(**inputs_direct).logits
            logits_direct2 = emo_mod_direct(**inputs_direct).logits
            logits_direct3 = emo_mod_direct(**inputs_direct).logits

        max_diff_det = max(
            np.max(np.abs((logits_direct1 - logits_direct2).numpy())),
            np.max(np.abs((logits_direct1 - logits_direct3).numpy()))
        )
        if max_diff_det > 1e-6:
            emo_deterministic = False

        probs_direct = torch.sigmoid(logits_direct1).squeeze(0).numpy()
        id2label_emo = {int(k): str(v) for k, v in emo_mod_direct.config.id2label.items()}
        probs_direct_dict = {id2label_emo[i]: float(probs_direct[i]) for i in range(len(probs_direct))}
        top5_direct = sorted(probs_direct_dict.items(), key=lambda x: x[1], reverse=True)[:5]

        # Service
        svc_res = emo_svc.predict_emotion(text, top_n=5)
        inputs_svc = emo_svc.tokenizer(text.strip(), return_tensors="pt", padding=True, truncation=True, max_length=emo_svc.max_length)
        with torch.no_grad():
            logits_svc = emo_svc.model(**inputs_svc).logits
        
        tok_match = (
            inputs_direct["input_ids"].tolist() == inputs_svc["input_ids"].tolist() and
            inputs_direct["attention_mask"].tolist() == inputs_svc["attention_mask"].tolist()
        )

        logit_diff = np.max(np.abs(logits_direct1.numpy() - logits_svc.numpy()))
        if logit_diff > 1e-6:
            emo_direct_vs_svc_match = False

        # API
        api_resp = client.post("/api/emotion", json={"text": text, "top_n": 5})
        api_data = api_resp.json()

        # Compare service vs API top 5
        svc_top_labels = [x["label"] for x in svc_res["top_emotions"]]
        api_top_labels = [x["label"] for x in api_data["top_emotions"]]
        if svc_top_labels != api_top_labels:
            emo_svc_vs_api_match = False

        print(f"\nSentence: '{text}'")
        print(f"  Token match (Direct vs Svc): {tok_match}")
        print(f"  Logit max diff: {logit_diff:.8f}")
        print(f"  Top 5 Direct:  {top5_direct}")
        print(f"  Top 5 Service: {[(x['label'], round(x['probability'], 4)) for x in svc_res['top_emotions']]}")
        print(f"  Top 5 API:     {[(x['label'], round(x['probability'], 4)) for x in api_data['top_emotions']]}")

        emo_cases.append({
            "text": text,
            "token_ids_match": tok_match,
            "logit_max_diff": float(logit_diff),
            "top_5_emotions": api_data["top_emotions"],
            "full_28_label_distribution_sample": {k: float(api_data["probabilities"][k]) for k in list(api_data["probabilities"].keys())[:5]}
        })

    # Batch vs Single Input test for Sentiment
    print("\n=== BATCH VS SINGLE INPUT CHECK ===")
    inputs_single = [sent_tok_direct(s, return_tensors="pt") for s in TEST_SENTENCES]
    inputs_batch = sent_tok_direct(TEST_SENTENCES, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        logits_batch = sent_mod_direct(**inputs_batch).logits
    
    batch_vs_single_diffs = []
    for i, s in enumerate(TEST_SENTENCES):
        with torch.no_grad():
            logits_single = sent_mod_direct(**inputs_single[i]).logits
        diff = np.max(np.abs(logits_single.numpy() - logits_batch[i:i+1].numpy()))
        batch_vs_single_diffs.append(float(diff))
        print(f"  Sentence {i+1}: single vs batch logit max diff = {diff:.8f}")

    print("\n=== SUMMARY OF FINDINGS ===")
    print(f"Sentiment Direct vs Service logit match (tol=1e-6): {sent_direct_vs_svc_match}")
    print(f"Sentiment Service vs API match (tol=1e-6):        {sent_svc_vs_api_match}")
    print(f"Sentiment Determinism (tol=1e-6):                 {sent_deterministic}")
    print(f"Emotion Direct vs Service logit match (tol=1e-6):   {emo_direct_vs_svc_match}")
    print(f"Emotion Service vs API match (tol=1e-6):           {emo_svc_vs_api_match}")
    print(f"Emotion Determinism (tol=1e-6):                    {emo_deterministic}")

if __name__ == "__main__":
    run_audit()
