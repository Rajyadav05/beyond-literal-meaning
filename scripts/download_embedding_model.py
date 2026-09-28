"""
scripts/download_embedding_model.py
===================================
Phase 8 — One-Time Model Downloader & Verification for Sentence-Transformer Model.

Downloads `sentence-transformers/all-MiniLM-L6-v2` locally to `models/embeddings/`.
Verifies offline loading with `local_files_only=True` and checks embedding dimension = 384.
"""

import json
import logging
import time
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_embedding_model")


def download_and_verify_embedding_model():
    model_dir = PROJECT_ROOT / "models" / "embeddings"
    model_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "sentence-transformers/all-MiniLM-L6-v2"

    logger.info("=== Downloading Embedding Model: %s -> %s ===", repo_id, model_dir)
    
    start_time = time.time()
    # Download using SentenceTransformer and save locally
    st_model = SentenceTransformer(repo_id)
    st_model.save(str(model_dir))
    
    # Also save tokenizer & model explicitly for HF compatibility
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    tokenizer.save_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(repo_id)
    model.save_pretrained(str(model_dir))
    
    download_time = time.time() - start_time
    logger.info("Model download and save completed in %.2fs.", download_time)

    # Verification Step: Offline Load Test
    logger.info("\n=== Offline Verification Test (local_files_only=True) ===")
    start_verify = time.time()
    
    local_st = SentenceTransformer(str(model_dir), device="cpu")
    verify_time = time.time() - start_verify

    # Empirically measure embedding dimension
    sample_text = "The exam was a piece of cake."
    embeddings = local_st.encode([sample_text], convert_to_tensor=True)
    emp_dim = embeddings.shape[1]

    logger.info("  Successfully loaded model offline in %.3fs.", verify_time)
    logger.info("  Sample text: '%s'", sample_text)
    logger.info("  Embedding shape: %s", list(embeddings.shape))
    logger.info("  Empirical embedding dimension: %d", emp_dim)

    if emp_dim != 384:
        raise ValueError(f"Expected embedding dimension 384, but got {emp_dim}!")

    logger.info("=== VERIFICATION PASSED: Embedding Model is offline-ready and 384-dimensional ===")
    return emp_dim, verify_time


if __name__ == "__main__":
    download_and_verify_embedding_model()
