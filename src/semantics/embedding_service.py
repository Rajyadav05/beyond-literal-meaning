"""
src/semantics/embedding_service.py
==================================
Phase 8 — Local Sentence Embedding & Semantic Similarity Service.

Uses local pretrained sentence-transformers model from `models/embeddings/` (all-MiniLM-L6-v2).
Provides deterministic CPU-based embedding generation and cosine similarity calculation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("embedding_service")


class EmbeddingService:
    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            model_dir = PROJECT_ROOT / "models" / "embeddings"
        self.model_dir = model_dir
        self.device = torch.device("cpu")
        self.embedding_dimension = 384

        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if not self.model_dir.exists():
            logger.warning("Embedding model directory missing at: %s", self.model_dir)
            return

        try:
            # Load local model offline on CPU
            self.model = SentenceTransformer(str(self.model_dir), device="cpu")
            self.is_loaded = True
            
            # Empirically verify dimension
            test_emb = self.model.encode(["test"], convert_to_numpy=True)
            self.embedding_dimension = int(test_emb.shape[1])
            logger.info("Loaded EmbeddingService successfully from %s (Dimension: %d)", self.model_dir, self.embedding_dimension)
        except Exception as e:
            logger.error("Failed to load EmbeddingService from %s: %s", self.model_dir, e)
            self.is_loaded = False

    def encode(self, text: str) -> List[float]:
        """
        Encodes a single sentence into a dense vector (List[float]).
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only.")

        if not self.is_loaded:
            raise RuntimeError("Embedding model is not loaded or local weights are missing.")

        with torch.no_grad():
            emb = self.model.encode(text.strip(), convert_to_numpy=True, normalize_embeddings=False)

        return [float(x) for x in emb]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes a list of sentences into a list of dense vectors (List[List[float]]).
        """
        if not texts:
            return []

        cleaned_texts = [t.strip() for t in texts if t and t.strip()]
        if len(cleaned_texts) != len(texts):
            raise ValueError("Input texts list contains empty or whitespace-only strings.")

        if not self.is_loaded:
            raise RuntimeError("Embedding model is not loaded or local weights are missing.")

        with torch.no_grad():
            embs = self.model.encode(cleaned_texts, convert_to_numpy=True, normalize_embeddings=False)

        return [[float(x) for x in vec] for vec in embs]

    def similarity(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """
        Computes cosine semantic similarity between two texts.
        Returns similarity score, embedding dimension, and method metadata.
        """
        if not text_a or not text_a.strip() or not text_b or not text_b.strip():
            raise ValueError("Input texts cannot be empty or whitespace-only.")

        vec_a = np.array(self.encode(text_a), dtype=np.float32)
        vec_b = np.array(self.encode(text_b), dtype=np.float32)

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

        return {
            "text_a": text_a.strip(),
            "text_b": text_b.strip(),
            "similarity": cos_sim,
            "embedding_dimension": self.embedding_dimension,
            "method": "cosine_similarity"
        }
