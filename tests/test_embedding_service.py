"""
tests/test_embedding_service.py
================================
Phase 8 Unit Tests for EmbeddingService using all-MiniLM-L6-v2.

Verifies:
- Local offline model loading on CPU
- Empirical embedding dimension = 384
- Inference determinism across 3 repeated runs
- Cosine similarity range [-1, 1]
- Identical text similarity ~1.0
- Unrelated text similarity < identical text similarity
- Single vs batch inference consistency
- Empty/whitespace input validation
"""

import math
import pytest
import numpy as np
from src.semantics.embedding_service import EmbeddingService


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


def test_embedding_model_load_and_dimension(embedding_service):
    """Verify local model loading and 384-dimensional output."""
    assert embedding_service.is_loaded
    assert embedding_service.embedding_dimension == 384

    vec = embedding_service.encode("The exam was a piece of cake.")
    assert isinstance(vec, list)
    assert len(vec) == 384


def test_embedding_determinism(embedding_service):
    """Verify 3 consecutive runs produce identical embeddings."""
    text = "Rahul finally spilled the beans."
    emb1 = embedding_service.encode(text)
    emb2 = embedding_service.encode(text)
    emb3 = embedding_service.encode(text)

    diff1 = np.max(np.abs(np.array(emb1) - np.array(emb2)))
    diff2 = np.max(np.abs(np.array(emb1) - np.array(emb3)))

    assert diff1 < 1e-6
    assert diff2 < 1e-6


def test_cosine_similarity_properties(embedding_service):
    """Verify cosine similarity properties: identical = ~1.0, range [-1, 1]."""
    text1 = "The exam was a piece of cake."
    text2 = "The test was very easy."
    text3 = "Quantum astrophysics involves black holes."

    # Identical texts
    sim_identical = embedding_service.similarity(text1, text1)
    assert math.isclose(sim_identical["similarity"], 1.0, abs_tol=1e-5)
    assert sim_identical["embedding_dimension"] == 384
    assert sim_identical["method"] == "cosine_similarity"

    # Similar vs Unrelated
    sim_similar = embedding_service.similarity(text1, text2)["similarity"]
    sim_unrelated = embedding_service.similarity(text1, text3)["similarity"]

    assert -1.0 <= sim_similar <= 1.0
    assert -1.0 <= sim_unrelated <= 1.0
    assert sim_similar > sim_unrelated


def test_single_vs_batch_consistency(embedding_service):
    """Verify single encode vs batch encode produces identical vectors."""
    texts = [
        "The exam was a piece of cake.",
        "She ate a piece of cake.",
        "Rahul finally spilled the beans."
    ]

    single_embs = [embedding_service.encode(t) for t in texts]
    batch_embs = embedding_service.encode_batch(texts)

    assert len(single_embs) == len(batch_embs) == 3
    for i in range(len(texts)):
        diff = np.max(np.abs(np.array(single_embs[i]) - np.array(batch_embs[i])))
        assert diff < 1e-6


def test_empty_input_validation(embedding_service):
    """Verify empty or whitespace-only inputs raise ValueError."""
    with pytest.raises(ValueError):
        embedding_service.encode("")

    with pytest.raises(ValueError):
        embedding_service.encode("   ")

    with pytest.raises(ValueError):
        embedding_service.similarity("valid", "")
