"""
tests/test_evidence_fusion.py
==============================
Phase 9 Unit Tests for EvidenceFusionEngine & ConflictDetector.

Verifies:
- Structured evidence representation across Sarcasm, Idiom, Sentiment, Emotion, Semantics
- Preservation of original model predictions and metadata
- DistilBERT exploratory subset scope transparency
- GoEmotions native 28-label preservation
- Transparent conflict, agreement, and unavailable signal detection
- Context ON/OFF comparison deltas
- Absence of fabricated overall_confidence numbers
"""

import pytest
from src.explainability.evidence_fusion import EvidenceFusionEngine
from src.explainability.conflict_detector import ConflictDetector


@pytest.fixture(scope="module")
def fusion_engine():
    return EvidenceFusionEngine()


def test_evidence_fusion_structure(fusion_engine):
    """Verify evidence fusion returns structured evidence list and explanation trace."""
    sentence = "The exam was a piece of cake."
    res = fusion_engine.analyze(sentence)

    assert res["input_text"] == sentence
    assert res["context"] is None
    assert isinstance(res["evidence"], list)
    assert len(res["evidence"]) >= 4

    # Verify structured fields in evidence items
    for item in res["evidence"]:
        assert "source" in item
        assert "type" in item
        assert "signal" in item
        assert "description" in item
        assert "supports" in item
        assert "limitations" in item

    # Verify explanation trace stages
    stages = [stage["stage"] for stage in res["explanation_trace"]]
    assert "input" in stages
    assert "sarcasm" in stages
    assert "idiom" in stages
    assert "sentiment" in stages
    assert "emotion" in stages
    assert "interpretation" in stages


def test_no_fabricated_confidence(fusion_engine):
    """Verify that overall_confidence is NOT present in interpretation summary."""
    sentence = "Rahul finally spilled the beans."
    res = fusion_engine.analyze(sentence)

    summary = res["interpretation_summary"]
    assert "interpretation" in summary
    assert "basis" in summary
    assert "sarcasm_model_probability" in summary
    assert "overall_confidence" not in summary  # MUST NOT fabricate overall_confidence


def test_sarcasm_and_distilbert_metadata(fusion_engine):
    """Verify DistilBERT exploratory subset metadata is preserved."""
    sentence = "I absolutely LOVE waiting three hours for the bus."
    res = fusion_engine.analyze(sentence)

    evidence_sources = [ev["source"] for ev in res["evidence"]]
    assert "sarcasm_classifier" in evidence_sources

    # If DistilBERT model is loaded, check its exploratory metadata flag
    for ev in res["evidence"]:
        if ev["source"] == "sarcasm_classifier_distilbert":
            assert ev["experiment_type"] == "exploratory_subset"
            assert ev["training_subset_size"] == 1000
            assert ev["test_subset_size"] == 500


def test_native_emotion_and_sentiment_preservation(fusion_engine):
    """Verify surface sentiment and native GoEmotions 28 labels are preserved."""
    sentence = "Wonderful."
    res = fusion_engine.analyze(sentence)

    sent_ev = next(ev for ev in res["evidence"] if ev["source"] == "sentiment_model")
    assert sent_ev["value"] in ["positive", "negative", "neutral"]
    assert "probabilities" in sent_ev

    emo_ev = next(ev for ev in res["evidence"] if ev["source"] == "emotion_model")
    assert isinstance(emo_ev["value"], list)
    assert len(emo_ev["value"]) == 5  # Top 5 GoEmotions rankings


def test_contextual_incongruity_and_agreement(fusion_engine):
    """Verify positive wording + adverse context triggers incongruity signal."""
    context = "My laptop crashed five minutes before my presentation."
    sentence = "Wow, exactly what I needed!"

    res = fusion_engine.analyze(sentence, context=context)
    signals = res["signal_alignment"]

    assert isinstance(signals["agreement"], list)
    assert isinstance(signals["disagreement"], list)
    assert isinstance(signals["unavailable"], list)

    # Contextual incongruity signal check
    context_ev = next((ev for ev in res["evidence"] if ev["source"] == "contextual_signal"), None)
    assert context_ev is not None
    assert context_ev["signal"] == "adverse_context"


def test_unavailable_context_signal(fusion_engine):
    """Verify missing context reports unavailable signal summary."""
    sentence = "The exam was a piece of cake."
    res = fusion_engine.analyze(sentence, context=None)

    unavail = res["signal_alignment"]["unavailable"]
    assert any("context" in msg.lower() for msg in unavail)


def test_context_on_off_comparison(fusion_engine):
    """Verify context ON/OFF comparison computes exact deltas without overclaiming."""
    context = "My phone fell into the swimming pool."
    sentence = "Wonderful."

    comp_res = fusion_engine.compare_context(sentence, context)

    assert comp_res["sentence"] == sentence
    assert comp_res["context"] == context
    assert "without_context" in comp_res
    assert "with_context" in comp_res

    delta = comp_res["delta"]
    assert "sarcasm_probability_without_context" in delta
    assert "sarcasm_probability_with_context" in delta
    assert "sarcasm_probability_delta" in delta
    assert "prediction_changed" in delta
    assert isinstance(delta["prediction_changed"], bool)
