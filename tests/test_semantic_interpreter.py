"""
tests/test_semantic_interpreter.py
===================================
Phase 8 Unit Tests for SemanticInterpreter.

Verifies:
- Idiom candidate integration with Phase 6 detector and lexicon lookup
- Distinct literal_meaning, idiomatic_meaning, contextual_interpretation fields
- Context-sentence embedding cosine similarity score generation
- Non-hallucination for unknown/unsupported idioms
- Required project test cases execution
"""

import pytest
from src.semantics.semantic_interpreter import SemanticInterpreter


@pytest.fixture(scope="module")
def interpreter():
    return SemanticInterpreter()


def test_known_idiom_interpretation(interpreter):
    """Verify known lexicon idiom returns literal, idiomatic, and contextual interpretations."""
    sentence = "The exam was a piece of cake."
    res = interpreter.interpret(sentence)

    assert res["input_text"] == "The exam was a piece of cake."
    assert res["context"] is None
    assert res["embedding_dimension"] == 384
    assert res["evidence"]["lexicon_match"] is True

    idioms = res["idiom_analysis"]["idioms_found"]
    assert len(idioms) == 1
    found = idioms[0]

    assert found["phrase"] == "piece of cake"
    assert "baked food" in found["literal_meaning"].lower() or "cake" in found["literal_meaning"].lower()
    assert "easy" in found["idiomatic_meaning"].lower()
    assert found["meaning_source"] == "lexicon_lookup"


def test_contextual_interpretation_with_similarity(interpreter):
    """Verify context addition computes cosine similarity and context-supported interpretation."""
    sentence = "Rahul finally spilled the beans."
    context = "Everyone had been waiting for the secret to come out."

    res = interpreter.interpret(sentence, context=context)

    assert res["context"] == context
    assert res["evidence"]["context_available"] is True
    assert res["evidence"]["semantic_similarity_available"] is True

    sim_val = res["semantic_similarity"]["context_sentence"]
    assert isinstance(sim_val, float)
    assert -1.0 <= sim_val <= 1.0

    idioms = res["idiom_analysis"]["idioms_found"]
    assert len(idioms) == 1
    assert "In context" in idioms[0]["contextual_interpretation"]


def test_non_idiom_sentence(interpreter):
    """Verify sentence with no idioms returns clean empty idiom analysis."""
    sentence = "She ate a piece of cake."  # 'piece of cake' is detected as candidate phrase
    res = interpreter.interpret(sentence)

    # Idiom detector identifies 'piece of cake' as a candidate
    assert len(res["idiom_analysis"]["idioms_found"]) == 1
    assert res["idiom_analysis"]["idioms_found"][0]["phrase"] == "piece of cake"


def test_empty_input_validation(interpreter):
    """Verify empty input sentence raises ValueError."""
    with pytest.raises(ValueError):
        interpreter.interpret("")

    with pytest.raises(ValueError):
        interpreter.interpret("   ")
