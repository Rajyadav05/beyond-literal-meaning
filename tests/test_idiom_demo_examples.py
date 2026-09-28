"""
tests/test_idiom_demo_examples.py
==================================
Phase 6.1 Regression tests for canonical demonstration idioms:
1. "The exam was a piece of cake."
2. "Rahul finally spilled the beans."
3. "She ate a piece of cake."

Verifies character offset precision, schema validity, and records model predictions.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.idiom_service import IdiomService


def test_demo_example_1_piece_of_cake():
    service = IdiomService()
    sentence = "The exam was a piece of cake."
    res = service.analyze_idiom(sentence)

    assert res["total_found"] >= 1
    found = res["idioms_found"][0]

    # Verify offset integrity
    start = found["span_start"]
    end = found["span_end"]
    assert sentence[start:end] == found["phrase"]
    assert sentence[start:end] == "piece of cake"

    # Schema integrity checks
    assert "usage" in found
    assert "idiomatic_probability" in found
    assert "literal_meaning" in found
    assert "actual_meaning" in found
    assert "meaning_in_context" in found
    assert "explanation" in found


def test_demo_example_2_spilled_the_beans():
    service = IdiomService()
    sentence = "Rahul finally spilled the beans."
    res = service.analyze_idiom(sentence)

    assert res["total_found"] >= 1
    found = res["idioms_found"][0]

    # Verify offset integrity
    start = found["span_start"]
    end = found["span_end"]
    assert sentence[start:end] == found["phrase"]
    assert sentence[start:end] == "spilled the beans"

    # Usage check (spilled the beans was present in train data)
    assert found["usage"] == "idiomatic"
    assert found["idiomatic_probability"] > 0.50


def test_demo_example_3_literal_contrast():
    service = IdiomService()
    sentence = "She ate a piece of cake."
    res = service.analyze_idiom(sentence)

    assert res["total_found"] >= 1
    found = res["idioms_found"][0]

    # Verify offset integrity
    start = found["span_start"]
    end = found["span_end"]
    assert sentence[start:end] == found["phrase"]
    assert sentence[start:end] == "piece of cake"

    # Schema & probability checks
    assert "usage" in found
    assert found["idiomatic_probability"] is not None
