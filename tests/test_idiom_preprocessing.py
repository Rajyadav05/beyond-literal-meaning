"""
tests/test_idiom_preprocessing.py
==================================
Unit tests for idiom preprocessing module (`src/idiom/idiom_preprocessor.py`).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.idiom.idiom_preprocessor import extract_idiom_span


def test_extract_idiom_span_exact():
    sentence = "The exam was a piece of cake."
    idiom = "piece of cake"
    start, end, matched = extract_idiom_span(sentence, idiom)

    assert start == 15
    assert end == 28
    assert matched == "piece of cake"
    assert sentence[start:end] == "piece of cake"


def test_extract_idiom_span_case_insensitive():
    sentence = "Rahul finally Spilled The Beans yesterday."
    idiom = "spilled the beans"
    start, end, matched = extract_idiom_span(sentence, idiom)

    assert start == 14
    assert end == 31
    assert matched == "Spilled The Beans"


def test_extract_idiom_span_not_found():
    sentence = "Normal sentence without idioms."
    idiom = "spill the beans"
    start, end, matched = extract_idiom_span(sentence, idiom)

    assert start == -1
    assert end == -1
    assert matched == ""
