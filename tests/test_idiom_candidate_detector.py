"""
tests/test_idiom_candidate_detector.py
======================================
Unit tests for idiom candidate detector (`src/idiom/candidate_detector.py`).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.idiom.candidate_detector import IdiomCandidateDetector


def test_candidate_detector_piece_of_cake():
    detector = IdiomCandidateDetector()
    sentence = "The exam was a piece of cake."
    candidates = detector.detect_candidates(sentence)

    assert len(candidates) >= 1
    top = candidates[0]
    assert top["expression"] == "piece of cake"
    assert top["matched_text"] == "piece of cake"
    assert top["start_char"] == 15
    assert top["end_char"] == 28
    assert sentence[top["start_char"]:top["end_char"]] == "piece of cake"


def test_candidate_detector_spilled_the_beans():
    detector = IdiomCandidateDetector()
    sentence = "Rahul finally spilled the beans."
    candidates = detector.detect_candidates(sentence)

    assert len(candidates) >= 1
    top = candidates[0]
    assert top["expression"] in ["spilled the beans", "spill the beans"]
    assert top["matched_text"] == "spilled the beans"
    assert top["start_char"] == 14
    assert top["end_char"] == 31


def test_candidate_detector_no_idiom():
    detector = IdiomCandidateDetector()
    sentence = "The weather today is cloudy and cold."
    candidates = detector.detect_candidates(sentence)
    assert len(candidates) == 0


def test_candidate_detector_punctuation_and_casing():
    detector = IdiomCandidateDetector()
    sentence = "Piece of cake! That was so easy."
    candidates = detector.detect_candidates(sentence)

    assert len(candidates) >= 1
    top = candidates[0]
    assert top["matched_text"] == "Piece of cake"
    assert top["start_char"] == 0
    assert top["end_char"] == 13
