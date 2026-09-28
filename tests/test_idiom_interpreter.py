"""
tests/test_idiom_interpreter.py
================================
Unit tests for idiom interpreter (`src/idiom/interpreter.py`).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.idiom.interpreter import IdiomInterpreter


def test_interpreter_idiomatic_usage():
    interpreter = IdiomInterpreter()
    res = interpreter.interpret("piece of cake", "The exam was a piece of cake.", is_idiomatic=True)

    assert res["usage"] == "idiomatic"
    assert "very easy" in res["intended_meaning"].lower()
    assert res["is_dictionary_supported"] is True


def test_interpreter_literal_usage():
    interpreter = IdiomInterpreter()
    res = interpreter.interpret("piece of cake", "She ate a piece of cake.", is_idiomatic=False)

    assert res["usage"] == "literal"
    assert "baked food" in res["literal_meaning"].lower() or "piece of cake" in res["literal_meaning"].lower()
    assert res["is_dictionary_supported"] is True
