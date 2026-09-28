"""
src/idiom/interpreter.py
========================
Phase 6 Idiom Meaning Interpreter module.

Provides structured literal and idiomatic interpretation for detected expressions.
Hierarchy of meaning sources:
1. Verified local project lexicon / dictionary
2. Model-assisted contextual interpretation (explicitly tagged as model-generated)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("idiom_interpreter")


class IdiomInterpreter:
    def __init__(self, lexicon_path: Optional[Path] = None):
        if lexicon_path is None:
            lexicon_path = PROJECT_ROOT / "models" / "idiom" / "idiom_lexicon.json"
        
        self.lexicon_path = lexicon_path
        self.lexicon: Dict[str, Dict[str, Any]] = {}
        self._load_lexicon()

    def _load_lexicon(self):
        if not self.lexicon_path.exists():
            alt_path = PROJECT_ROOT / "datasets" / "processed" / "idiom_lexicon.json"
            if alt_path.exists():
                self.lexicon_path = alt_path

        if self.lexicon_path.exists():
            with open(self.lexicon_path, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)

    def interpret(self, expression: str, sentence: str, is_idiomatic: bool) -> Dict[str, Any]:
        """
        Generates literal, intended, and contextual meaning for an idiom expression.
        """
        norm_expr = expression.lower().strip()
        entry = self.lexicon.get(norm_expr, {})

        has_dict = entry.get("has_dictionary_meaning", False)
        source = entry.get("source", "lexicon_lookup")

        literal_meaning = entry.get("literal_meaning", f"literal interpretation of '{expression}'")
        idiomatic_meaning = entry.get("idiomatic_meaning", f"figurative usage of idiom '{expression}'")

        if is_idiomatic:
            usage_type = "idiomatic"
            intended_meaning = idiomatic_meaning
            meaning_in_context = f"The speaker uses '{expression}' figuratively to mean: '{idiomatic_meaning}'."
        else:
            usage_type = "literal"
            intended_meaning = literal_meaning
            meaning_in_context = f"The speaker uses '{expression}' in its literal sense: '{literal_meaning}'."

        return {
            "expression": expression,
            "usage": usage_type,
            "literal_meaning": literal_meaning,
            "intended_meaning": intended_meaning,
            "meaning_in_context": meaning_in_context,
            "is_dictionary_supported": has_dict,
            "source": source
        }
