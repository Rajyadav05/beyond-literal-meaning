"""
src/idiom/candidate_detector.py
================================
Idiom Candidate Detector module.

Scans input sentences for potential idiom candidates using the local idiom lexicon.
Returns structured candidate objects with exact character span offsets, matched text,
and normalized lexicon key.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class IdiomCandidateDetector:
    def __init__(self, lexicon_path: Optional[Path] = None):
        if lexicon_path is None:
            lexicon_path = PROJECT_ROOT / "models" / "idiom" / "idiom_lexicon.json"

        self.lexicon_path = lexicon_path
        self.lexicon: Dict[str, Dict[str, Any]] = {}
        self.expressions: List[str] = []
        self._load_lexicon()

    def _load_lexicon(self):
        if not self.lexicon_path.exists():
            # Fallback to datasets/processed if models/idiom doesn't exist yet
            alt_path = PROJECT_ROOT / "datasets" / "processed" / "idiom_lexicon.json"
            if alt_path.exists():
                self.lexicon_path = alt_path

        if self.lexicon_path.exists():
            with open(self.lexicon_path, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)
            # Sort expressions by length descending so longer idioms match first (greedy / boundary-aware)
            self.expressions = sorted(self.lexicon.keys(), key=lambda x: len(x), reverse=True)
        else:
            self.lexicon = {}
            self.expressions = []

    def detect_candidates(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Detects all valid idiom candidates in a sentence.
        Returns a list of candidate dictionaries containing:
        - expression: canonical lexicon expression
        - normalized_expression: lowercased lexicon key
        - matched_text: actual string substring in input
        - start_char: start character offset in input
        - end_char: end character offset in input
        - source: lexicon source dataset
        """
        if not sentence or not sentence.strip():
            return []

        candidates = []
        occupied_spans: List[Tuple[int, int]] = []

        # 1. Exact / regex search for lexicon expressions (longest phrase first)
        for norm_expr in self.expressions:
            # Escape for regex, but allow flexible whitespace / punctuation boundaries
            # Handle lemmatized / tense variations like spill/spilled by fuzzy regex or word boundary matching
            words = norm_expr.split()
            if len(words) == 1:
                # Avoid single generic words matching as idioms unless multi-word
                continue

            # Build regex pattern allowing word boundaries and optional past-tense / plural variants for verbs
            pattern_parts = []
            for w in words:
                escaped_w = re.escape(w)
                if w in ["spill", "bite", "break", "take", "give", "come", "kick"]:
                    # Allow common verb inflection endings (e.g. spilled, spilling, spills, bit, bitten)
                    pattern_parts.append(rf"{escaped_w}[a-z]*")
                else:
                    pattern_parts.append(escaped_w)

            pattern_str = r"\b" + r"\s+".join(pattern_parts) + r"\b"
            pattern = re.compile(pattern_str, re.IGNORECASE)

            for match in pattern.finditer(sentence):
                start, end = match.start(), match.end()

                # Boundary check: avoid overlapping sub-phrases if already covered by longer idiom match
                is_subspan = False
                for occ_start, occ_end in occupied_spans:
                    if occ_start <= start and end <= occ_end:
                        is_subspan = True
                        break

                if not is_subspan:
                    matched_txt = sentence[start:end]
                    lex_entry = self.lexicon[norm_expr]
                    candidates.append({
                        "expression": lex_entry["expression"],
                        "normalized_expression": norm_expr,
                        "matched_text": matched_txt,
                        "start_char": start,
                        "end_char": end,
                        "source": lex_entry.get("source", "lexicon"),
                        "literal_meaning": lex_entry.get("literal_meaning", ""),
                        "idiomatic_meaning": lex_entry.get("idiomatic_meaning", ""),
                        "has_dictionary_meaning": lex_entry.get("has_dictionary_meaning", False)
                    })
                    occupied_spans.append((start, end))

        # Sort candidates by start_char offset
        candidates.sort(key=lambda x: x["start_char"])
        return candidates
