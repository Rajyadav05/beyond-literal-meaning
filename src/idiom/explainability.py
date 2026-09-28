"""
src/idiom/explainability.py
===========================
Phase 6 Idiom Pipeline Explainability module.

Provides step-by-step decision trace and feature evidence for idiom detection
and contextual usage classification.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("idiom_explainability")


class IdiomExplainer:
    def explain(
        self,
        candidate: Dict[str, Any],
        is_idiomatic: bool,
        probability_idiomatic: float,
        model_name: str = "TF-IDF + Logistic Regression"
    ) -> Dict[str, Any]:
        """
        Generates a transparent 4-step explanation trace and evidence summary.
        """
        expr = candidate.get("expression", "")
        start_char = candidate.get("start_char", 0)
        end_char = candidate.get("end_char", 0)
        source = candidate.get("source", "idiom_lexicon")

        usage_str = "idiomatic" if is_idiomatic else "literal"

        steps = [
            {
                "step": 1,
                "stage": "Candidate Span Detection",
                "description": f"Detected target candidate phrase '{expr}' between character positions {start_char}:{end_char}."
            },
            {
                "step": 2,
                "stage": "Lexicon Matching",
                "description": f"Verified expression '{expr}' in the local idiom lexicon (Source: {source})."
            },
            {
                "step": 3,
                "stage": "Contextual Classification",
                "description": f"Classified usage as {usage_str} using {model_name} (idiomatic probability: {probability_idiomatic:.4f})."
            },
            {
                "step": 4,
                "stage": "Interpretation Mapping",
                "description": f"Mapped classified usage ({usage_str}) to dictionary-supported meaning representation."
            }
        ]

        evidence = [
            f"Phrase '{expr}' is present in input text span [{start_char}:{end_char}].",
            f"Contextual classifier assigned a {probability_idiomatic*100:.1f}% probability of idiomatic usage.",
            f"Decision rule: probability >= 0.50 -> {usage_str}."
        ]

        return {
            "method": "Pipeline Step-Trace & Probability Evidence",
            "model_family": model_name,
            "steps": steps,
            "evidence": evidence
        }
