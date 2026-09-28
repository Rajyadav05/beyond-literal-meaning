"""
src/explainability/conflict_detector.py
========================================
Phase 9 — Conflict & Agreement Detection for Evidence Fusion.

Analyzes supporting and conflicting linguistic/model signals across:
- Surface sentiment vs. contextual situation
- Surface sentiment vs. idiomatic interpretation
- Idiom candidate detection vs. usage classifier prediction
- Context availability
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger("conflict_detector")


class ConflictDetector:
    """
    Analyzes evidence items produced by model services and extracts
    transparent agreement, disagreement, and unavailable signal summaries.
    """

    def analyze_conflicts(self, evidence_list: List[Dict[str, Any]], has_context: bool) -> Dict[str, List[str]]:
        agreement: List[str] = []
        disagreement: List[str] = []
        unavailable: List[str] = []

        # Index evidence by source
        evidence_by_source: Dict[str, Dict[str, Any]] = {}
        for item in evidence_list:
            source = item.get("source", "")
            evidence_by_source[source] = item

        # 1. Contextual Incongruity (Positive Wording + Adverse Context)
        sarcasm_ev = evidence_by_source.get("sarcasm_classifier")
        sentiment_ev = evidence_by_source.get("sentiment_model")
        context_ev = evidence_by_source.get("contextual_signal")
        idiom_ev = evidence_by_source.get("idiom_detector")

        sentence_sentiment = sentiment_ev.get("value") if sentiment_ev else None
        sarcasm_pred = sarcasm_ev.get("value") if sarcasm_ev else None

        if has_context and context_ev:
            context_adverse = context_ev.get("signal") == "adverse_context"
            if context_adverse and sentence_sentiment == "positive":
                agreement.append("Positive surface sentiment conflicts with adverse contextual situation (incongruity signal).")

            if sarcasm_pred == "sarcastic":
                agreement.append("Sarcasm classifier predicts sarcastic interpretation.")
        elif not has_context:
            unavailable.append("Contextual semantic similarity and context-sentence incongruity analysis unavailable (no context provided).")

        # 2. Surface Sentiment vs. Idiom Interpretation Disagreement
        if idiom_ev and idiom_ev.get("value") and sentiment_ev:
            idioms = idiom_ev.get("value", [])
            for id_item in idioms:
                idiom_meaning = id_item.get("idiomatic_meaning", "").lower()
                if "easy" in idiom_meaning and sentence_sentiment == "negative":
                    disagreement.append(
                        f"Surface sentiment ('negative') differs from idiomatic interpretation of '{id_item.get('phrase')}' ('very easy')."
                    )

        # 3. Known Idiom Candidate vs. Usage Classifier Disagreement
        if idiom_ev and idiom_ev.get("value"):
            idioms = idiom_ev.get("value", [])
            for id_item in idioms:
                if id_item.get("detected") and id_item.get("usage") == "literal":
                    disagreement.append(
                        f"Known idiom candidate '{id_item.get('phrase')}' detected, but contextual classifier predicts literal usage."
                    )

        # Fallback default if no explicit agreements/disagreements triggered
        if not agreement and sarcasm_pred:
            if sarcasm_pred == "not_sarcastic":
                agreement.append("Sarcasm model signals are consistent with literal / non-sarcastic reading.")

        return {
            "agreement": agreement,
            "disagreement": disagreement,
            "unavailable": unavailable
        }
