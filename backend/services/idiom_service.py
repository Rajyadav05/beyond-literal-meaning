"""
backend/services/idiom_service.py
==================================
Service module for idiom candidate detection, classification, and meaning interpretation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from src.idiom.candidate_detector import IdiomCandidateDetector
from src.idiom.baseline import IdiomTfidfBaseline
from src.idiom.interpreter import IdiomInterpreter
from src.idiom.explainability import IdiomExplainer

logger = logging.getLogger("idiom_service")


class IdiomService:
    def __init__(self):
        self.detector = IdiomCandidateDetector()
        self.interpreter = IdiomInterpreter()
        self.explainer = IdiomExplainer()
        self.classifier = IdiomTfidfBaseline()
        self.model_loaded = False
        self._load_classifier()

    def _load_classifier(self):
        model_dir = PROJECT_ROOT / "models" / "idiom"
        try:
            if (model_dir / "tfidf_idiom_sentence_classifier.joblib").exists():
                self.classifier.load(model_dir, model_name="tfidf_idiom_sentence")
                self.model_loaded = True
                logger.info("Idiom classifier service loaded baseline model.")
            elif (model_dir / "tfidf_idiom_context_classifier.joblib").exists():
                self.classifier.load(model_dir, model_name="tfidf_idiom_context")
                self.model_loaded = True
                logger.info("Idiom classifier service loaded context baseline model.")
        except Exception as e:
            logger.warning("Could not load idiom classifier weights: %s", str(e))

    def analyze_idiom(self, sentence: str, context: Optional[str] = None) -> Dict[str, Any]:
        candidates = self.detector.detect_candidates(sentence)

        idioms_found = []
        for cand in candidates:
            phrase = cand["matched_text"]
            start_char = cand["start_char"]
            end_char = cand["end_char"]

            # Predict contextual usage
            if self.model_loaded:
                eval_text = f"{context} {sentence}" if context else sentence
                probs = self.classifier.predict_proba(pd_series_text([eval_text]))[0]
                prob_idiomatic = float(probs[1])
                is_idiomatic = prob_idiomatic >= 0.50
            else:
                # Fallback to rule/lexicon default if model is unavailable
                prob_idiomatic = 0.85
                is_idiomatic = True

            interpretation = self.interpreter.interpret(cand["expression"], sentence, is_idiomatic)
            explanation = self.explainer.explain(cand, is_idiomatic, prob_idiomatic,
                                                 model_name="TF-IDF + Logistic Regression" if self.model_loaded else "Lexicon Rule Classifier")

            idioms_found.append({
                "detected": True,
                "phrase": phrase,
                "span_start": start_char,
                "span_end": end_char,
                "literal_meaning": interpretation["literal_meaning"],
                "actual_meaning": interpretation["intended_meaning"],
                "meaning_in_context": interpretation["meaning_in_context"],
                "usage": interpretation["usage"],
                "idiomatic_probability": prob_idiomatic,
                "explanation": explanation
            })

        return {
            "sentence": sentence,
            "idioms_found": idioms_found,
            "total_found": len(idioms_found)
        }


def pd_series_text(texts: List[str]):
    import pandas as pd
    return pd.Series(texts)
