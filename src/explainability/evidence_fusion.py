"""
src/explainability/evidence_fusion.py
======================================
Phase 9 — Transparent Evidence Fusion & Explainability Engine.

Aggregates independently produced predictions and evidence from:
1. Sarcasm Classifiers (TF-IDF Baseline + DistilBERT Exploratory Model)
2. Idiom Pipeline (Candidate Detector + Interpreter + Lexicon)
3. Sentiment Service (3-Class Surface Sentiment)
4. Emotion Service (28 Native GoEmotions Labels)
5. Semantic Embedding Service (384-dim all-MiniLM-L6-v2 Cosine Similarity)

Provides structured evidence representations, inspectable explanation traces,
conflict/agreement signal summaries, and context ON/OFF comparisons without
fabricating ensemble weights or artificial confidence scores.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.explainability.conflict_detector import ConflictDetector
from src.idiom.candidate_detector import IdiomCandidateDetector
from src.idiom.interpreter import IdiomInterpreter
from src.sarcasm.tfidf_baseline import build_tfidf_pipeline
from src.semantics.embedding_service import EmbeddingService
from src.semantics.semantic_interpreter import SemanticInterpreter
from src.sentiment.sentiment_service import SentimentService
from src.emotion.emotion_service import EmotionService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("evidence_fusion")

# Lightweight lexical keywords for contextual incongruity check
ADVERSE_CONTEXT_KEYWORDS = [
    "crashed", "pool", "fell", "waiting", "three hours", "delay", "broken",
    "fail", "failed", "error", "lost", "stolen", "late", "damage", "worst"
]
POSITIVE_WORDING_KEYWORDS = [
    "love", "wonderful", "great", "excellent", "exactly what i needed", "wow", "amazing", "piece of cake"
]


class EvidenceFusionEngine:
    def __init__(self):
        self.conflict_detector = ConflictDetector()
        
        # Load Services
        self.sentiment_service = SentimentService()
        self.emotion_service = EmotionService()
        self.embedding_service = EmbeddingService()
        self.semantic_interpreter = SemanticInterpreter(
            embedding_service=self.embedding_service
        )
        self.idiom_detector = IdiomCandidateDetector()
        self.idiom_interpreter = IdiomInterpreter()

        # Load Sarcasm Models
        self.tfidf_model = None
        self.distilbert_tokenizer = None
        self.distilbert_model = None
        self._load_sarcasm_models()

    def _load_sarcasm_models(self):
        # 1. Load TF-IDF Baseline
        tfidf_path = PROJECT_ROOT / "models" / "sarcasm" / "tfidf_logreg_sentence.joblib"
        if tfidf_path.exists():
            try:
                self.tfidf_model = joblib.load(tfidf_path)
                logger.info("Loaded TF-IDF Sarcasm model from %s", tfidf_path)
            except Exception as e:
                logger.warning("Could not load TF-IDF Sarcasm model: %s", e)

        # 2. Load DistilBERT Sarcasm Model (Exploratory Subset)
        distilbert_path = PROJECT_ROOT / "models" / "sarcasm" / "distilbert_sentence"
        if distilbert_path.exists() and (distilbert_path / "config.json").exists():
            try:
                self.distilbert_tokenizer = AutoTokenizer.from_pretrained(str(distilbert_path), local_files_only=True)
                self.distilbert_model = AutoModelForSequenceClassification.from_pretrained(str(distilbert_path), local_files_only=True)
                self.distilbert_model.to(torch.device("cpu"))
                self.distilbert_model.eval()
                logger.info("Loaded DistilBERT Sarcasm model from %s", distilbert_path)
            except Exception as e:
                logger.warning("Could not load DistilBERT Sarcasm model: %s", e)

    def analyze(self, sentence: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Main evidence fusion method.
        Collects model predictions, constructs evidence list, builds explanation trace,
        detects conflicts, and returns a transparent interpretation summary.
        """
        if not sentence or not sentence.strip():
            raise ValueError("Input sentence cannot be empty or whitespace-only.")

        sentence_clean = sentence.strip()
        context_clean = context.strip() if context and context.strip() else None
        has_context = context_clean is not None

        evidence_list: List[Dict[str, Any]] = []
        explanation_trace: List[Dict[str, Any]] = []

        # Step 1: Input & Context Trace
        explanation_trace.append({
            "stage": "input",
            "observation": sentence_clean,
            "role": "target_sentence"
        })
        if has_context:
            explanation_trace.append({
                "stage": "context",
                "observation": context_clean,
                "role": "preceding_conversational_context"
            })

        # Step 2: Sarcasm Classifier Evidence
        sarcasm_info = self._get_sarcasm_predictions(sentence_clean, context_clean)
        if sarcasm_info:
            for item in sarcasm_info["evidence"]:
                evidence_list.append(item)
            explanation_trace.append({
                "stage": "sarcasm",
                "observation": sarcasm_info["primary_prediction"],
                "probabilities": sarcasm_info["probabilities"],
                "model_used": sarcasm_info["primary_model"],
                "role": "sarcasm_classification"
            })

        # Step 3: Idiom Pipeline Evidence
        semantic_res = self.semantic_interpreter.interpret(sentence_clean, context_clean)
        idiom_found = semantic_res["idiom_analysis"]["idioms_found"]
        
        evidence_list.append({
            "source": "idiom_detector",
            "type": "idiom_evidence",
            "signal": "idiom_candidates",
            "value": idiom_found,
            "description": f"Detected {len(idiom_found)} idiom candidate(s).",
            "supports": "figurative_or_literal_idiom_usage",
            "limitations": "Candidate detection relies on greedy n-gram lexicon matching."
        })
        explanation_trace.append({
            "stage": "idiom",
            "observation": f"{len(idiom_found)} idiom(s) detected",
            "candidates": [item["phrase"] for item in idiom_found],
            "role": "idiom_detection_and_meaning"
        })

        # Step 4: Surface Sentiment Evidence
        sentiment_res = self.sentiment_service.predict_sentiment(sentence_clean)
        evidence_list.append({
            "source": "sentiment_model",
            "type": "model_prediction",
            "signal": "surface_sentiment",
            "value": sentiment_res["label"],
            "probabilities": sentiment_res["probabilities"],
            "description": f"Surface-level sentiment predicted as '{sentiment_res['label']}'.",
            "supports": "surface_emotional_polarity",
            "limitations": "Surface sentiment model evaluates literal token statistics without context or sarcasm awareness."
        })
        explanation_trace.append({
            "stage": "sentiment",
            "observation": sentiment_res["label"],
            "probabilities": sentiment_res["probabilities"],
            "role": "surface_sentiment"
        })

        # Step 5: Emotion Evidence (GoEmotions 28 Native Labels)
        emotion_res = self.emotion_service.predict_emotion(sentence_clean, top_n=5)
        evidence_list.append({
            "source": "emotion_model",
            "type": "emotion_evidence",
            "signal": "top_emotions",
            "value": emotion_res["top_emotions"],
            "description": f"Top emotion predicted as '{emotion_res['top_emotions'][0]['label']}' ({emotion_res['top_emotions'][0]['probability']:.4f}).",
            "supports": "fine_grained_emotion_distribution",
            "limitations": "Independent multi-label Sigmoid predictions over 28 GoEmotions native categories."
        })
        explanation_trace.append({
            "stage": "emotion",
            "observation": emotion_res["top_emotions"][0]["label"],
            "top_5": emotion_res["top_emotions"],
            "role": "fine_grained_emotion"
        })

        # Step 6: Semantic Embedding & Context Similarity
        if has_context:
            sim_val = semantic_res["semantic_similarity"]["context_sentence"]
            evidence_list.append({
                "source": "semantic_similarity",
                "type": "semantic_similarity",
                "signal": "cosine_similarity",
                "value": sim_val,
                "description": f"Vector-space cosine similarity between context and sentence embeddings is {sim_val:.4f}.",
                "supports": "context_sentence_semantic_relationship",
                "limitations": "Cosine similarity measures vector distance in embedding space, not human semantic truth."
            })
            explanation_trace.append({
                "stage": "semantics",
                "observation": f"cosine_similarity = {sim_val:.4f}",
                "role": "semantic_distance"
            })

            # Check Contextual Incongruity Signal
            context_lower = context_clean.lower()
            sent_lower = sentence_clean.lower()
            is_adverse_context = any(kw in context_lower for kw in ADVERSE_CONTEXT_KEYWORDS)
            has_positive_wording = any(kw in sent_lower for kw in POSITIVE_WORDING_KEYWORDS) or sentiment_res["label"] == "positive"

            if is_adverse_context:
                evidence_list.append({
                    "source": "contextual_signal",
                    "type": "contextual_signal",
                    "signal": "adverse_context",
                    "value": True,
                    "description": "Preceding context describes an adverse, inconvenient, or negative situation.",
                    "supports": "context_incongruity_analysis",
                    "limitations": "Keyword and sentiment-based situational heuristic."
                })
                explanation_trace.append({
                    "stage": "contextual_signals",
                    "observation": "adverse_context_detected",
                    "role": "situational_polarity"
                })
        else:
            explanation_trace.append({
                "stage": "semantics",
                "observation": "context_unavailable",
                "role": "semantic_distance"
            })

        # Step 7: Conflict & Agreement Analysis
        conflict_res = self.conflict_detector.analyze_conflicts(evidence_list, has_context)
        explanation_trace.append({
            "stage": "conflict_analysis",
            "agreement_count": len(conflict_res["agreement"]),
            "disagreement_count": len(conflict_res["disagreement"]),
            "role": "signal_alignment"
        })

        # Step 8: Interpretation Summary Generation
        summary = self._generate_interpretation_summary(
            sentence_clean, context_clean, sarcasm_info, sentiment_res, idiom_found, conflict_res
        )

        explanation_trace.append({
            "stage": "interpretation",
            "observation": summary["interpretation"],
            "role": "model_generated_interpretation"
        })

        return {
            "input_text": sentence_clean,
            "context": context_clean,
            "evidence": evidence_list,
            "explanation_trace": explanation_trace,
            "signal_alignment": conflict_res,
            "interpretation_summary": summary
        }

    def _get_sarcasm_predictions(self, sentence: str, context: Optional[str]) -> Optional[Dict[str, Any]]:
        evidence = []
        primary_pred = "not_sarcastic"
        primary_model = "TF-IDF + Logistic Regression"
        probs_dict = {}

        # 1. TF-IDF Baseline Prediction
        if self.tfidf_model:
            eval_text = f"{context} {sentence}" if context else sentence
            probs = self.tfidf_model.predict_proba([eval_text])[0]
            prob_sarcastic = float(probs[1])
            pred_label = "sarcastic" if prob_sarcastic >= 0.50 else "not_sarcastic"
            
            primary_pred = pred_label
            probs_dict["tfidf_sarcastic"] = prob_sarcastic

            evidence.append({
                "source": "sarcasm_classifier",
                "type": "model_prediction",
                "signal": "sarcasm_prediction_tfidf",
                "value": pred_label,
                "sarcasm_model_probability": prob_sarcastic,
                "model_name": "TF-IDF + Logistic Regression Baseline",
                "experiment_scope": "Full Dataset Benchmark (807,969 train rows)",
                "description": f"TF-IDF model predicts '{pred_label}' (sarcasm probability: {prob_sarcastic:.4f}).",
                "supports": "sarcasm_detection",
                "limitations": "Bag-of-words / n-gram TF-IDF baseline without deep transformer self-attention."
            })

        # 2. DistilBERT Exploratory Prediction (if available)
        if self.distilbert_model and self.distilbert_tokenizer:
            inputs = self.distilbert_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                logits = self.distilbert_model(**inputs).logits
                probs_bert = torch.softmax(logits, dim=-1).squeeze(0).numpy()
            
            prob_bert_sarcastic = float(probs_bert[1])
            pred_bert_label = "sarcastic" if prob_bert_sarcastic >= 0.50 else "not_sarcastic"
            probs_dict["distilbert_sarcastic"] = prob_bert_sarcastic

            evidence.append({
                "source": "sarcasm_classifier_distilbert",
                "type": "model_prediction",
                "signal": "sarcasm_prediction_distilbert",
                "value": pred_bert_label,
                "sarcasm_model_probability": prob_bert_sarcastic,
                "model_name": "DistilBERT Transformer",
                "experiment_type": "exploratory_subset",
                "training_subset_size": 1000,
                "test_subset_size": 500,
                "description": f"DistilBERT exploratory model predicts '{pred_bert_label}' (probability: {prob_bert_sarcastic:.4f}).",
                "supports": "sarcasm_detection_transformer",
                "limitations": "Fine-tuned on CPU-safe 1,000-example exploratory subset; not a full-dataset benchmark."
            })

        return {
            "primary_prediction": primary_pred,
            "primary_model": primary_model,
            "probabilities": probs_dict,
            "evidence": evidence
        }

    def _generate_interpretation_summary(
        self,
        sentence: str,
        context: Optional[str],
        sarcasm_info: Optional[Dict[str, Any]],
        sentiment_res: Dict[str, Any],
        idiom_found: List[Dict[str, Any]],
        conflict_res: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        
        basis = []
        is_sarcastic = sarcasm_info and sarcasm_info["primary_prediction"] == "sarcastic"
        
        if sarcasm_info:
            basis.append(f"Sarcasm model prediction ({sarcasm_info['primary_prediction']})")

        if idiom_found:
            basis.append(f"Detected idiom phrase: '{idiom_found[0]['phrase']}'")

        if sentiment_res:
            basis.append(f"Surface sentiment: '{sentiment_res['label']}'")

        if context:
            basis.append(f"Contextual situation: '{context}'")

        # Contextual Sarcasm vs Idiom vs Literal Synthesis
        if is_sarcastic and context:
            interpretation = "Likely sarcastic expression of dissatisfaction or ironical comment."
        elif idiom_found:
            idiom_meaning = idiom_found[0]["idiomatic_meaning"]
            interpretation = f"Idiomatic expression interpreted as: '{idiom_meaning}'."
        elif sentiment_res["label"] == "positive":
            interpretation = "Literal positive expression of emotion or satisfaction."
        elif sentiment_res["label"] == "negative":
            interpretation = "Literal negative expression of dissatisfaction."
        else:
            interpretation = "Literal neutral observation."

        sarc_prob = sarcasm_info["probabilities"].get("tfidf_sarcastic") if sarcasm_info else None

        return {
            "interpretation": interpretation,
            "basis": basis,
            "sarcasm_model_probability": sarc_prob
        }

    def compare_context(self, sentence: str, context: str) -> Dict[str, Any]:
        """
        Performs side-by-side evidence fusion comparison between:
        1. sentence WITHOUT context
        2. sentence WITH context
        Calculates exact deltas without overclaiming 'performance improvement'.
        """
        without_ctx = self.analyze(sentence, context=None)
        with_ctx = self.analyze(sentence, context=context)

        prob_no_ctx = without_ctx["interpretation_summary"].get("sarcasm_model_probability")
        prob_with_ctx = with_ctx["interpretation_summary"].get("sarcasm_model_probability")

        delta_prob = (prob_with_ctx - prob_no_ctx) if (prob_with_ctx is not None and prob_no_ctx is not None) else 0.0
        pred_no_ctx = without_ctx["explanation_trace"][2].get("observation") if len(without_ctx["explanation_trace"]) > 2 else None
        pred_with_ctx = with_ctx["explanation_trace"][2].get("observation") if len(with_ctx["explanation_trace"]) > 2 else None

        sim_val = with_ctx["evidence"][-2].get("value") if len(with_ctx["evidence"]) >= 2 else None

        return {
            "sentence": sentence,
            "context": context,
            "without_context": without_ctx,
            "with_context": with_ctx,
            "delta": {
                "sarcasm_probability_without_context": prob_no_ctx,
                "sarcasm_probability_with_context": prob_with_ctx,
                "sarcasm_probability_delta": float(delta_prob),
                "semantic_similarity": sim_val,
                "prediction_changed": pred_no_ctx != pred_with_ctx
            }
        }
