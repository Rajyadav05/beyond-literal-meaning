"""
src/semantics/semantic_interpreter.py
=====================================
Phase 8 — Semantic Meaning Interpretation Layer.

Integrates Phase 6 idiom candidate detection & lexicon lookup with Phase 8 sentence embeddings.
Provides structured literal, idiomatic, and contextual interpretations without hallucinating or overriding meanings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.idiom.candidate_detector import IdiomCandidateDetector
from src.idiom.interpreter import IdiomInterpreter
from src.semantics.embedding_service import EmbeddingService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("semantic_interpreter")


class SemanticInterpreter:
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        candidate_detector: Optional[IdiomCandidateDetector] = None,
        idiom_interpreter: Optional[IdiomInterpreter] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.candidate_detector = candidate_detector or IdiomCandidateDetector()
        self.idiom_interpreter = idiom_interpreter or IdiomInterpreter()

    def interpret(self, sentence: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyzes a sentence (and optional context) for semantic representation,
        context similarity, and idiom meaning interpretations.
        """
        if not sentence or not sentence.strip():
            raise ValueError("Input sentence cannot be empty or whitespace-only.")

        sentence_clean = sentence.strip()
        context_clean = context.strip() if context and context.strip() else None

        # 1. Compute Semantic Similarity if Context is Provided
        context_similarity = None
        if context_clean and self.embedding_service.is_loaded:
            sim_res = self.embedding_service.similarity(context_clean, sentence_clean)
            context_similarity = sim_res["similarity"]

        # 2. Candidate Idiom Detection via Phase 6 Detector
        candidates = self.candidate_detector.detect_candidates(sentence_clean)
        
        idiom_results = []
        has_lexicon_match = False

        for cand in candidates:
            phrase = cand["matched_text"]
            expression = cand["expression"]
            start_char = cand["start_char"]
            end_char = cand["end_char"]

            # Lookup via Phase 6 IdiomInterpreter
            norm_expr = expression.lower().strip()
            lexicon_entry = self.idiom_interpreter.lexicon.get(norm_expr, {})

            if lexicon_entry and lexicon_entry.get("has_dictionary_meaning", False):
                has_lexicon_match = True
                source = "lexicon_lookup"
                literal_meaning = lexicon_entry.get("literal_meaning", f"Literal interpretation of '{expression}'.")
                idiomatic_meaning = lexicon_entry.get("idiomatic_meaning", f"Figurative meaning of '{expression}'.")

                if context_clean:
                    contextual_interpretation = (
                        f"In context ('{context_clean}'), the phrase '{phrase}' is interpreted idiomatically "
                        f"as: '{idiomatic_meaning}'."
                    )
                else:
                    contextual_interpretation = f"The phrase '{phrase}' is interpreted idiomatically as: '{idiomatic_meaning}'."
            else:
                source = "unknown"
                literal_meaning = f"Literal reading of '{phrase}'."
                idiomatic_meaning = "Meaning unavailable from the current local idiom knowledge base."
                contextual_interpretation = "Meaning unavailable from the current local idiom knowledge base."

            idiom_results.append({
                "detected": True,
                "phrase": phrase,
                "start": start_char,
                "end": end_char,
                "literal_meaning": literal_meaning,
                "idiomatic_meaning": idiomatic_meaning,
                "contextual_interpretation": contextual_interpretation,
                "meaning_source": source
            })

        return {
            "input_text": sentence_clean,
            "context": context_clean,
            "embedding_dimension": self.embedding_service.embedding_dimension if self.embedding_service.is_loaded else 384,
            "semantic_similarity": {
                "context_sentence": context_similarity
            },
            "idiom_analysis": {
                "idioms_found": idiom_results,
                "total_found": len(idiom_results)
            },
            "evidence": {
                "lexicon_match": has_lexicon_match,
                "semantic_similarity_available": context_similarity is not None,
                "context_available": context_clean is not None
            }
        }
