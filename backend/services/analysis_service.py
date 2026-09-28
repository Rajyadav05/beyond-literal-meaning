"""
backend/services/analysis_service.py
====================================
Phase 10 — Application-level Analysis Service.

Thin wrapper delegating unified pipeline execution to EvidenceFusionEngine.
Loaded once at application startup and reused across requests.
"""

import logging
from typing import Dict, Any, Optional

from src.explainability.evidence_fusion import EvidenceFusionEngine

logger = logging.getLogger("analysis_service")


class AnalysisService:
    def __init__(self):
        self.fusion_engine = None
        self.is_loaded = False
        self._load_engine()

    def _load_engine(self):
        try:
            self.fusion_engine = EvidenceFusionEngine()
            self.is_loaded = True
            logger.info("AnalysisService initialized successfully with EvidenceFusionEngine.")
        except Exception as e:
            logger.error("Failed to initialize AnalysisService: %s", e)
            self.is_loaded = False

    def analyze_sentence(self, sentence: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes unified NLP pipeline on sentence (and optional context).
        Rejects empty or whitespace-only inputs cleanly.
        """
        if not sentence or not sentence.strip():
            raise ValueError("Sentence must not be empty or whitespace-only.")

        if not self.is_loaded or self.fusion_engine is None:
            raise RuntimeError("AnalysisService / model weights are unavailable.")

        # Treat empty string / whitespace context as None
        cleaned_context = context.strip() if context and context.strip() else None

        return self.fusion_engine.analyze(sentence.strip(), context=cleaned_context)

    def compare_sentence_context(self, sentence: str, context: str) -> Dict[str, Any]:
        """
        Executes side-by-side context comparison (without_context vs with_context).
        """
        if not sentence or not sentence.strip():
            raise ValueError("Sentence must not be empty or whitespace-only.")

        if not context or not context.strip():
            raise ValueError("Context must not be empty or whitespace-only for comparison.")

        if not self.is_loaded or self.fusion_engine is None:
            raise RuntimeError("AnalysisService / model weights are unavailable.")

        return self.fusion_engine.compare_context(sentence.strip(), context.strip())
