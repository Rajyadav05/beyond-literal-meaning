"""
backend/schemas.py
==================
All Pydantic v2 request and response models for the Non-Literal Language Lab API.

Every schema is designed to be:
- Fully JSON-serialisable
- Stable across API versions (additive changes only)
- Free of placeholder / fake fields

Values that require real model inference (probabilities, scores) are typed as
Optional[float] so they can be returned as null when a model is not yet loaded,
rather than returning fabricated numbers.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ===========================================================================
# Shared / utility models
# ===========================================================================

class TokenImportance(BaseModel):
    """Single token and its contribution to the model's prediction."""
    token: str = Field(..., description="The surface token (word or sub-word).")
    importance: float = Field(..., description="Importance score in [-1, 1] range.")
    reason: Optional[str] = Field(None, description="Human-readable reason string.")


class SentimentResult(BaseModel):
    """Predicted sentiment for a piece of text."""
    label: str = Field(..., description="e.g. 'POSITIVE', 'NEGATIVE', 'NEUTRAL'.")
    score: float = Field(..., description="Model confidence in [0, 1].")


class EmotionResult(BaseModel):
    """Top-k emotions predicted for a piece of text."""
    emotions: dict[str, float] = Field(
        ...,
        description="Mapping of emotion label to probability, e.g. {'joy': 0.72}.",
    )
    dominant: str = Field(..., description="The emotion with the highest probability.")


# ===========================================================================
# Request models
# ===========================================================================

class AnalyzeRequest(BaseModel):
    """
    POST /api/analyze
    Main analysis pipeline input.
    """
    sentence: str = Field(
        ...,
        min_length=1,
        description="The sentence to analyse. Required.",
    )
    context: Optional[str] = Field(
        None,
        description=(
            "Optional preceding conversational context. "
            "When provided, the model uses it for contextual sarcasm detection."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "context": "My laptop crashed five minutes before my presentation.",
                    "sentence": "Wow, exactly what I needed!",
                }
            ]
        }
    }


class ContextCompareRequest(BaseModel):
    """
    POST /api/analyze/context
    Side-by-side comparison of sentence-only vs context+sentence analysis.
    """
    sentence: str = Field(..., min_length=1)
    context: str = Field(
        ...,
        min_length=1,
        description="Context is REQUIRED for this endpoint — comparison is its purpose.",
    )


class IdiomRequest(BaseModel):
    """
    POST /api/idiom
    Standalone idiom detection and meaning request.
    """
    sentence: str = Field(..., min_length=1)
    context: Optional[str] = Field(None)


class SentimentRequest(BaseModel):
    """
    POST /api/sentiment
    Standalone sentiment classification request.
    """
    text: str = Field(..., min_length=1, description="Input text to classify.")


class SentimentResponse(BaseModel):
    """
    POST /api/sentiment -> response body.
    """
    text: str
    label: str = Field(..., description="'negative' | 'neutral' | 'positive'.")
    probabilities: dict[str, float] = Field(..., description="Mapping of sentiment label to probability.")


class EmotionRequest(BaseModel):
    """
    POST /api/emotion
    Standalone multi-label emotion prediction request.
    """
    text: str = Field(..., min_length=1, description="Input text to classify.")
    top_n: int = Field(5, ge=1, le=28, description="Number of top emotion rankings to return.")


class EmotionScore(BaseModel):
    label: str
    probability: float


class EmotionResponse(BaseModel):
    """
    POST /api/emotion -> response body.
    """
    text: str
    probabilities: dict[str, float] = Field(..., description="Full 28-label GoEmotions probability distribution.")
    top_emotions: list[EmotionScore] = Field(..., description="Sorted top-N emotion scores.")


class ChallengeSubmitRequest(BaseModel):
    """
    POST /api/challenge
    User evaluation submission request.
    """
    sentence: str = Field(..., min_length=1, description="Target sentence evaluated.")
    context: Optional[str] = Field(None, description="Optional preceding context.")
    model_prediction: str = Field(..., min_length=1, description="Prediction produced by model, e.g. 'sarcastic' or 'not_sarcastic'.")
    model_interpretation: Optional[str] = Field(None, description="Model-generated interpretation summary.")
    user_verdict: str = Field(..., description="User evaluation verdict: 'correct' or 'incorrect'.")
    user_feedback: Optional[str] = Field(None, max_length=500, description="Optional reviewer feedback note.")



# ===========================================================================
# Response models
# ===========================================================================

class LanguageTypeResult(BaseModel):
    sarcasm: bool
    idiom: bool


class SarcasmResult(BaseModel):
    probability: Optional[float] = Field(
        None,
        description="Model probability of the sarcastic class in [0, 1].",
    )
    label: str = Field(..., description="'sarcastic' or 'not_sarcastic'.")
    model_used: str = Field(..., description="Which model produced this result.")


class IdiomResult(BaseModel):
    detected: bool
    phrase: Optional[str] = Field(None, description="Matched idiomatic phrase.")
    span_start: Optional[int] = Field(None, description="Character offset start.")
    span_end: Optional[int] = Field(None, description="Character offset end.")
    literal_meaning: Optional[str] = None
    actual_meaning: Optional[str] = None
    meaning_in_context: Optional[str] = None
    usage: Optional[str] = Field(None, description="'idiomatic' or 'literal'.")
    idiomatic_probability: Optional[float] = Field(None, description="Model probability in [0, 1].")


class MeaningResult(BaseModel):
    """
    Predicted/estimated interpretations.
    These are model outputs, not ground-truth human understanding.
    """
    literal: str = Field(..., description="Surface / literal reading of the sentence.")
    intended: str = Field(
        ...,
        description="Model-estimated intended meaning (contextual interpretation).",
    )
    simple: str = Field(..., description="Plain-language summary.")


class ExplanationResult(BaseModel):
    influential_tokens: list[TokenImportance] = Field(default_factory=list)
    reasoning: list[str] = Field(
        default_factory=list,
        description="Linguistic reasoning steps, e.g. 'positive wording + negative context → incongruity'.",
    )


class AnalysisMetadata(BaseModel):
    model: str
    context_used: bool
    processing_time_ms: Optional[float] = None


class EvidenceItemSchema(BaseModel):
    source: str
    type: str
    signal: str
    value: Optional[Any] = None
    sarcasm_model_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    model_name: Optional[str] = None
    experiment_scope: Optional[str] = None
    experiment_type: Optional[str] = None
    training_subset_size: Optional[int] = None
    test_subset_size: Optional[int] = None
    probabilities: Optional[dict[str, float]] = None
    description: str
    supports: str
    limitations: str


class SignalAlignmentSchema(BaseModel):
    agreement: list[str] = Field(default_factory=list)
    disagreement: list[str] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)


class InterpretationSummarySchema(BaseModel):
    interpretation: str
    basis: list[str] = Field(default_factory=list)
    sarcasm_model_probability: Optional[float] = Field(None, ge=0.0, le=1.0)


class UnifiedAnalyzeResponse(BaseModel):
    """
    POST /api/analyze -> Unified Phase 10 Evidence Fusion Response body.
    """
    input_text: str
    context: Optional[str] = None
    evidence: list[EvidenceItemSchema]
    explanation_trace: list[dict[str, Any]]
    signal_alignment: SignalAlignmentSchema
    interpretation_summary: InterpretationSummarySchema


class UnifiedContextCompareResponse(BaseModel):
    """
    POST /api/analyze/context -> Side-by-side context comparison response body.
    """
    sentence: str
    context: str
    without_context: UnifiedAnalyzeResponse
    with_context: UnifiedAnalyzeResponse
    delta: dict[str, Any]


class APIErrorDetail(BaseModel):
    code: str
    message: str


class APIErrorResponse(BaseModel):
    error: APIErrorDetail


# ---------------------------------------------------------------------------
# Idiom response
# ---------------------------------------------------------------------------

class IdiomResponse(BaseModel):
    sentence: str
    idioms_found: list[IdiomResult]
    total_found: int


# ---------------------------------------------------------------------------
# Challenge responses
# ---------------------------------------------------------------------------

class ChallengeSubmitResponse(BaseModel):
    submission_id: int
    user_verdict: str
    created_at: str
    message: str = "Evaluation saved successfully."


class ChallengeStats(BaseModel):
    total_reviews: int
    correct: int
    incorrect: int
    accuracy: Optional[float] = Field(
        None,
        description="Correct / total_reviews; null when total_reviews == 0.",
    )


class RecentChallengeItem(BaseModel):
    id: int
    sentence: str
    context: Optional[str] = None
    model_prediction: str
    model_interpretation: Optional[str] = None
    user_verdict: str
    user_feedback: Optional[str] = None
    created_at: str


class ChallengeStatsResponse(BaseModel):
    stats: ChallengeStats
    recent: list[RecentChallengeItem] = Field(default_factory=list)



# ---------------------------------------------------------------------------
# Analytics / model status responses
# ---------------------------------------------------------------------------

class AnalyticsResponse(BaseModel):
    total_analyses: int
    sarcasm_detected_count: int
    idiom_detected_count: int
    avg_sarcasm_probability: Optional[float]
    most_common_emotions: list[dict[str, Any]] = Field(default_factory=list)


class ModelMetricsResponse(BaseModel):
    """
    GET /api/model/metrics
    Returns metrics loaded from results/metrics/*.json — never hard-coded.
    """
    available_experiments: list[str]
    metrics: dict[str, Any] = Field(
        ...,
        description="Raw content of saved metrics files, keyed by experiment name.",
    )


class ModelStatusEntry(BaseModel):
    loaded: bool
    path: Optional[str] = None
    error: Optional[str] = None


class ModelStatusResponse(BaseModel):
    sarcasm_distilbert: ModelStatusEntry
    sarcasm_baseline: ModelStatusEntry
    sentiment: ModelStatusEntry
    emotion: ModelStatusEntry
    embeddings: ModelStatusEntry


# ---------------------------------------------------------------------------
# Health check response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description="'ready' | 'degraded' | 'unavailable'.")
    offline_mode: bool = True
    models: dict[str, bool] = Field(
        ...,
        description="True if the model's weight files are present on disk.",
    )
    datasets: dict[str, bool] = Field(
        ...,
        description="True if the dataset directory contains at least one file.",
    )
    database: bool = Field(..., description="True if the SQLite database is reachable.")
    missing: list[str] = Field(
        default_factory=list,
        description="Human-readable list of missing components.",
    )
