"""
backend/main.py
===============
FastAPI application entry point for the Non-Literal Language Lab.

Phase 1 — Environment & Structure:
    Only the GET /api/health endpoint is fully implemented here.
    All other endpoints return HTTP 503 stubs so the frontend can be
    developed against stable URL paths before the ML pipeline is ready.

Startup checks:
    On application startup we verify that critical filesystem components
    exist (model directories, dataset directories, idiom dictionary,
    SQLite database).  Missing items are reported in the /api/health response
    rather than silently ignored.

CORS:
    Configured to allow all origins in development.  Set the
    ALLOWED_ORIGINS environment variable (comma-separated) to restrict in
    production / demo environments.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import (
    HealthResponse,
    AnalyzeRequest,
    ContextCompareRequest,
    UnifiedAnalyzeResponse,
    UnifiedContextCompareResponse,
    IdiomRequest,
    IdiomResponse,
    SentimentRequest,
    SentimentResponse,
    EmotionRequest,
    EmotionResponse,
    ChallengeSubmitRequest,
    ChallengeSubmitResponse,
    ChallengeStats,
    AnalyticsResponse,
    ModelMetricsResponse,
    ModelStatusResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers (all relative to the project root, never hard-coded)
# ---------------------------------------------------------------------------
# __file__ = beyond-literal-meaning/backend/main.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"

# Paths that must exist for the system to be considered "ready"
_REQUIRED_MODEL_DIRS: dict[str, Path] = {
    "sarcasm": MODELS_DIR / "sarcasm",
    "sentiment": MODELS_DIR / "sentiment",
    "emotion": MODELS_DIR / "emotion",
    "embeddings": MODELS_DIR / "embeddings",
    "idiom": MODELS_DIR / "idiom",
}
_REQUIRED_DATASET_DIRS: dict[str, Path] = {
    "sarc": DATASETS_DIR / "raw" / "sarc",
    "isarcasmeval": DATASETS_DIR / "raw" / "isarcasmeval",
    "magpie": DATASETS_DIR / "raw" / "magpie",
    "semeval_idiom": DATASETS_DIR / "raw" / "semeval_idiom",
    "goemotions": DATASETS_DIR / "raw" / "goemotions",
}

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def _get_allowed_origins() -> list[str]:
    """Read allowed CORS origins from the environment (comma-separated)."""
    raw = os.getenv("ALLOWED_ORIGINS", "")
    if raw.strip():
        return [o.strip() for o in raw.split(",") if o.strip()]
    # Default: allow everything in local dev
    return ["*"]


def create_app() -> FastAPI:
    application = FastAPI(
        title="Non-Literal Language Lab",
        description=(
            "Context-aware detection and interpretation of sarcasm and idioms. "
            "All model predictions come from locally loaded pretrained / fine-tuned models. "
            "No external inference APIs are used."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=_get_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return application


app = create_app()

# ---------------------------------------------------------------------------
# Startup health probe
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup_probe() -> None:
    """Log the readiness status of all required components at startup."""
    logger.info("=== Non-Literal Language Lab — startup check ===")
    logger.info("Project root: %s", PROJECT_ROOT)

    for name, path in _REQUIRED_MODEL_DIRS.items():
        status = "OK" if _dir_has_files(path) else "MISSING (run download_assets.py)"
        logger.info("  model/%s: %s", name, status)

    for name, path in _REQUIRED_DATASET_DIRS.items():
        status = "OK" if _dir_has_files(path) else "MISSING (run download_assets.py)"
        logger.info("  dataset/%s: %s", name, status)

    logger.info("=== startup check complete ===")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _dir_has_files(path: Path) -> bool:
    """Return True if *path* is an existing directory that contains at least one file."""
    if not path.is_dir():
        return False
    return any(p.is_file() for p in path.rglob("*"))


def _build_health_payload() -> dict[str, Any]:
    """Construct the full health payload by inspecting the filesystem."""
    models_status: dict[str, bool] = {}
    datasets_status: dict[str, bool] = {}
    missing: list[str] = []

    for name, path in _REQUIRED_MODEL_DIRS.items():
        ok = _dir_has_files(path)
        models_status[name] = ok
        if not ok:
            missing.append(f"model/{name}")

    for name, path in _REQUIRED_DATASET_DIRS.items():
        ok = _dir_has_files(path)
        datasets_status[name] = ok
        if not ok:
            missing.append(f"dataset/{name}")

    # Check database reachability
    db_ok = _check_database()
    if not db_ok:
        missing.append("database")

    all_models_ready = all(models_status.values())
    all_datasets_ready = all(datasets_status.values())

    if all_models_ready and all_datasets_ready and db_ok:
        overall = "ready"
    elif not all_models_ready or not all_datasets_ready:
        overall = "unavailable"
    else:
        overall = "degraded"

    return {
        "status": overall,
        "offline_mode": True,
        "models": models_status,
        "datasets": datasets_status,
        "database": db_ok,
        "missing": missing,
    }


def _check_database() -> bool:
    """Try to open (or create) the SQLite database and run a trivial query."""
    try:
        import sqlite3
        db_path = PROJECT_ROOT / "backend" / "lab.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("SELECT 1")
        conn.close()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database check failed: %s", exc)
        return False


def _not_implemented_response(endpoint: str) -> JSONResponse:
    """Return a consistent 503 for Phase-1 stub endpoints."""
    return JSONResponse(
        status_code=503,
        content={
            "error": "not_implemented",
            "message": (
                f"{endpoint} is not yet implemented. "
                "This stub is a placeholder for a future phase. "
                "See the README for the development phase roadmap."
            ),
        },
    )


# ===========================================================================
# Routes
# ===========================================================================

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="System and model readiness check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """
    Returns the readiness status of all components:
    - model weight directories
    - dataset directories
    - SQLite database

    The application reports `status: 'unavailable'` when models or datasets
    are missing, directing the operator to run `scripts/download_assets.py`.
    """
    payload = _build_health_payload()
    logger.info("Health check — status: %s", payload["status"])
    return HealthResponse(**payload)


# ---------------------------------------------------------------------------
# Services initialization (Singleton instances loaded once at startup)
# ---------------------------------------------------------------------------

from backend.services.idiom_service import IdiomService
from backend.services.analysis_service import AnalysisService
from src.sentiment.sentiment_service import SentimentService
from src.emotion.emotion_service import EmotionService

_idiom_service = IdiomService()
_sentiment_service = SentimentService()
_emotion_service = EmotionService()
_analysis_service = AnalysisService()


# ---------------------------------------------------------------------------
# Unified Analysis pipeline (Phase 10 implementation)
# ---------------------------------------------------------------------------

@app.post(
    "/api/analyze",
    response_model=UnifiedAnalyzeResponse,
    summary="Full unified NLP analysis pipeline",
    tags=["Analysis"],
)
async def analyze(request: AnalyzeRequest) -> Any:
    """
    **Phase 10** — Full unified NLP analysis pipeline.

    Accepts a sentence (and optional context) and returns structured
    evidence and predictions across Sarcasm, Idiom, Sentiment, Emotion,
    Semantic Similarity, Signal Alignment, and Interpretation.
    """
    if not request.sentence or not request.sentence.strip():
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "INVALID_INPUT", "message": "Sentence must not be empty or whitespace-only."}}
        )

    if not _analysis_service.is_loaded:
        return JSONResponse(status_code=503, content={"error": {"code": "SERVICE_UNAVAILABLE", "message": "Analysis service is unavailable."}})

    try:
        res = _analysis_service.analyze_sentence(request.sentence, request.context)
        return UnifiedAnalyzeResponse(**res)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": {"code": "INVALID_INPUT", "message": str(e)}})
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": {"code": "SERVICE_UNAVAILABLE", "message": str(e)}})
    except Exception as e:
        logger.exception("Unexpected error during /api/analyze: %s", e)
        return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "An unexpected error occurred."}})


@app.post(
    "/api/analyze/context",
    response_model=UnifiedContextCompareResponse,
    summary="Context vs no-context side-by-side comparison",
    tags=["Analysis"],
)
async def analyze_context(request: ContextCompareRequest) -> Any:
    """
    **Phase 10** — Side-by-side comparison of analysis with and without context.

    Returns `without_context`, `with_context`, and a `delta` object showing
    numeric differences and prediction changes when context is added.
    """
    if not request.sentence or not request.sentence.strip():
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "INVALID_INPUT", "message": "Sentence must not be empty or whitespace-only."}}
        )

    if not request.context or not request.context.strip():
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "INVALID_INPUT", "message": "Context must not be empty or whitespace-only for comparison."}}
        )

    if not _analysis_service.is_loaded:
        return JSONResponse(status_code=503, content={"error": {"code": "SERVICE_UNAVAILABLE", "message": "Analysis service is unavailable."}})

    try:
        res = _analysis_service.compare_sentence_context(request.sentence, request.context)
        return UnifiedContextCompareResponse(**res)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": {"code": "INVALID_INPUT", "message": str(e)}})
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"error": {"code": "SERVICE_UNAVAILABLE", "message": str(e)}})
    except Exception as e:
        logger.exception("Unexpected error during /api/analyze/context: %s", e)
        return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "An unexpected error occurred."}})


@app.post(
    "/api/idiom",
    response_model=IdiomResponse,
    summary="Standalone idiom detection",
    tags=["Idiom"],
)
async def detect_idiom(request: IdiomRequest) -> IdiomResponse:
    """
    **Phase 6** — Detect idiomatic phrases in a sentence and return
    span offsets, literal meaning, actual meaning, and contextual meaning.
    """
    result = _idiom_service.analyze_idiom(request.sentence, request.context)
    return IdiomResponse(**result)


# ---------------------------------------------------------------------------
# Sentiment & Emotion endpoints (Phase 7.3 implementation)
# ---------------------------------------------------------------------------

@app.post(
    "/api/sentiment",
    response_model=SentimentResponse,
    summary="Standalone sentiment classification",
    tags=["Sentiment"],
)
async def analyze_sentiment(request: SentimentRequest) -> SentimentResponse:
    """
    **Phase 7.3** — Perform local sentiment classification using
    cardiffnlp/twitter-roberta-base-sentiment-latest (negative, neutral, positive).
    """
    if not _sentiment_service.is_loaded:
        return JSONResponse(status_code=503, content={"error": "Sentiment model weights unavailable."})
    try:
        res = _sentiment_service.predict_sentiment(request.text)
        return SentimentResponse(**res)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})


@app.post(
    "/api/emotion",
    response_model=EmotionResponse,
    summary="Standalone multi-label emotion analysis",
    tags=["Emotion"],
)
async def analyze_emotion(request: EmotionRequest) -> EmotionResponse:
    """
    **Phase 7.3** — Perform local multi-label emotion prediction using
    SamLowe/roberta-base-go_emotions (28 native GoEmotions categories).
    """
    if not _emotion_service.is_loaded:
        return JSONResponse(status_code=503, content={"error": "Emotion model weights unavailable."})
    try:
        res = _emotion_service.predict_emotion(request.text, top_n=request.top_n)
        return EmotionResponse(**res)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Challenge system (Phase 12 implementation)
# ---------------------------------------------------------------------------

from backend.database import (
    init_db,
    record_challenge_evaluation,
    get_challenge_stats as query_challenge_stats,
    get_recent_challenge_evaluations,
)

@app.on_event("startup")
async def _init_database_tables() -> None:
    """Initialize database tables at application startup."""
    try:
        init_db()
        logger.info("SQLite database tables initialized successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database initialization warning: %s", exc)


@app.post(
    "/api/challenge",
    response_model=ChallengeSubmitResponse,
    summary="Submit a challenge evaluation verdict",
    tags=["Challenge"],
)
async def submit_challenge(request: ChallengeSubmitRequest) -> Any:
    """
    **Phase 12** — Submit a user's evaluation verdict ('correct' or 'incorrect')
    on a model prediction. Saved to SQLite `challenge_evaluations`.
    """
    if not request.sentence or not request.sentence.strip():
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "INVALID_INPUT", "message": "Sentence must not be empty or whitespace-only."}}
        )

    verdict_clean = request.user_verdict.strip().lower() if request.user_verdict else ""
    if verdict_clean not in ("correct", "incorrect"):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "INVALID_INPUT", "message": "user_verdict must be either 'correct' or 'incorrect'."}}
        )

    try:
        res = record_challenge_evaluation(
            sentence=request.sentence,
            context=request.context,
            model_prediction=request.model_prediction,
            model_interpretation=request.model_interpretation,
            user_verdict=verdict_clean,
            user_feedback=request.user_feedback,
        )
        return ChallengeSubmitResponse(
            submission_id=res["submission_id"],
            user_verdict=res["user_verdict"],
            created_at=res["created_at"],
            message="Evaluation saved successfully."
        )
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": {"code": "INVALID_INPUT", "message": str(e)}})
    except Exception as e:
        logger.exception("Unexpected error during /api/challenge: %s", e)
        return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "Failed to store evaluation."}})


@app.get(
    "/api/challenge/stats",
    summary="Challenge accuracy statistics & recent reviews",
    tags=["Challenge"],
)
async def challenge_stats() -> Any:
    """
    **Phase 12** — Return aggregate SQLite challenge statistics:
    total_reviews, correct, incorrect, accuracy, plus top 10 recent submissions.
    """
    try:
        stats = query_challenge_stats()
        recent_rows = get_recent_challenge_evaluations(limit=10)
        return {
            "stats": stats,
            "recent": recent_rows,
        }
    except Exception as e:
        logger.exception("Unexpected error during /api/challenge/stats: %s", e)
        return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "Failed to retrieve stats."}})


@app.get(
    "/api/challenge/recent",
    summary="Get recent challenge evaluation submissions",
    tags=["Challenge"],
)
async def challenge_recent() -> Any:
    """
    **Phase 12** — Bounded query returning the most recent 10 challenge evaluations.
    """
    try:
        return get_recent_challenge_evaluations(limit=10)
    except Exception as e:
        logger.exception("Unexpected error during /api/challenge/recent: %s", e)
        return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "Failed to retrieve recent evaluations."}})



# ---------------------------------------------------------------------------
# Analytics (Phase 13 stub)
# ---------------------------------------------------------------------------

@app.get(
    "/api/analytics",
    summary="Aggregate usage analytics",
    tags=["Analytics"],
)
async def analytics() -> JSONResponse:
    """
    **Phase 13** — Aggregate statistics over all analyses stored in SQLite:
    total analyses, sarcasm rate, idiom rate, average probabilities.
    """
    return _not_implemented_response("GET /api/analytics")


# ---------------------------------------------------------------------------
# Model metadata (Phase 11 stub)
# ---------------------------------------------------------------------------

@app.get(
    "/api/model/metrics",
    summary="Saved evaluation metrics",
    tags=["Model"],
)
async def model_metrics() -> JSONResponse:
    """
    **Phase 11** — Returns the contents of all JSON files under
    `results/metrics/`.  Metrics are written by `scripts/evaluate.py`
    and are never hard-coded.
    """
    metrics_dir = RESULTS_DIR / "metrics"
    if not metrics_dir.is_dir():
        return JSONResponse(
            status_code=404,
            content={
                "error": "no_metrics",
                "message": (
                    "No metrics found. Run scripts/evaluate.py after training."
                ),
            },
        )

    import json

    collected: dict[str, Any] = {}
    for json_file in sorted(metrics_dir.glob("*.json")):
        try:
            with open(json_file, encoding="utf-8") as fh:
                collected[json_file.stem] = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            collected[json_file.stem] = {"error": str(exc)}

    return JSONResponse(
        content={
            "available_experiments": list(collected.keys()),
            "metrics": collected,
        }
    )


@app.get(
    "/api/model/status",
    summary="Which models are currently loaded",
    tags=["Model"],
)
async def model_status() -> JSONResponse:
    """
    **Phase 11** — Reports which model weight files are present and loaded.
    Returns a structured status object for each model component.
    """
    statuses: dict[str, dict[str, Any]] = {}
    for name, path in _REQUIRED_MODEL_DIRS.items():
        has_files = _dir_has_files(path)
        statuses[name] = {
            "loaded": False,  # actual loading happens in Phase 11
            "weights_present": has_files,
            "path": str(path),
        }
    return JSONResponse(content={"models": statuses})


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": str(exc),
        },
    )
