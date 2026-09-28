"""
backend/database.py
===================
SQLite database storage and query handling for the Non-Literal Language Lab.

Provides connection pooling, table initialization, and CRUD methods for user
challenge feedback evaluations and statistics.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "backend" / "lab.db"


def get_db_connection(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Return a SQLite connection with Row factory enabled."""
    raw_path = db_path if db_path is not None else DEFAULT_DB_PATH
    target_path = Path(raw_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn



def init_db(db_path: Optional[str | Path] = None) -> None:
    """
    Initialize the database schema if tables do not exist.
    Creates the `challenge_evaluations` table.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS challenge_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sentence TEXT NOT NULL,
                    context TEXT,
                    model_prediction TEXT NOT NULL,
                    model_interpretation TEXT,
                    user_verdict TEXT NOT NULL CHECK(user_verdict IN ('correct', 'incorrect')),
                    user_feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_challenge_verdict ON challenge_evaluations(user_verdict);
                """
            )
    finally:
        conn.close()


def record_challenge_evaluation(
    sentence: str,
    model_prediction: str,
    user_verdict: str,
    context: Optional[str] = None,
    model_interpretation: Optional[str] = None,
    user_feedback: Optional[str] = None,
    db_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """
    Insert a new challenge evaluation into `challenge_evaluations`.

    Returns a dict containing submission_id, user_verdict, and created_at ISO string.
    """
    if not sentence or not sentence.strip():
        raise ValueError("Sentence must not be empty or whitespace-only.")

    verdict_clean = user_verdict.strip().lower()
    if verdict_clean not in ("correct", "incorrect"):
        raise ValueError("user_verdict must be either 'correct' or 'incorrect'.")

    created_iso = datetime.now(timezone.utc).isoformat()
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO challenge_evaluations (
                    sentence, context, model_prediction, model_interpretation,
                    user_verdict, user_feedback, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    sentence.strip(),
                    context.strip() if context and context.strip() else None,
                    model_prediction.strip(),
                    model_interpretation.strip() if model_interpretation and model_interpretation.strip() else None,
                    verdict_clean,
                    user_feedback.strip() if user_feedback and user_feedback.strip() else None,
                    created_iso,
                ),
            )
            submission_id = cursor.lastrowid
            return {
                "submission_id": submission_id,
                "user_verdict": verdict_clean,
                "created_at": created_iso,
            }
    finally:
        conn.close()


def get_challenge_stats(db_path: Optional[str | Path] = None) -> dict[str, Any]:
    """
    Calculate and return aggregate challenge statistics:
    total_reviews, correct, incorrect, and accuracy (correct / total_reviews).
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total FROM challenge_evaluations;")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) AS total_correct FROM challenge_evaluations WHERE user_verdict = 'correct';")
        correct = cursor.fetchone()["total_correct"]

        cursor.execute("SELECT COUNT(*) AS total_incorrect FROM challenge_evaluations WHERE user_verdict = 'incorrect';")
        incorrect = cursor.fetchone()["total_incorrect"]

        accuracy = round(correct / total, 4) if total > 0 else None

        return {
            "total_reviews": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy,
        }
    finally:
        conn.close()


def get_recent_challenge_evaluations(
    limit: int = 10, db_path: Optional[str | Path] = None
) -> list[dict[str, Any]]:
    """
    Query and return a bounded list of recent challenge evaluations ordered by ID desc.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, sentence, context, model_prediction, model_interpretation,
                   user_verdict, user_feedback, created_at
            FROM challenge_evaluations
            ORDER BY id DESC
            LIMIT ?;
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
