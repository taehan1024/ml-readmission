"""app/monitoring.py

Prediction monitoring via SQLite.

Every call to ``/predict`` or ``/predict/batch`` is logged to a local
SQLite database so you can track prediction distributions over time,
detect drift, and audit individual decisions.

The database is created automatically on first use.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from app.config import settings

logger = logging.getLogger(__name__)

# SQL schema ──────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    facility_id     TEXT,
    facility_name   TEXT,
    state           TEXT,
    input_features  TEXT    NOT NULL,   -- JSON blob of feature values
    risk_score      REAL    NOT NULL,
    risk_level      TEXT    NOT NULL,
    model_version   TEXT    NOT NULL
);
"""

_INSERT_ROW = """
INSERT INTO predictions
    (timestamp, facility_id, facility_name, state,
     input_features, risk_score, risk_level, model_version)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_RECENT = """
SELECT id, timestamp, facility_id, facility_name, state,
       input_features, risk_score, risk_level, model_version
FROM   predictions
ORDER  BY id DESC
LIMIT  ?;
"""

_SELECT_STATS = """
SELECT
    COUNT(*)                          AS total,
    AVG(risk_score)                   AS mean_risk_score,
    SUM(CASE WHEN risk_level='high'   THEN 1 ELSE 0 END) AS high_count,
    SUM(CASE WHEN risk_level='medium' THEN 1 ELSE 0 END) AS medium_count,
    SUM(CASE WHEN risk_level='low'    THEN 1 ELSE 0 END) AS low_count
FROM predictions;
"""


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db_path() -> Path:
    """Return the absolute path to the monitoring database.

    Returns
    -------
    Path
        Absolute path derived from ``settings.monitoring_db_abs_path``.
    """
    return settings.monitoring_db_abs_path


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    """Yield an auto-committing SQLite connection, creating the DB if needed.

    Yields
    ------
    sqlite3.Connection
    """
    db = _db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_CREATE_TABLE)
        conn.commit()
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Public API ────────────────────────────────────────────────────────────────

def log_prediction(
    facility_id: str | None,
    facility_name: str | None,
    state: str | None,
    input_features: dict,
    risk_score: float,
    risk_level: str,
    model_version: str,
) -> None:
    """Persist a single prediction to the monitoring database.

    Parameters
    ----------
    facility_id:
        CMS provider number (may be ``None``).
    facility_name:
        Hospital name (may be ``None``).
    state:
        Two-letter state code (may be ``None``).
    input_features:
        Dict of feature name → value sent to the model.
    risk_score:
        Model output probability in [0, 1].
    risk_level:
        ``"low"``, ``"medium"``, or ``"high"``.
    model_version:
        String identifier of the model version used.
    """
    if not settings.log_predictions:
        return

    try:
        with _get_conn() as conn:
            conn.execute(
                _INSERT_ROW,
                (
                    datetime.now(timezone.utc).isoformat(),
                    facility_id,
                    facility_name,
                    state,
                    json.dumps(input_features),
                    risk_score,
                    risk_level,
                    model_version,
                ),
            )
    except Exception:
        # Monitoring must never crash the prediction endpoint
        logger.exception("Failed to log prediction to monitoring DB")


def get_recent_predictions(limit: int = 100) -> list[dict]:
    """Return the most recent predictions from the monitoring database.

    Parameters
    ----------
    limit:
        Maximum number of rows to return (default 100, max 1000).

    Returns
    -------
    list[dict]
        List of prediction records, newest first.
    """
    limit = min(limit, 1000)
    try:
        with _get_conn() as conn:
            rows = conn.execute(_SELECT_RECENT, (limit,)).fetchall()
        result = []
        for row in rows:
            record = dict(row)
            # Deserialise the JSON features blob
            try:
                record["input_features"] = json.loads(record["input_features"])
            except (json.JSONDecodeError, TypeError):
                pass
            result.append(record)
        return result
    except Exception:
        logger.exception("Failed to retrieve predictions from monitoring DB")
        return []


def get_prediction_stats() -> dict:
    """Return aggregate statistics over all logged predictions.

    Returns
    -------
    dict
        Keys: ``total``, ``mean_risk_score``, ``high_count``,
        ``medium_count``, ``low_count``.
    """
    try:
        with _get_conn() as conn:
            row = conn.execute(_SELECT_STATS).fetchone()
        if row:
            return {
                "total": row["total"] or 0,
                "mean_risk_score": round(row["mean_risk_score"] or 0.0, 4),
                "high_count": row["high_count"] or 0,
                "medium_count": row["medium_count"] or 0,
                "low_count": row["low_count"] or 0,
            }
    except Exception:
        logger.exception("Failed to retrieve prediction stats")
    return {"total": 0, "mean_risk_score": 0.0, "high_count": 0,
            "medium_count": 0, "low_count": 0}
