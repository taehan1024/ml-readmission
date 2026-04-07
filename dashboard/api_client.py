"""dashboard/api_client.py

Thin HTTP client wrapping all FastAPI calls.
Every function returns a plain Python dict/list or raises on non-2xx.
No model imports — all data comes from the API.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Default timeout for all requests (seconds)
_TIMEOUT = 15


def _url(base: str, path: str) -> str:
    """Concatenate base URL and path, stripping accidental double slashes.

    Parameters
    ----------
    base:
        API base URL (e.g. ``http://localhost:8000``).
    path:
        Endpoint path (e.g. ``/health``).

    Returns
    -------
    str
        Full URL.
    """
    return base.rstrip("/") + "/" + path.lstrip("/")


def get_health(base_url: str) -> dict[str, Any]:
    """Call GET /health and return the parsed response body.

    Parameters
    ----------
    base_url:
        API base URL.

    Returns
    -------
    dict
        ``{"status": "ok", "model_loaded": true, "model_version": "1"}``.

    Raises
    ------
    requests.HTTPError
        On non-2xx response.
    requests.ConnectionError
        If the API is unreachable.
    """
    resp = requests.get(_url(base_url, "/health"), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_model_info(base_url: str) -> dict[str, Any]:
    """Call GET /model/info and return parsed metadata.

    Parameters
    ----------
    base_url:
        API base URL.

    Returns
    -------
    dict
        Model name, version, stage, training metrics, feature names.
    """
    resp = requests.get(_url(base_url, "/model/info"), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def predict_single(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call POST /predict with a single hospital payload.

    Parameters
    ----------
    base_url:
        API base URL.
    payload:
        HospitalInput-compatible dict.

    Returns
    -------
    dict
        ``{"risk_score": 0.72, "risk_level": "high", ...}``.
    """
    resp = requests.post(
        _url(base_url, "/predict"), json=payload, timeout=_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def predict_batch(
    base_url: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Call POST /predict/batch with a list of hospital records.

    Parameters
    ----------
    base_url:
        API base URL.
    records:
        List of HospitalInput-compatible dicts (max 500).

    Returns
    -------
    dict
        ``{"predictions": [...], "total": N, "high_risk_count": K, ...}``.
    """
    resp = requests.post(
        _url(base_url, "/predict/batch"), json=records, timeout=_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def get_history(base_url: str, limit: int = 100) -> dict[str, Any]:
    """Call GET /predictions/history.

    Parameters
    ----------
    base_url:
        API base URL.
    limit:
        Number of recent records to fetch (1–1000).

    Returns
    -------
    dict
        ``{"stats": {...}, "predictions": [...]}``.
    """
    resp = requests.get(
        _url(base_url, "/predictions/history"),
        params={"limit": limit},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()
