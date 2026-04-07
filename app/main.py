"""app/main.py

FastAPI application for the Hospital Readmission Prediction API.

Endpoints
---------
GET  /health                — liveness check + model status
GET  /model/info            — model metadata and training metrics
POST /predict               — single hospital risk prediction
POST /predict/batch         — batch prediction with summary stats
GET  /predictions/history   — last N logged predictions

Run locally
-----------
    uvicorn app.main:app --reload --port 8000
    # then open http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from app.config import settings
from app.model import model_manager
from app.monitoring import get_prediction_stats, get_recent_predictions, log_prediction
from app.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    HospitalInput,
    ModelInfoResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; nothing to clean up on shutdown."""
    logger.info("Loading model …")
    try:
        model_manager.load(
            model_name=settings.model_name,
            model_stage=settings.model_stage,
            local_path=settings.model_local_abs_path,
        )
        logger.info(
            "Model ready — version=%s stage=%s",
            model_manager.model_version,
            model_manager.model_stage,
        )
    except RuntimeError as exc:
        # Log the error but let the app start so /health returns degraded status
        logger.error("Model failed to load: %s", exc)
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description=(
        "Predict 30-day readmission risk for US hospitals using CMS "
        "Hospital Readmissions Reduction Program metrics. "
        "Risk score ∈ [0, 1]; higher = more likely to exceed expected readmissions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Operations"],
    summary="Service liveness check",
)
def health() -> HealthResponse:
    """Return service status and whether a model is loaded.

    Returns ``status: "ok"`` when a model is loaded and ready to serve
    predictions. Returns ``status: "degraded"`` when the model failed to
    load at startup (predictions will be unavailable).
    """
    return HealthResponse(
        status="ok" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        model_version=model_manager.model_version,
    )


# ── Model info ────────────────────────────────────────────────────────────────

@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Operations"],
    summary="Model metadata and training metrics",
)
def model_info() -> ModelInfoResponse:
    """Return model metadata including training metrics and feature list.

    Raises 503 if no model is currently loaded.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check /health for details.",
        )
    return ModelInfoResponse(
        model_name=settings.model_name,
        model_version=model_manager.model_version,
        model_stage=model_manager.model_stage,
        training_metrics=model_manager.training_metrics,
        feature_importances=model_manager.feature_importances,
        feature_count=len(model_manager.feature_names),
        feature_names=model_manager.feature_names,
    )


# ── Single prediction ─────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict readmission risk for a single hospital",
)
def predict(hospital: HospitalInput) -> PredictionResponse:
    """Score a single hospital's readmission risk.

    Accepts per-measure CMS readmission metrics and returns a probability
    score plus a human-readable risk tier (low / medium / high).

    - **risk_score** < 0.40 → low
    - **risk_score** 0.40–0.65 → medium
    - **risk_score** > 0.65 → high

    Raises 503 if the model is not loaded.
    Raises 422 if input validation fails (missing required measures, etc.).
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check /health for details.",
        )

    try:
        data = hospital.model_dump()
        risk_score, risk_level = model_manager.predict_single(data)
    except Exception as exc:
        logger.exception("Prediction failed for facility_id=%s", hospital.facility_id)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    log_prediction(
        facility_id=hospital.facility_id,
        facility_name=hospital.facility_name,
        state=hospital.state,
        input_features={
            k: v for k, v in data.items()
            if k not in ("facility_id", "facility_name", "state") and v is not None
        },
        risk_score=risk_score,
        risk_level=risk_level,
        model_version=model_manager.model_version,
    )

    return PredictionResponse(
        facility_id=hospital.facility_id,
        facility_name=hospital.facility_name,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        model_version=model_manager.model_version,
    )


# ── Batch prediction ──────────────────────────────────────────────────────────

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Predict readmission risk for multiple hospitals",
)
def predict_batch(
    hospitals: Annotated[
        list[HospitalInput],
        Body(description="List of hospital inputs (max 500)"),
    ],
) -> BatchPredictionResponse:
    """Score a list of hospitals in a single request.

    Returns individual predictions plus batch-level summary statistics.
    Maximum batch size is 500 hospitals.

    Raises 400 if the batch is empty or exceeds 500 items.
    Raises 503 if the model is not loaded.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check /health for details.",
        )
    if not hospitals:
        raise HTTPException(status_code=400, detail="Batch must not be empty.")
    if len(hospitals) > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(hospitals)} exceeds maximum of 500.",
        )

    try:
        records = [h.model_dump() for h in hospitals]
        results = model_manager.predict_batch(records)
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {exc}",
        ) from exc

    predictions: list[PredictionResponse] = []
    for hospital, (risk_score, risk_level) in zip(hospitals, results):
        data = hospital.model_dump()
        log_prediction(
            facility_id=hospital.facility_id,
            facility_name=hospital.facility_name,
            state=hospital.state,
            input_features={
                k: v for k, v in data.items()
                if k not in ("facility_id", "facility_name", "state") and v is not None
            },
            risk_score=risk_score,
            risk_level=risk_level,
            model_version=model_manager.model_version,
        )
        predictions.append(
            PredictionResponse(
                facility_id=hospital.facility_id,
                facility_name=hospital.facility_name,
                risk_score=round(risk_score, 4),
                risk_level=risk_level,
                model_version=model_manager.model_version,
            )
        )

    scores = [p.risk_score for p in predictions]
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=sum(1 for p in predictions if p.risk_level == "high"),
        mean_risk_score=round(sum(scores) / len(scores), 4),
        model_version=model_manager.model_version,
    )


# ── Prediction history ────────────────────────────────────────────────────────

@app.get(
    "/predictions/history",
    tags=["Monitoring"],
    summary="Last N logged predictions",
)
def predictions_history(
    limit: Annotated[int, Query(ge=1, le=1000, description="Number of records to return")] = 100,
) -> JSONResponse:
    """Return the most recent prediction log entries from the monitoring DB.

    Also includes aggregate statistics (total count, mean score, risk-level
    distribution) in the ``stats`` key.

    Parameters
    ----------
    limit:
        How many records to return (1–1000, default 100).
    """
    records = get_recent_predictions(limit=limit)
    stats = get_prediction_stats()
    return JSONResponse(content={"stats": stats, "predictions": records})
