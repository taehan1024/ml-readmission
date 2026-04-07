"""app/schemas.py

Pydantic request/response models for the readmission prediction API.

Input
-----
``HospitalInput`` accepts per-measure CMS readmission metrics (all optional
because not every hospital reports every measure). Aggregate features are
computed server-side in ``app/model.py`` before inference.

Output
------
``PredictionResponse`` returns a risk score, a human-readable risk level,
and the model version that produced the prediction.
``BatchPredictionResponse`` wraps a list of predictions with summary stats.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ── Input ──────────────────────────────────────────────────────────────────────

class HospitalInput(BaseModel):
    """Per-hospital readmission metrics used for risk prediction.

    All measure-specific fields are optional because CMS suppresses data
    for low-volume hospitals. At least one measure must be provided.

    Attributes
    ----------
    facility_id, facility_name, state:
        Optional identifiers — not used by the model but included in
        monitoring logs and batch response tables.
    {measure}_excess_ratio:
        Excess readmission ratio for that measure (typical range 0.5–1.5).
        Values > 1.0 indicate worse-than-expected readmissions.
    {measure}_predicted_rate:
        Risk-adjusted predicted readmission rate (%).
    {measure}_expected_rate:
        Expected readmission rate for a hospital with average performance (%).
    {measure}_discharges:
        Number of eligible discharges for that measure.
    """

    # Identifiers (passed through to response, not fed to model)
    facility_id: Optional[str] = Field(None, description="CMS provider number")
    facility_name: Optional[str] = Field(None, description="Hospital name")
    state: Optional[str] = Field(None, description="Two-letter US state code", max_length=2)

    # AMI — Acute Myocardial Infarction
    ami_excess_ratio: Optional[float] = Field(None, ge=0.0, le=5.0)
    ami_predicted_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    ami_expected_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    ami_discharges: Optional[float] = Field(None, ge=0.0)

    # CABG — Coronary Artery Bypass Graft
    cabg_excess_ratio: Optional[float] = Field(None, ge=0.0, le=5.0)
    cabg_predicted_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    cabg_expected_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    cabg_discharges: Optional[float] = Field(None, ge=0.0)

    # COPD — Chronic Obstructive Pulmonary Disease
    copd_excess_ratio: Optional[float] = Field(None, ge=0.0, le=5.0)
    copd_predicted_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    copd_expected_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    copd_discharges: Optional[float] = Field(None, ge=0.0)

    # HF — Heart Failure
    hf_excess_ratio: Optional[float] = Field(None, ge=0.0, le=5.0)
    hf_predicted_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    hf_expected_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    hf_discharges: Optional[float] = Field(None, ge=0.0)

    # Hip/Knee — Hip and Knee Arthroplasty
    hip_knee_excess_ratio: Optional[float] = Field(None, ge=0.0, le=5.0)
    hip_knee_predicted_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    hip_knee_expected_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    hip_knee_discharges: Optional[float] = Field(None, ge=0.0)

    # PN — Pneumonia
    pn_excess_ratio: Optional[float] = Field(None, ge=0.0, le=5.0)
    pn_predicted_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    pn_expected_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    pn_discharges: Optional[float] = Field(None, ge=0.0)

    @model_validator(mode="after")
    def at_least_one_measure(self) -> "HospitalInput":
        """Reject inputs where no excess ratio is provided at all."""
        excess_cols = [
            self.ami_excess_ratio,
            self.cabg_excess_ratio,
            self.copd_excess_ratio,
            self.hf_excess_ratio,
            self.hip_knee_excess_ratio,
            self.pn_excess_ratio,
        ]
        if all(v is None for v in excess_cols):
            raise ValueError(
                "At least one excess readmission ratio must be provided "
                "(ami_excess_ratio, cabg_excess_ratio, copd_excess_ratio, "
                "hf_excess_ratio, hip_knee_excess_ratio, or pn_excess_ratio)."
            )
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "facility_id": "010001",
            "facility_name": "Example Medical Center",
            "state": "AL",
            "ami_excess_ratio": 1.05,
            "ami_predicted_rate": 13.5,
            "ami_expected_rate": 12.8,
            "ami_discharges": 273,
            "hf_excess_ratio": 0.97,
            "hf_predicted_rate": 21.3,
            "hf_expected_rate": 22.0,
            "hf_discharges": 412,
            "pn_excess_ratio": 1.12,
            "pn_predicted_rate": 15.2,
            "pn_expected_rate": 13.6,
            "pn_discharges": 189,
        }
    }}


# ── Output ─────────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Single-hospital prediction result.

    Attributes
    ----------
    facility_id:
        Echoed from input (``None`` if not provided).
    facility_name:
        Echoed from input (``None`` if not provided).
    risk_score:
        Model probability estimate in [0, 1]. Higher = more likely to be
        penalised for excess readmissions.
    risk_level:
        Human-readable tier: ``"low"`` (< 0.4), ``"medium"`` (0.4–0.65),
        or ``"high"`` (> 0.65).
    model_version:
        Version string of the model that produced this prediction.
    """

    facility_id: Optional[str] = None
    facility_name: Optional[str] = None
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Readmission risk probability")
    risk_level: str = Field(..., description="low | medium | high")
    model_version: str = Field(..., description="Model version identifier")

    model_config = {"json_schema_extra": {
        "example": {
            "facility_id": "010001",
            "facility_name": "Example Medical Center",
            "risk_score": 0.72,
            "risk_level": "high",
            "model_version": "1",
        }
    }}


class BatchPredictionResponse(BaseModel):
    """Batch prediction result with per-hospital predictions and summary stats.

    Attributes
    ----------
    predictions:
        Ordered list of ``PredictionResponse`` objects, one per input hospital.
    total:
        Total number of hospitals in the batch.
    high_risk_count:
        Number of hospitals with ``risk_level == "high"``.
    mean_risk_score:
        Mean risk score across all hospitals in the batch.
    model_version:
        Model version used for all predictions in this batch.
    """

    predictions: list[PredictionResponse]
    total: int
    high_risk_count: int
    mean_risk_score: float = Field(..., ge=0.0, le=1.0)
    model_version: str


# ── Health / Info ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Service health check response."""

    status: str = Field(..., description="ok | degraded")
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    """Model metadata and training metrics surfaced from MLflow."""

    model_name: str
    model_version: str
    model_stage: str
    training_metrics: dict[str, float]
    feature_importances: dict[str, float] = {}
    feature_count: int
    feature_names: list[str]
