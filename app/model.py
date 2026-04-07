"""app/model.py

Model loading and inference for the readmission prediction API.

Load order
----------
1. Attempt to load the ``Production`` model from the MLflow Model Registry.
2. Fall back to ``models/model.pkl`` if the registry is unavailable.

The module exposes a single ``ModelManager`` instance (``model_manager``)
that is initialised once at application startup via FastAPI's lifespan hook.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature columns in the exact order the model was trained on.
# Populated at load time from the pickle payload or MLflow model signature.
_MEASURES = ["ami", "cabg", "copd", "hf", "hip_knee", "pn"]
_MEASURE_FEATURES = [
    f"{m}_{s}"
    for m in _MEASURES
    for s in ("excess_ratio", "predicted_rate", "expected_rate", "discharges")
]
_AGGREGATE_FEATURES = [
    "mean_excess_ratio",
    "max_excess_ratio",
    "min_excess_ratio",
    "n_measures_over_threshold",
    "n_valid_measures",
    "mean_predicted_rate",
    "total_discharges",
]
_STATE_FEATURES = ["state_encoded"]

DEFAULT_FEATURE_NAMES: list[str] = (
    _MEASURE_FEATURES + _AGGREGATE_FEATURES + _STATE_FEATURES
)

# Risk level thresholds (tuned to match training target definition)
_LOW_THRESHOLD = 0.4
_HIGH_THRESHOLD = 0.65

# Simple label-encoding lookup built from CMS state list (alphabetical = 0-based)
_STATES = [
    "AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","GU","HI","IA",
    "ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT",
    "NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","PR","RI",
    "SC","SD","TN","TX","UT","VA","VI","VT","WA","WI","WV","WY",
]
_STATE_ENCODER: dict[str, int] = {s: i for i, s in enumerate(_STATES)}


# ── Feature assembly ──────────────────────────────────────────────────────────

def _assemble_features(data: dict[str, Any]) -> pd.DataFrame:
    """Convert a raw input dict into the feature vector expected by the model.

    Computes aggregate features (mean/max excess ratio, etc.) from the
    per-measure values so callers only need to provide raw CMS metrics.

    Parameters
    ----------
    data:
        Dict of field name → value, typically ``HospitalInput.model_dump()``.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns matching ``DEFAULT_FEATURE_NAMES``.
    """
    row: dict[str, float | None] = {}

    # Per-measure features
    for col in _MEASURE_FEATURES:
        row[col] = data.get(col)

    # Aggregate features derived from excess ratios
    excess_vals = [
        data.get(f"{m}_excess_ratio")
        for m in _MEASURES
        if data.get(f"{m}_excess_ratio") is not None
    ]
    if excess_vals:
        arr = np.array(excess_vals, dtype=float)
        row["mean_excess_ratio"] = float(np.mean(arr))
        row["max_excess_ratio"] = float(np.max(arr))
        row["min_excess_ratio"] = float(np.min(arr))
        row["n_measures_over_threshold"] = float(np.sum(arr > 1.0))
        row["n_valid_measures"] = float(len(arr))
    else:
        row["mean_excess_ratio"] = None
        row["max_excess_ratio"] = None
        row["min_excess_ratio"] = None
        row["n_measures_over_threshold"] = None
        row["n_valid_measures"] = None

    predicted_vals = [
        data.get(f"{m}_predicted_rate")
        for m in _MEASURES
        if data.get(f"{m}_predicted_rate") is not None
    ]
    row["mean_predicted_rate"] = (
        float(np.mean(predicted_vals)) if predicted_vals else None
    )

    discharge_vals = [
        data.get(f"{m}_discharges")
        for m in _MEASURES
        if data.get(f"{m}_discharges") is not None
    ]
    row["total_discharges"] = (
        float(np.sum(discharge_vals)) if discharge_vals else None
    )

    # State encoding
    state = data.get("state")
    row["state_encoded"] = float(_STATE_ENCODER.get(str(state).upper(), 0))

    return pd.DataFrame([row])


def _risk_level(score: float) -> str:
    """Map a probability score to a human-readable risk tier.

    Parameters
    ----------
    score:
        Model output probability in [0, 1].

    Returns
    -------
    str
        ``"low"``, ``"medium"``, or ``"high"``.
    """
    if score >= _HIGH_THRESHOLD:
        return "high"
    if score >= _LOW_THRESHOLD:
        return "medium"
    return "low"


# ── Model manager ─────────────────────────────────────────────────────────────

@dataclass
class ModelManager:
    """Holds the loaded model and exposes prediction methods.

    Attributes
    ----------
    model:
        Fitted model object (XGBClassifier or mlflow pyfunc wrapper).
    feature_names:
        Ordered list of feature column names used at training time.
    model_version:
        Version string for display in API responses.
    model_stage:
        Registry stage (``"Production"``, ``"local"``, etc.).
    training_metrics:
        Metrics from the best MLflow run (empty dict if unavailable).
    """

    model: Any = field(default=None)
    feature_names: list[str] = field(default_factory=lambda: DEFAULT_FEATURE_NAMES)
    model_version: str = "unknown"
    model_stage: str = "unknown"
    training_metrics: dict[str, float] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)

    @property
    def is_loaded(self) -> bool:
        """Return True if a model has been successfully loaded."""
        return self.model is not None

    def load(self, model_name: str, model_stage: str, local_path: Path) -> None:
        """Load model from MLflow registry, falling back to local pickle.

        Load order:
        1. MLflow model registry (requires a running tracking server).
        2. Download from ``settings.model_download_url`` if set (Railway deploy).
        3. Local pickle at ``local_path``.

        Parameters
        ----------
        model_name:
            Registered model name in MLflow.
        model_stage:
            Registry stage to load (``"Production"`` etc.).
        local_path:
            Absolute path to the fallback ``model.pkl`` file.
        """
        loaded = self._try_mlflow(model_name, model_stage)
        if not loaded:
            self._maybe_download(local_path)
            self._load_local(local_path)

    def _maybe_download(self, local_path: Path) -> None:
        """Download model.pkl from MODEL_DOWNLOAD_URL if set and file absent.

        Parameters
        ----------
        local_path:
            Destination path for the downloaded pickle.
        """
        from app.config import settings
        url = settings.model_download_url
        if not url or local_path.exists():
            return
        import urllib.request
        logger.info("Downloading model from %s …", url)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, str(local_path))
            logger.info("Model downloaded to %s", local_path)
        except Exception as exc:
            logger.warning("Model download failed: %s", exc)

    def _try_mlflow(self, model_name: str, model_stage: str) -> bool:
        """Attempt to load from the MLflow model registry.

        Parameters
        ----------
        model_name:
            Registered model name.
        model_stage:
            Stage to load.

        Returns
        -------
        bool
            True on success, False if the registry is unreachable.
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            from app.config import settings
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            client = MlflowClient()

            # Get latest version in the requested stage
            versions = client.get_latest_versions(model_name, stages=[model_stage])
            if not versions:
                logger.warning(
                    "No '%s' versions found in stage '%s'.",
                    model_name, model_stage,
                )
                return False

            mv = versions[0]
            model_uri = f"models:/{model_name}/{model_stage}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.model_version = mv.version
            self.model_stage = model_stage

            # Fetch training metrics from the source run
            try:
                run = client.get_run(mv.run_id)
                fi_metrics = {}
                other_metrics = {}
                for k, v in run.data.metrics.items():
                    if k.startswith("fi_"):
                        fi_metrics[k] = v
                    else:
                        other_metrics[k] = v
                self.training_metrics = other_metrics
                # fi_ keys are like "fi_01_mean_excess_ratio" — strip the rank prefix
                if fi_metrics:
                    self.feature_importances = {
                        "_".join(k.split("_")[2:]): v
                        for k, v in fi_metrics.items()
                    }
            except Exception:
                logger.debug("Could not fetch training metrics from run.", exc_info=True)

            # Feature names from model signature if available
            sig = self.model.metadata.signature
            if sig and sig.inputs:
                self.feature_names = [col.name for col in sig.inputs]

            logger.info(
                "Loaded model '%s' version %s from MLflow (%s).",
                model_name, self.model_version, model_stage,
            )
            return True

        except Exception as exc:
            logger.warning("MLflow registry unavailable (%s) — trying local fallback.", exc)
            return False

    def _load_local(self, local_path: Path) -> None:
        """Load model from the local pickle fallback.

        Parameters
        ----------
        local_path:
            Absolute path to ``models/model.pkl``.

        Raises
        ------
        RuntimeError
            If neither MLflow nor the local fallback is available.
        """
        if not local_path.exists():
            raise RuntimeError(
                f"No model found. MLflow registry unavailable and local fallback "
                f"'{local_path}' does not exist. Run `python pipeline/train.py` first."
            )

        with open(local_path, "rb") as f:
            payload = pickle.load(f)

        self.model = payload["model"]
        self.feature_names = payload.get("feature_names", DEFAULT_FEATURE_NAMES)
        self.model_version = payload.get("model_version", "local")
        self.model_stage = "local"
        # Extract feature importances from native XGBClassifier
        try:
            importances = self.model.feature_importances_
            self.feature_importances = dict(zip(self.feature_names, importances.tolist()))
        except Exception:
            logger.debug("Could not extract feature importances from local model.", exc_info=True)
        logger.info("Loaded model from local fallback: %s", local_path)

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_single(self, data: dict[str, Any]) -> tuple[float, str]:
        """Predict readmission risk for a single hospital.

        Parameters
        ----------
        data:
            ``HospitalInput.model_dump()`` dict.

        Returns
        -------
        tuple[float, str]
            ``(risk_score, risk_level)`` where risk_score ∈ [0, 1].
        """
        X = _assemble_features(data)
        X = X.reindex(columns=self.feature_names)
        score = self._predict_proba(X)
        return score, _risk_level(score)

    def predict_batch(
        self, records: list[dict[str, Any]]
    ) -> list[tuple[float, str]]:
        """Predict readmission risk for a list of hospitals.

        Parameters
        ----------
        records:
            List of ``HospitalInput.model_dump()`` dicts.

        Returns
        -------
        list[tuple[float, str]]
            List of ``(risk_score, risk_level)`` pairs, one per input.
        """
        frames = [_assemble_features(r) for r in records]
        X = pd.concat(frames, ignore_index=True).reindex(columns=self.feature_names)
        probas = self._predict_proba_batch(X)
        return [(float(p), _risk_level(float(p))) for p in probas]

    def _predict_proba(self, X: pd.DataFrame) -> float:
        """Return the positive-class probability for a single row.

        Handles both native XGBClassifier and mlflow pyfunc wrappers.

        Parameters
        ----------
        X:
            Single-row feature DataFrame aligned to training columns.

        Returns
        -------
        float
            Probability in [0, 1].
        """
        return float(self._predict_proba_batch(X)[0])

    def _predict_proba_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities for all rows in X.

        Parameters
        ----------
        X:
            Feature DataFrame aligned to training columns.

        Returns
        -------
        np.ndarray
            1-D array of probabilities in [0, 1].
        """
        try:
            # Native XGBClassifier (loaded from local pickle)
            return self.model.predict_proba(X)[:, 1]
        except AttributeError:
            # mlflow pyfunc wrapper — returns a DataFrame
            result = self.model.predict(X)
            if hasattr(result, "values"):
                result = result.values
            arr = np.array(result, dtype=float).flatten()
            # pyfunc may return class 1 probability or raw scores — normalise
            if arr.max() > 1.0 or arr.min() < 0.0:
                # Assume logit output — apply sigmoid
                arr = 1.0 / (1.0 + np.exp(-arr))
            return arr


# Module-level singleton — initialised via FastAPI lifespan
model_manager = ModelManager()
