"""tests/test_model.py

Tests for model training utilities and inference behaviour.
Uses a small synthetic dataset — no MLflow server or file I/O required
for the core logic tests.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from pipeline.train import (
    REGISTERED_MODEL_NAME,
    TARGET_COL,
    compute_metrics,
    load_features,
    save_local,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_feature_df(n: int = 100) -> pd.DataFrame:
    """Build a minimal synthetic feature DataFrame matching the pipeline schema.

    Parameters
    ----------
    n:
        Number of rows (hospitals) to generate.

    Returns
    -------
    pd.DataFrame
        Synthetic feature matrix with target column.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "facility_id": [f"H{i:04d}" for i in range(n)],
            "facility_name": [f"Hospital {i}" for i in range(n)],
            "state": rng.choice(["AL", "CA", "TX", "NY"], size=n),
            "ami_excess_ratio": rng.uniform(0.7, 1.3, n),
            "hf_excess_ratio": rng.uniform(0.7, 1.3, n),
            "pn_excess_ratio": rng.uniform(0.7, 1.3, n),
            "mean_excess_ratio": rng.uniform(0.8, 1.2, n),
            "max_excess_ratio": rng.uniform(0.9, 1.3, n),
            "n_valid_measures": rng.integers(3, 7, n),
            "state_encoded": rng.integers(0, 4, n),
            TARGET_COL: rng.integers(0, 2, n),
        }
    )
    return df


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """100-row synthetic feature DataFrame."""
    return _make_feature_df(100)


@pytest.fixture
def fitted_model(feature_df: pd.DataFrame) -> XGBClassifier:
    """A small XGBClassifier fitted on synthetic data."""
    X = feature_df.drop(
        columns=["facility_id", "facility_name", "state", TARGET_COL]
    )
    y = feature_df[TARGET_COL]
    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)
    return model


# ── load_features ─────────────────────────────────────────────────────────────

class TestLoadFeatures:
    def test_raises_if_file_missing(self) -> None:
        """FileNotFoundError raised when feature parquet does not exist."""
        with pytest.raises(FileNotFoundError, match="Feature matrix not found"):
            load_features(Path("/nonexistent/features.parquet"))

    def test_returns_x_and_y(self, tmp_path: Path, feature_df: pd.DataFrame) -> None:
        """Returns (X, y) tuple; X excludes identifier and target columns."""
        parquet_path = tmp_path / "features.parquet"
        feature_df.to_parquet(parquet_path, index=False)
        X, y = load_features(parquet_path)
        assert TARGET_COL not in X.columns
        assert "facility_id" not in X.columns
        assert len(X) == len(y)

    def test_y_is_binary(self, tmp_path: Path, feature_df: pd.DataFrame) -> None:
        """Target series contains only 0 and 1."""
        parquet_path = tmp_path / "features.parquet"
        feature_df.to_parquet(parquet_path, index=False)
        _, y = load_features(parquet_path)
        assert set(y.unique()).issubset({0, 1})


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_all_keys(
        self, fitted_model: XGBClassifier, feature_df: pd.DataFrame
    ) -> None:
        """compute_metrics returns all expected metric keys."""
        X = feature_df.drop(
            columns=["facility_id", "facility_name", "state", TARGET_COL]
        )
        y = feature_df[TARGET_COL]
        metrics = compute_metrics(fitted_model, X, y)
        assert set(metrics.keys()) == {"auc", "f1", "precision", "recall", "brier_score"}

    def test_auc_in_range(
        self, fitted_model: XGBClassifier, feature_df: pd.DataFrame
    ) -> None:
        """AUC is between 0.0 and 1.0."""
        X = feature_df.drop(
            columns=["facility_id", "facility_name", "state", TARGET_COL]
        )
        y = feature_df[TARGET_COL]
        metrics = compute_metrics(fitted_model, X, y)
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_brier_score_in_range(
        self, fitted_model: XGBClassifier, feature_df: pd.DataFrame
    ) -> None:
        """Brier score is between 0.0 and 1.0."""
        X = feature_df.drop(
            columns=["facility_id", "facility_name", "state", TARGET_COL]
        )
        y = feature_df[TARGET_COL]
        metrics = compute_metrics(fitted_model, X, y)
        assert 0.0 <= metrics["brier_score"] <= 1.0

    def test_metrics_are_floats(
        self, fitted_model: XGBClassifier, feature_df: pd.DataFrame
    ) -> None:
        """All metric values are Python floats."""
        X = feature_df.drop(
            columns=["facility_id", "facility_name", "state", TARGET_COL]
        )
        y = feature_df[TARGET_COL]
        metrics = compute_metrics(fitted_model, X, y)
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"


# ── save_local ────────────────────────────────────────────────────────────────

class TestSaveLocal:
    def test_file_created(
        self, fitted_model: XGBClassifier, tmp_path: Path, monkeypatch
    ) -> None:
        """save_local creates models/model.pkl."""
        monkeypatch.setattr(
            "pipeline.train.MODELS_DIR", tmp_path / "models"
        )
        feature_names = ["ami_excess_ratio", "hf_excess_ratio", "state_encoded"]
        out = save_local(fitted_model, feature_names)
        assert out.exists()

    def test_pickle_contains_expected_keys(
        self, fitted_model: XGBClassifier, tmp_path: Path, monkeypatch
    ) -> None:
        """Saved pickle contains 'model', 'feature_names', 'model_name'."""
        monkeypatch.setattr(
            "pipeline.train.MODELS_DIR", tmp_path / "models"
        )
        feature_names = ["ami_excess_ratio", "hf_excess_ratio", "state_encoded"]
        out = save_local(fitted_model, feature_names)
        with open(out, "rb") as f:
            payload = pickle.load(f)
        assert "model" in payload
        assert "feature_names" in payload
        assert "model_name" in payload
        assert payload["model_name"] == REGISTERED_MODEL_NAME

    def test_loaded_model_predicts(
        self, fitted_model: XGBClassifier, tmp_path: Path, monkeypatch
    ) -> None:
        """Model loaded from pickle produces valid probability predictions."""
        monkeypatch.setattr(
            "pipeline.train.MODELS_DIR", tmp_path / "models"
        )
        feature_names = ["ami_excess_ratio", "hf_excess_ratio", "state_encoded"]
        out = save_local(fitted_model, feature_names)
        with open(out, "rb") as f:
            payload = pickle.load(f)

        sample = pd.DataFrame(
            [[0.95, 1.05, 2]],
            columns=feature_names,
        )
        proba = payload["model"].predict_proba(sample)[0, 1]
        assert 0.0 <= proba <= 1.0


# ── Inference behaviour ───────────────────────────────────────────────────────

class TestInferenceBehaviour:
    def test_predict_proba_range(
        self, fitted_model: XGBClassifier, feature_df: pd.DataFrame
    ) -> None:
        """predict_proba output is in [0, 1] for all rows."""
        X = feature_df.drop(
            columns=["facility_id", "facility_name", "state", TARGET_COL]
        )
        proba = fitted_model.predict_proba(X)[:, 1]
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()

    def test_predict_proba_batch_length(
        self, fitted_model: XGBClassifier, feature_df: pd.DataFrame
    ) -> None:
        """Batch prediction returns one probability per input row."""
        X = feature_df.drop(
            columns=["facility_id", "facility_name", "state", TARGET_COL]
        )
        proba = fitted_model.predict_proba(X)[:, 1]
        assert len(proba) == len(X)

    def test_single_row_predict(self, fitted_model: XGBClassifier) -> None:
        """Single-row prediction returns a float in [0, 1]."""
        X_single = pd.DataFrame(
            [[0.95, 1.05, 1.0, 1.1, 4, 2]],
            columns=[
                "ami_excess_ratio",
                "hf_excess_ratio",
                "pn_excess_ratio",
                "mean_excess_ratio",
                "n_valid_measures",
                "state_encoded",
            ],
        )
        # Use only the columns the model was trained on
        trained_cols = fitted_model.feature_names_in_
        X_aligned = X_single.reindex(columns=trained_cols, fill_value=0)
        proba = fitted_model.predict_proba(X_aligned)[0, 1]
        assert isinstance(float(proba), float)
        assert 0.0 <= float(proba) <= 1.0
