"""tests/test_api.py

Integration tests for the FastAPI application.

Uses FastAPI's TestClient (backed by httpx) so no live server is needed.
The model_manager is patched with a lightweight fake model so tests run
without a trained model on disk or a live MLflow server.

Coverage
--------
- GET  /health
- GET  /model/info
- POST /predict         (happy path, validation errors, model-not-loaded)
- POST /predict/batch   (happy path, empty batch, oversized batch)
- GET  /predictions/history
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Fake model ────────────────────────────────────────────────────────────────

class _FakeXGB:
    """Minimal XGBClassifier stand-in that always returns 0.75."""

    feature_names_in_ = None

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.25), np.full(n, 0.75)])


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_model(tmp_path: Path):
    """Inject a fake model into model_manager before every test.

    Writes a minimal pickle to tmp_path and patches MODELS_DIR so the
    manager finds it as the local fallback.  No MLflow server required.
    """
    from pipeline.train import REGISTERED_MODEL_NAME

    feature_names = [
        "ami_excess_ratio", "ami_predicted_rate", "ami_expected_rate",
        "ami_discharges", "cabg_excess_ratio", "cabg_predicted_rate",
        "cabg_expected_rate", "cabg_discharges", "copd_excess_ratio",
        "copd_predicted_rate", "copd_expected_rate", "copd_discharges",
        "hf_excess_ratio", "hf_predicted_rate", "hf_expected_rate",
        "hf_discharges", "hip_knee_excess_ratio", "hip_knee_predicted_rate",
        "hip_knee_expected_rate", "hip_knee_discharges", "pn_excess_ratio",
        "pn_predicted_rate", "pn_expected_rate", "pn_discharges",
        "mean_excess_ratio", "max_excess_ratio", "min_excess_ratio",
        "n_measures_over_threshold", "n_valid_measures",
        "mean_predicted_rate", "total_discharges", "state_encoded",
    ]
    payload = {
        "model": _FakeXGB(),
        "feature_names": feature_names,
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": "test-1",
    }
    pkl = tmp_path / "models" / "model.pkl"
    pkl.parent.mkdir()
    with open(pkl, "wb") as f:
        pickle.dump(payload, f)

    with (
        patch("app.model.model_manager.model_version", "test-1"),
        patch("pipeline.train.MODELS_DIR", tmp_path / "models"),
        patch("app.config.settings.model_local_path", str(pkl)),
        patch("app.config.settings.model_local_abs_path", pkl),
        patch("app.config.settings.log_predictions", False),
    ):
        # Force fresh load from the tmp pickle
        from app.model import model_manager, ModelManager
        mgr = ModelManager()
        mgr._load_local(pkl)

        with patch("app.main.model_manager", mgr), \
             patch("app.model.model_manager", mgr):
            yield mgr


@pytest.fixture
def client(_patch_model) -> TestClient:
    """TestClient with the patched model_manager in scope."""
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    """Minimal valid HospitalInput payload."""
    return {
        "facility_id": "010001",
        "facility_name": "Test Hospital",
        "state": "AL",
        "ami_excess_ratio": 1.05,
        "ami_predicted_rate": 13.5,
        "ami_expected_rate": 12.8,
        "ami_discharges": 273,
        "hf_excess_ratio": 0.97,
        "hf_predicted_rate": 21.3,
        "hf_expected_rate": 22.0,
        "hf_discharges": 412,
    }


# ── GET /health ───────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client: TestClient) -> None:
        """Health endpoint returns HTTP 200."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_ok_when_model_loaded(self, client: TestClient) -> None:
        """Status is 'ok' when model is successfully loaded."""
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_model_version_present(self, client: TestClient) -> None:
        """Response includes a non-empty model_version string."""
        body = client.get("/health").json()
        assert body["model_version"]

    def test_degraded_when_no_model(self, client: TestClient) -> None:
        """Status is 'degraded' when no model is loaded."""
        from app.model import ModelManager
        empty_mgr = ModelManager()  # model=None

        with patch("app.main.model_manager", empty_mgr):
            resp = TestClient(__import__("app.main", fromlist=["app"]).app,
                              raise_server_exceptions=False).get("/health")
        assert resp.json()["status"] == "degraded"
        assert resp.json()["model_loaded"] is False


# ── GET /model/info ───────────────────────────────────────────────────────────

class TestModelInfo:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/model/info")
        assert resp.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        """Response contains all required ModelInfoResponse fields."""
        body = client.get("/model/info").json()
        assert "model_name" in body
        assert "model_version" in body
        assert "model_stage" in body
        assert "training_metrics" in body
        assert isinstance(body["feature_names"], list)
        assert body["feature_count"] == len(body["feature_names"])

    def test_feature_count_positive(self, client: TestClient) -> None:
        body = client.get("/model/info").json()
        assert body["feature_count"] > 0

    def test_503_when_no_model(self) -> None:
        """Returns 503 when model is not loaded."""
        from app.model import ModelManager
        from app.main import app
        empty_mgr = ModelManager()
        with patch("app.main.model_manager", empty_mgr):
            resp = TestClient(app, raise_server_exceptions=False).get("/model/info")
        assert resp.status_code == 503


# ── POST /predict ─────────────────────────────────────────────────────────────

class TestPredict:
    def test_happy_path_200(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Valid input returns HTTP 200."""
        resp = client.post("/predict", json=valid_payload)
        assert resp.status_code == 200

    def test_response_schema(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Response contains risk_score, risk_level, model_version."""
        body = client.post("/predict", json=valid_payload).json()
        assert "risk_score" in body
        assert "risk_level" in body
        assert "model_version" in body

    def test_risk_score_in_range(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """risk_score is a float in [0, 1]."""
        body = client.post("/predict", json=valid_payload).json()
        assert 0.0 <= body["risk_score"] <= 1.0

    def test_risk_level_valid_values(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """risk_level is one of low / medium / high."""
        body = client.post("/predict", json=valid_payload).json()
        assert body["risk_level"] in ("low", "medium", "high")

    def test_facility_id_echoed(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """facility_id from input is echoed in the response."""
        body = client.post("/predict", json=valid_payload).json()
        assert body["facility_id"] == valid_payload["facility_id"]

    def test_fake_model_returns_high_risk(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Fake model always returns 0.75 → risk_level == 'high'."""
        body = client.post("/predict", json=valid_payload).json()
        assert body["risk_score"] == pytest.approx(0.75, abs=0.01)
        assert body["risk_level"] == "high"

    def test_422_no_excess_ratios(self, client: TestClient) -> None:
        """Input with no excess ratio fields at all returns 422."""
        payload = {
            "facility_id": "X",
            "ami_predicted_rate": 13.0,  # predicted rate but NO excess ratio
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_422_wrong_type_for_numeric(self, client: TestClient) -> None:
        """Non-numeric value for a float field returns 422."""
        payload = {"ami_excess_ratio": "not-a-number"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_422_ratio_out_of_range(self, client: TestClient) -> None:
        """excess_ratio > 5.0 (schema upper bound) returns 422."""
        payload = {"ami_excess_ratio": 99.9}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_422_empty_body(self, client: TestClient) -> None:
        """Empty JSON body (no measures at all) returns 422."""
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_503_when_no_model(self, valid_payload: dict) -> None:
        """Returns 503 when model is not loaded."""
        from app.model import ModelManager
        from app.main import app
        empty_mgr = ModelManager()
        with patch("app.main.model_manager", empty_mgr):
            resp = TestClient(app, raise_server_exceptions=False).post(
                "/predict", json=valid_payload
            )
        assert resp.status_code == 503

    def test_optional_fields_null(self, client: TestClient) -> None:
        """Prediction succeeds when optional identifier fields are null."""
        payload = {"ami_excess_ratio": 1.1}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["facility_id"] is None


# ── POST /predict/batch ───────────────────────────────────────────────────────

class TestPredictBatch:
    def test_happy_path_200(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Two-item batch returns HTTP 200."""
        resp = client.post("/predict/batch", json=[valid_payload, valid_payload])
        assert resp.status_code == 200

    def test_response_total_matches_input(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """'total' field equals the number of hospitals submitted."""
        batch = [valid_payload] * 3
        body = client.post("/predict/batch", json=batch).json()
        assert body["total"] == 3
        assert len(body["predictions"]) == 3

    def test_mean_risk_score_in_range(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """mean_risk_score is in [0, 1]."""
        body = client.post("/predict/batch", json=[valid_payload, valid_payload]).json()
        assert 0.0 <= body["mean_risk_score"] <= 1.0

    def test_high_risk_count_correct(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """high_risk_count matches actual high-risk predictions (all 0.75 → high)."""
        body = client.post("/predict/batch", json=[valid_payload] * 4).json()
        assert body["high_risk_count"] == 4

    def test_400_empty_batch(self, client: TestClient) -> None:
        """Empty list returns HTTP 400."""
        resp = client.post("/predict/batch", json=[])
        assert resp.status_code == 400

    def test_400_oversized_batch(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Batch larger than 500 returns HTTP 400."""
        resp = client.post("/predict/batch", json=[valid_payload] * 501)
        assert resp.status_code == 400

    def test_422_invalid_item_in_batch(self, client: TestClient) -> None:
        """A single invalid item in the batch triggers 422 for the whole request."""
        bad = {"ami_excess_ratio": "oops"}
        resp = client.post("/predict/batch", json=[bad])
        assert resp.status_code == 422

    def test_model_version_consistent(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """All predictions in the batch share the same model_version."""
        body = client.post("/predict/batch", json=[valid_payload] * 2).json()
        versions = {p["model_version"] for p in body["predictions"]}
        assert len(versions) == 1


# ── GET /predictions/history ──────────────────────────────────────────────────

class TestPredictionsHistory:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/predictions/history")
        assert resp.status_code == 200

    def test_response_has_stats_and_predictions(
        self, client: TestClient
    ) -> None:
        """Response body contains 'stats' and 'predictions' keys."""
        body = client.get("/predictions/history").json()
        assert "stats" in body
        assert "predictions" in body

    def test_limit_param_accepted(self, client: TestClient) -> None:
        """?limit param is accepted without error."""
        resp = client.get("/predictions/history?limit=10")
        assert resp.status_code == 200

    def test_limit_too_large_returns_422(self, client: TestClient) -> None:
        """limit > 1000 is rejected with 422."""
        resp = client.get("/predictions/history?limit=9999")
        assert resp.status_code == 422

    def test_limit_zero_returns_422(self, client: TestClient) -> None:
        """limit=0 is rejected with 422 (minimum is 1)."""
        resp = client.get("/predictions/history?limit=0")
        assert resp.status_code == 422

    def test_stats_keys_present(self, client: TestClient) -> None:
        """Stats dict contains expected aggregate keys."""
        stats = client.get("/predictions/history").json()["stats"]
        assert "total" in stats
        assert "mean_risk_score" in stats
        assert "high_count" in stats
