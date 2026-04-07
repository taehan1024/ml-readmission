"""dashboard/app.py

Streamlit dashboard for the Hospital Readmission Prediction System.

All data is fetched from the FastAPI backend — no model is imported here.
Configure the API URL in the sidebar or via Streamlit secrets
(key: ``FASTAPI_URL``).

Run locally
-----------
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from api_client import (
    get_health,
    get_history,
    get_model_info,
    predict_batch,
)
from components import (
    batch_results_table,
    batch_summary_metrics,
    feature_importance_chart,
    risk_distribution_chart,
    score_over_time_chart,
    state_risk_map,
    training_metrics_chart,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Readmission Risk Dashboard",
    page_icon="🏥",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏥 Readmission Risk")
    st.caption("Hospital Readmissions Reduction Program")

    default_url = st.secrets.get("FASTAPI_URL", "http://localhost:8000")
    api_url = st.text_input("API endpoint", value=default_url)

    # Connection status indicator
    try:
        health = get_health(api_url)
        status_colour = "🟢" if health["status"] == "ok" else "🟠"
        st.success(
            f"{status_colour} Connected  •  model v{health['model_version']}"
        )
    except Exception as exc:
        st.error(f"🔴 API unreachable\n\n`{exc}`")
        st.stop()

    st.divider()
    st.caption("Built with FastAPI + XGBoost + MLflow")
    st.divider()
    st.caption(
        "**Dataset:** CMS Hospital Readmissions Reduction Program (HRRP) — "
        "~3,100 US hospitals, 6 conditions (AMI, CABG, COPD, Heart Failure, "
        "Hip/Knee, Pneumonia), published annually.\n\n"
        "**Goal:** Predict whether a hospital's readmission rate exceeds CMS "
        "expectations (excess ratio > 1.0), indicating risk of financial penalty."
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_batch, tab_info, tab_monitor = st.tabs([
    "📋 Batch Prediction",
    "📊 Model Info",
    "📋 Prediction Log",
])


# ── Tab 1: Batch prediction ───────────────────────────────────────────────────

# Path to the processed feature matrix (used for random sampling)
_FEATURES_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.parquet"
# Bundled fallback CSV for Streamlit Cloud (committed to repo)
_BUNDLED_CSV = Path(__file__).resolve().parent / "data" / "hospitals.csv"


def _load_features() -> pd.DataFrame | None:
    """Load hospital feature data: parquet if available, else bundled CSV."""
    if _FEATURES_PATH.exists():
        return pd.read_parquet(_FEATURES_PATH)
    if _BUNDLED_CSV.exists():
        return pd.read_csv(_BUNDLED_CSV)
    return None

# Columns from features.parquet that map directly to HospitalInput fields
_HOSPITAL_INPUT_COLS = [
    "facility_id", "facility_name", "state",
    "ami_excess_ratio", "ami_predicted_rate", "ami_expected_rate", "ami_discharges",
    "cabg_excess_ratio", "cabg_predicted_rate", "cabg_expected_rate", "cabg_discharges",
    "copd_excess_ratio", "copd_predicted_rate", "copd_expected_rate", "copd_discharges",
    "hf_excess_ratio", "hf_predicted_rate", "hf_expected_rate", "hf_discharges",
    "hip_knee_excess_ratio", "hip_knee_predicted_rate", "hip_knee_expected_rate", "hip_knee_discharges",
    "pn_excess_ratio", "pn_predicted_rate", "pn_expected_rate", "pn_discharges",
]

with tab_batch:
    st.header("Batch Hospital Risk Assessment")

    # ── Random sample section ─────────────────────────────────────────────────
    st.subheader("🎲 Random Sample Prediction")
    st.caption("Pick random hospitals from the local feature dataset and score them instantly.")

    features_df = _load_features()
    if features_df is None:
        st.info(
            "Random sampling requires the hospital dataset. "
            "Use the CSV upload below to score your hospitals."
        )
    else:
        col_btn, col_n = st.columns([1, 3])
        n_sample = col_n.slider("Sample size", min_value=10, max_value=500, value=100, step=10)
        run_sample = col_btn.button("▶ Run Random Sample", type="primary", use_container_width=True)

        if run_sample:
            available_cols = [c for c in _HOSPITAL_INPUT_COLS if c in features_df.columns]
            sample_df = features_df[available_cols].sample(n=min(n_sample, len(features_df)), random_state=None)

            st.write(f"**Sample data** ({len(sample_df)} hospitals):")
            st.dataframe(sample_df.head(10), use_container_width=True)

            records = sample_df.astype(object).where(pd.notna(sample_df), None).to_dict(orient="records")
            with st.spinner(f"Scoring {len(records)} hospitals …"):
                try:
                    response = predict_batch(api_url, records)
                    st.divider()
                    batch_summary_metrics(response)
                    st.subheader("Predictions")
                    df_out = batch_results_table(response["predictions"])
                    col_dist, col_map = st.columns(2)
                    with col_dist:
                        risk_distribution_chart(response["predictions"])
                    with col_map:
                        state_risk_map(response["predictions"])
                    csv_bytes = df_out.to_csv(index=False).encode()
                    st.download_button(
                        "⬇ Download results CSV",
                        data=csv_bytes,
                        file_name="sample_risk_results.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    st.divider()
    st.caption(
        "Or upload your own CSV with columns matching the HospitalInput schema. "
        "Download the template below to get started."
    )

    # CSV template download
    template_cols = (
        ["facility_id", "facility_name", "state"]
        + [f"{m}_{s}" for m in ["ami","cabg","copd","hf","hip_knee","pn"]
           for s in ["excess_ratio","predicted_rate","expected_rate","discharges"]]
    )
    template_df = pd.DataFrame(columns=template_cols)
    st.download_button(
        "⬇ Download CSV template",
        data=template_df.to_csv(index=False),
        file_name="hospital_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload hospital CSV", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df_in)}** hospitals.")
        st.dataframe(df_in.head(5), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            records = df_in.astype(object).where(pd.notna(df_in), None).to_dict(orient="records")
            if len(records) > 500:
                st.error("Batch exceeds 500 hospitals. Split your file and re-upload.")
            else:
                with st.spinner(f"Scoring {len(records)} hospitals …"):
                    try:
                        response = predict_batch(api_url, records)
                        st.divider()
                        batch_summary_metrics(response)
                        st.subheader("Results")
                        df_out = batch_results_table(response["predictions"])
                        col_dist, col_map = st.columns(2)
                        with col_dist:
                            risk_distribution_chart(response["predictions"])
                        with col_map:
                            state_risk_map(response["predictions"])
                        csv_bytes = df_out.to_csv(index=False).encode()
                        st.download_button(
                            "⬇ Download results CSV",
                            data=csv_bytes,
                            file_name="readmission_risk_results.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.error(f"Batch prediction failed: {exc}")


# ── Feature descriptions ──────────────────────────────────────────────────────

_MEASURE_LABELS = {
    "ami": "AMI (Heart Attack)",
    "cabg": "CABG (Bypass Surgery)",
    "copd": "COPD",
    "hf": "Heart Failure",
    "hip_knee": "Hip / Knee Replacement",
    "pn": "Pneumonia",
}

_FEATURE_DESCRIPTIONS: dict[str, str] = {
    **{
        f"{m}_excess_ratio": (
            f"[{_MEASURE_LABELS[m]}] Ratio of observed to expected 30-day readmissions "
            "(>1.0 = worse than expected; hospital may face CMS penalty)"
        )
        for m in _MEASURE_LABELS
    },
    **{
        f"{m}_predicted_rate": (
            f"[{_MEASURE_LABELS[m]}] CMS-predicted 30-day readmission rate (%) "
            "for this condition based on hospital case-mix"
        )
        for m in _MEASURE_LABELS
    },
    **{
        f"{m}_expected_rate": (
            f"[{_MEASURE_LABELS[m]}] Expected readmission rate (%) given this hospital's "
            "patient population characteristics"
        )
        for m in _MEASURE_LABELS
    },
    **{
        f"{m}_discharges": (
            f"[{_MEASURE_LABELS[m]}] Number of patient discharges for this condition "
            "in the CMS measurement period"
        )
        for m in _MEASURE_LABELS
    },
    "mean_excess_ratio": "Average excess ratio across all conditions with sufficient discharge volume",
    "max_excess_ratio": "Worst (highest) excess ratio across all conditions — captures the most penalised condition",
    "min_excess_ratio": "Best (lowest) excess ratio across all conditions",
    "n_measures_over_threshold": "Count of conditions where excess ratio > 1.0 (hospital is above expected readmissions)",
    "n_valid_measures": "Number of conditions with enough discharges to be officially measured by CMS",
    "mean_predicted_rate": "Average CMS-predicted readmission rate across all measured conditions",
    "total_discharges": "Total discharges summed across all six conditions",
    "state_encoded": "Integer label encoding of the hospital's US state (alphabetical order, 0 = AK … 53 = WY)",
}


# ── Tab 2: Model info ─────────────────────────────────────────────────────────

with tab_info:
    st.header("Model Metadata & Performance")
    try:
        info = get_model_info(api_url)

        c1, c2, c3 = st.columns(3)
        c1.metric("Model Name", info["model_name"])
        c2.metric("Version", info["model_version"])
        c3.metric("Stage", info["model_stage"])

        st.subheader("Training Metrics")
        metrics = info.get("training_metrics", {})
        if metrics:
            m_cols = st.columns(len(metrics))
            for col, (k, v) in zip(m_cols, metrics.items()):
                col.metric(k.upper(), f"{v:.4f}")
            training_metrics_chart(metrics)
        else:
            st.info("No training metrics available (model loaded from local fallback).")

        st.subheader("Feature Importances")
        feature_importance_chart(info.get("feature_importances", {}))

        st.subheader(f"All Features ({info['feature_count']} total)")
        feat_df = pd.DataFrame({
            "Feature": info["feature_names"],
            "Description": [
                _FEATURE_DESCRIPTIONS.get(f, "") for f in info["feature_names"]
            ],
        })
        feat_df.index += 1
        st.dataframe(feat_df, use_container_width=True, height=400)

    except Exception as exc:
        st.error(f"Could not load model info: {exc}")


# ── Tab 3: Prediction Log ─────────────────────────────────────────────────────

_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"

with tab_monitor:
    st.header("Prediction Log")

    st.info(
        "This page shows every prediction scored through this dashboard, stored in a local "
        "SQLite audit log. **The model and data are not updated automatically.** "
        "To retrain with fresh CMS data, run `python pipeline/flow.py` manually.",
        icon="ℹ️",
    )

    # ── Pipeline freshness ────────────────────────────────────────────────────
    st.subheader("Pipeline Freshness")

    def _fmt_mtime(path: Path) -> str:
        if not path.exists():
            return "Not found — run the pipeline"
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return mtime.strftime("%Y-%m-%d %H:%M UTC")

    if not _FEATURES_PATH.exists() and not _MODEL_PATH.exists():
        st.caption("Running on cloud — pipeline artifacts are not available locally.")
    else:
        col_data, col_model = st.columns(2)
        col_data.metric("Data last refreshed", _fmt_mtime(_FEATURES_PATH))
        col_model.metric("Model last trained", _fmt_mtime(_MODEL_PATH))

    st.divider()

    # ── Prediction history ────────────────────────────────────────────────────
    limit = st.slider("Records to load", min_value=10, max_value=500, value=100, step=10)
    if st.button("Refresh", type="secondary"):
        st.cache_data.clear()

    try:
        history = get_history(api_url, limit=limit)
        stats = history["stats"]
        preds = history["predictions"]

        # Summary KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Logged", stats["total"])
        k2.metric("High Risk", stats["high_count"])
        k3.metric("Medium Risk", stats["medium_count"])
        k4.metric("Low Risk", stats["low_count"])
        k5.metric("Mean Score", f"{stats['mean_risk_score']:.3f}")

        st.divider()
        col_dist, col_time = st.columns(2)
        with col_dist:
            risk_distribution_chart(preds)
        with col_time:
            score_over_time_chart(preds)

        if preds:
            st.subheader("Recent Predictions")
            history_df = pd.DataFrame(preds).drop(
                columns=["input_features"], errors="ignore"
            )
            st.dataframe(history_df, use_container_width=True, height=300)

    except Exception as exc:
        st.error(f"Could not load prediction log: {exc}")
