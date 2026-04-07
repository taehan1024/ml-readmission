"""dashboard/components.py

Reusable Streamlit UI components.
All functions render directly into the current Streamlit context.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Risk gauge ────────────────────────────────────────────────────────────────

def risk_gauge(score: float, title: str = "Readmission Risk Score") -> None:
    """Render a Plotly gauge chart for a single risk score.

    Parameters
    ----------
    score:
        Float in [0, 1].
    title:
        Chart title shown above the gauge.
    """
    colour = (
        "#d62728" if score >= 0.65
        else "#ff7f0e" if score >= 0.40
        else "#2ca02c"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": colour},
            "steps": [
                {"range": [0, 40], "color": "#e8f5e9"},
                {"range": [40, 65], "color": "#fff3e0"},
                {"range": [65, 100], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": colour, "width": 4},
                "thickness": 0.75,
                "value": round(score * 100, 1),
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)


def risk_badge(risk_level: str) -> None:
    """Render a coloured risk-level badge using st.metric styling.

    Parameters
    ----------
    risk_level:
        One of ``"low"``, ``"medium"``, or ``"high"``.
    """
    colours = {"high": "🔴", "medium": "🟠", "low": "🟢"}
    icon = colours.get(risk_level, "⚪")
    st.metric(label="Risk Level", value=f"{icon} {risk_level.upper()}")


# ── Batch results table ───────────────────────────────────────────────────────

def batch_results_table(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    """Render a styled results table and return the underlying DataFrame.

    Parameters
    ----------
    predictions:
        List of PredictionResponse dicts from the batch API call.

    Returns
    -------
    pd.DataFrame
        The rendered DataFrame (also returned for CSV download).
    """
    df = pd.DataFrame(predictions)[
        ["facility_id", "facility_name", "risk_score", "risk_level"]
    ].rename(columns={
        "facility_id": "Facility ID",
        "facility_name": "Hospital Name",
        "risk_score": "Risk Score",
        "risk_level": "Risk Level",
    })

    def _colour_row(row):
        colour = (
            "background-color: #ffebee" if row["Risk Level"] == "high"
            else "background-color: #fff3e0" if row["Risk Level"] == "medium"
            else "background-color: #e8f5e9"
        )
        return [colour] * len(row)

    st.dataframe(
        df.style.apply(_colour_row, axis=1).format({"Risk Score": "{:.4f}"}),
        use_container_width=True,
    )
    return df


def batch_summary_metrics(response: dict[str, Any]) -> None:
    """Render a row of st.metric cards summarising a batch result.

    Parameters
    ----------
    response:
        BatchPredictionResponse dict from the API.
    """
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Hospitals", response["total"])
    col2.metric("High Risk", response["high_risk_count"])
    col3.metric(
        "Low / Medium Risk",
        response["total"] - response["high_risk_count"],
    )
    col4.metric("Mean Risk Score", f"{response['mean_risk_score']:.3f}")


# ── Monitoring charts ─────────────────────────────────────────────────────────

def risk_distribution_chart(predictions: list[dict[str, Any]]) -> None:
    """Render a bar chart of risk level counts from history.

    Parameters
    ----------
    predictions:
        List of prediction history records from GET /predictions/history.
    """
    if not predictions:
        st.info("No prediction history yet.")
        return

    df = pd.DataFrame(predictions)
    counts = df["risk_level"].value_counts().reindex(
        ["low", "medium", "high"], fill_value=0
    )
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=["#2ca02c", "#ff7f0e", "#d62728"],
        text=counts.values,
        textposition="outside",
    ))
    fig.update_layout(
        title="Prediction Distribution by Risk Level",
        xaxis_title="Risk Level",
        yaxis_title="Count",
        height=350,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def feature_importance_chart(importances: dict[str, float], top_n: int = 20) -> None:
    """Render a horizontal bar chart of XGBoost feature importances.

    Parameters
    ----------
    importances:
        Dict mapping feature name → importance score.
    top_n:
        Number of top features to display.
    """
    if not importances:
        st.info("Feature importances not available for this model.")
        return

    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [i[0] for i in reversed(items)]
    scores = [i[1] for i in reversed(items)]

    bar_colours = [
        "#2ca02c" if s >= 0.05
        else "#ff7f0e" if s >= 0.02
        else "#aec7e8"
        for s in scores
    ]

    fig = go.Figure(go.Bar(
        x=scores,
        y=names,
        orientation="h",
        marker_color=bar_colours,
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Feature Importances (top {len(items)})",
        xaxis_title="Importance (gain)",
        yaxis_title="",
        height=max(350, len(items) * 22),
        margin=dict(t=40, b=40, l=200, r=80),
    )
    st.plotly_chart(fig, use_container_width=True)


def training_metrics_chart(metrics: dict[str, float]) -> None:
    """Render a horizontal bar chart of training evaluation metrics.

    Parameters
    ----------
    metrics:
        Dict of metric name → float value (expected range [0, 1]).
    """
    if not metrics:
        return

    names = [k.upper() for k in metrics]
    values = list(metrics.values())

    bar_colours = [
        "#2ca02c" if v >= 0.75
        else "#ff7f0e" if v >= 0.50
        else "#d62728"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        marker_color=bar_colours,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey",
                  annotation_text="baseline (0.5)")
    fig.update_layout(
        title="Training Evaluation Metrics",
        yaxis=dict(range=[0, 1.1], title="Score"),
        xaxis_title="",
        height=320,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def state_risk_map(predictions: list[dict[str, Any]]) -> None:
    """Render a US choropleth map of mean risk score by state.

    Only renders when at least 2 distinct states are present in predictions.

    Parameters
    ----------
    predictions:
        List of prediction dicts containing ``state`` and ``risk_score`` keys.
    """
    if not predictions:
        return

    df = pd.DataFrame(predictions)
    if "state" not in df.columns or df["state"].isna().all():
        return

    state_df = (
        df.dropna(subset=["state"])
        .groupby("state")["risk_score"]
        .agg(mean_risk="mean", count="count")
        .reset_index()
    )
    if len(state_df) < 2:
        return

    fig = go.Figure(go.Choropleth(
        locations=state_df["state"],
        z=state_df["mean_risk"],
        locationmode="USA-states",
        colorscale=[[0, "#2ca02c"], [0.4, "#ff7f0e"], [0.65, "#d62728"], [1, "#8b0000"]],
        zmin=0,
        zmax=1,
        colorbar_title="Mean Risk",
        text=state_df.apply(
            lambda r: f"{r['state']}<br>Mean risk: {r['mean_risk']:.3f}<br>Hospitals: {r['count']}",
            axis=1,
        ),
        hoverinfo="text",
    ))
    fig.update_layout(
        title="Mean Readmission Risk by State",
        geo_scope="usa",
        height=420,
        margin=dict(t=40, b=0, l=0, r=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def score_over_time_chart(predictions: list[dict[str, Any]]) -> None:
    """Render a line chart of risk scores over time.

    Parameters
    ----------
    predictions:
        List of prediction history records (newest first from API).
    """
    if not predictions:
        return

    df = pd.DataFrame(predictions)[["timestamp", "risk_score"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    fig = go.Figure(go.Scatter(
        x=df["timestamp"],
        y=df["risk_score"],
        mode="lines+markers",
        line=dict(color="#1f77b4"),
        marker=dict(size=4),
    ))
    fig.add_hline(y=0.65, line_dash="dash", line_color="#d62728",
                  annotation_text="High risk threshold")
    fig.add_hline(y=0.40, line_dash="dash", line_color="#ff7f0e",
                  annotation_text="Medium risk threshold")
    fig.update_layout(
        title="Risk Score Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 1]),
        height=350,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
