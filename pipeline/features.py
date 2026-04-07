"""pipeline/features.py

Feature engineering for the CMS HRRP dataset.

The raw data is in **long format**: one row per (hospital × measure).
Each hospital has up to 6 readmission measures:
  AMI, CABG, COPD, Heart Failure, Hip/Knee, Pneumonia.

This module pivots to **wide format** (one row per hospital) and engineers:
  - Per-measure numeric features (excess ratio, predicted/expected rates,
    discharge volume)
  - Cross-measure aggregate features (mean, max, counts)
  - State label encoding
  - Binary target: high_readmission_risk = 1 when mean excess ratio > 1.0
    (hospital is penalised by CMS on average across its reported measures)

Usage
-----
    python pipeline/features.py                    # uses cached raw parquet
    python pipeline/features.py --raw path/to.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "hrrp_raw.parquet"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"

# Mapping from CMS measure_name values to short prefixes used in column names
MEASURE_MAP: dict[str, str] = {
    "READM-30-AMI-HRRP": "ami",
    "READM-30-CABG-HRRP": "cabg",
    "READM-30-COPD-HRRP": "copd",
    "READM-30-HF-HRRP": "hf",
    "READM-30-HIP-KNEE-HRRP": "hip_knee",
    "READM-30-PN-HRRP": "pn",
}

# Numeric columns in the raw data that we want to pivot
NUMERIC_COLS = [
    "excess_readmission_ratio",
    "predicted_readmission_rate",
    "expected_readmission_rate",
    "number_of_discharges",
]

# Target threshold: mean excess ratio above this → high risk
EXCESS_RATIO_THRESHOLD = 1.0

# Validation bounds for key derived columns
VALIDATION_RULES: dict[str, tuple[float, float]] = {
    "mean_excess_ratio": (0.1, 3.0),
    "max_excess_ratio": (0.1, 3.0),
    "mean_predicted_rate": (0.0, 50.0),
    "n_valid_measures": (1.0, 6.0),
}


# ── Cleaning ──────────────────────────────────────────────────────────────────

def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to float, replacing non-numeric strings with NaN.

    CMS uses "N/A", "Too Few to Report", and empty strings as sentinel
    values. ``pd.to_numeric`` with ``errors='coerce'`` handles all of these.

    Parameters
    ----------
    series:
        Raw string/object column from the CMS dataset.

    Returns
    -------
    pd.Series
        Float series with non-parseable values replaced by ``np.nan``.
    """
    return pd.to_numeric(series, errors="coerce")


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns and normalise the measure name column.

    Parameters
    ----------
    df:
        Raw HRRP DataFrame as loaded from ``data/raw/hrrp_raw.parquet``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame; numeric columns are float, unknmeasures dropped.
    """
    df = df.copy()
    for col in NUMERIC_COLS:
        df[col] = _coerce_numeric(df[col])

    # Keep only the six known measures (drop any future additions or typos)
    df = df[df["measure_name"].isin(MEASURE_MAP)].copy()
    df["measure_short"] = df["measure_name"].map(MEASURE_MAP)

    logger.info(
        "After cleaning: %d rows across %d hospitals",
        len(df),
        df["facility_id"].nunique(),
    )
    return df


# ── Pivoting ──────────────────────────────────────────────────────────────────

def pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format HRRP data to one row per hospital.

    Creates columns named ``{measure}_{metric}``, e.g. ``ami_excess_ratio``,
    ``hf_predicted_readmission_rate``, etc.

    Parameters
    ----------
    df:
        Cleaned long-format DataFrame from :func:`clean_raw`.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by ``facility_id``.
    """
    frames: list[pd.DataFrame] = []

    for col in NUMERIC_COLS:
        pivot = df.pivot_table(
            index="facility_id",
            columns="measure_short",
            values=col,
            aggfunc="mean",   # mean handles duplicate rows for same hospital/measure
        )
        # Shorten column suffix for readability
        short_suffix = (
            col.replace("excess_readmission_ratio", "excess_ratio")
               .replace("predicted_readmission_rate", "predicted_rate")
               .replace("expected_readmission_rate", "expected_rate")
               .replace("number_of_discharges", "discharges")
        )
        pivot.columns = [f"{measure}_{short_suffix}" for measure in pivot.columns]
        frames.append(pivot)

    wide = pd.concat(frames, axis=1).reset_index()

    # Attach hospital metadata (name, state) — take first occurrence per hospital
    meta = (
        df.sort_values("facility_id")
        .groupby("facility_id")[["facility_name", "state"]]
        .first()
        .reset_index()
    )
    wide = meta.merge(wide, on="facility_id", how="inner")

    logger.info(
        "Pivoted to wide format: %d hospitals x %d columns",
        len(wide),
        wide.shape[1],
    )
    return wide


# ── Aggregate features ────────────────────────────────────────────────────────

def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-measure summary statistics for each hospital.

    Adds the following columns:

    - ``mean_excess_ratio``: mean excess readmission ratio across measures
    - ``max_excess_ratio``: worst (highest) excess ratio across measures
    - ``min_excess_ratio``: best (lowest) excess ratio across measures
    - ``n_measures_over_threshold``: count of measures with ratio > 1.0
    - ``n_valid_measures``: count of measures with non-null excess ratio
    - ``mean_predicted_rate``: mean predicted readmission rate
    - ``total_discharges``: sum of discharges across all measures

    Parameters
    ----------
    df:
        Wide-format DataFrame from :func:`pivot_wide`.

    Returns
    -------
    pd.DataFrame
        DataFrame with aggregate feature columns appended.
    """
    df = df.copy()

    excess_cols = [c for c in df.columns if c.endswith("_excess_ratio")]
    predicted_cols = [c for c in df.columns if c.endswith("_predicted_rate")]
    discharge_cols = [c for c in df.columns if c.endswith("_discharges")]

    excess = df[excess_cols]
    df["mean_excess_ratio"] = excess.mean(axis=1)
    df["max_excess_ratio"] = excess.max(axis=1)
    df["min_excess_ratio"] = excess.min(axis=1)
    df["n_measures_over_threshold"] = (excess > EXCESS_RATIO_THRESHOLD).sum(axis=1)
    df["n_valid_measures"] = excess.notna().sum(axis=1)
    df["mean_predicted_rate"] = df[predicted_cols].mean(axis=1)
    df["total_discharges"] = df[discharge_cols].sum(axis=1, min_count=1)

    logger.info("Added 7 aggregate features.")
    return df


# ── State encoding ────────────────────────────────────────────────────────────

def encode_state(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode the ``state`` column as ``state_encoded`` (integer).

    Keeps the original ``state`` column for interpretability.
    Unknown future states will map to -1 via a consistent encoder.

    Parameters
    ----------
    df:
        Wide-format DataFrame containing a ``state`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``state_encoded`` column added.
    """
    df = df.copy()
    le = LabelEncoder()
    df["state_encoded"] = le.fit_transform(df["state"].fillna("Unknown"))
    logger.info(
        "Encoded %d unique states.", df["state_encoded"].nunique()
    )
    return df


# ── Target variable ───────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target column ``high_readmission_risk``.

    A hospital is labelled **high risk (1)** when its mean excess readmission
    ratio across all reported measures exceeds 1.0 — meaning it readmits more
    patients than the national risk-adjusted expectation and faces CMS penalty.

    Hospitals with no valid measures are dropped.

    Parameters
    ----------
    df:
        Wide-format DataFrame with ``mean_excess_ratio`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``high_readmission_risk`` (0/1) column added and
        rows lacking a computable target removed.
    """
    df = df.copy()
    before = len(df)
    df = df.dropna(subset=["mean_excess_ratio"])
    dropped = before - len(df)
    if dropped:
        logger.warning(
            "Dropped %d hospitals with no valid excess ratio data.", dropped
        )

    df["high_readmission_risk"] = (
        df["mean_excess_ratio"] > EXCESS_RATIO_THRESHOLD
    ).astype(int)

    pos = df["high_readmission_risk"].sum()
    logger.info(
        "Target distribution: %d high-risk (%.1f%%), %d low-risk (%.1f%%)",
        pos,
        100 * pos / len(df),
        len(df) - pos,
        100 * (len(df) - pos) / len(df),
    )
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def validate_features(df: pd.DataFrame) -> None:
    """Assert that key feature columns are within expected ranges.

    Raises
    ------
    AssertionError
        If any validation check fails.
    ValueError
        If any key column contains nulls after feature engineering.
    """
    key_cols = ["facility_id", "state_encoded", "mean_excess_ratio",
                "n_valid_measures", "high_readmission_risk"]
    null_counts = df[key_cols].isnull().sum()
    if null_counts.any():
        raise ValueError(
            f"Unexpected nulls in key columns after feature engineering:\n"
            f"{null_counts[null_counts > 0]}"
        )

    for col, (lo, hi) in VALIDATION_RULES.items():
        if col not in df.columns:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        assert col_min >= lo, (
            f"{col} min={col_min:.4f} is below expected lower bound {lo}"
        )
        assert col_max <= hi, (
            f"{col} max={col_max:.4f} exceeds expected upper bound {hi}"
        )

    assert df["high_readmission_risk"].isin([0, 1]).all(), (
        "Target column contains values other than 0 and 1"
    )

    logger.info("All validation checks passed.")


# ── Report ────────────────────────────────────────────────────────────────────

def _report(df: pd.DataFrame) -> None:
    """Print a summary of the engineered feature matrix to stdout.

    Parameters
    ----------
    df:
        Final wide-format feature DataFrame.
    """
    feature_cols = [
        c for c in df.columns
        if c not in ("facility_id", "facility_name", "state", "high_readmission_risk")
    ]
    print(f"\n{'='*60}")
    print("  Feature Matrix Summary")
    print(f"{'='*60}")
    print(f"  Hospitals : {len(df):,}")
    print(f"  Features  : {len(feature_cols)}")
    print(f"  Target    : high_readmission_risk")
    print(f"    High-risk : {df['high_readmission_risk'].sum():,} "
          f"({100*df['high_readmission_risk'].mean():.1f}%)")
    print(f"    Low-risk  : {(df['high_readmission_risk']==0).sum():,}")
    print(f"\n  Feature columns:")
    for col in feature_cols:
        non_null = df[col].notna().sum()
        pct = 100 * non_null / len(df)
        print(f"    {col:<45}  {pct:5.1f}% non-null")
    print(f"{'='*60}\n")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def build_features(
    raw_path: Path = DEFAULT_RAW_PATH,
    out_path: Path = DEFAULT_OUT_PATH,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Steps:
        1. Load raw parquet
        2. Clean and coerce types
        3. Pivot long → wide
        4. Add aggregate features
        5. Encode state
        6. Add target variable
        7. Validate
        8. Save to ``data/processed/features.parquet``

    Parameters
    ----------
    raw_path:
        Path to the raw HRRP parquet file (default: ``data/raw/hrrp_raw.parquet``).
    out_path:
        Destination path for the processed feature parquet.

    Returns
    -------
    pd.DataFrame
        Final feature matrix ready for model training.
    """
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. "
            "Run `python pipeline/ingest.py` first."
        )

    logger.info("Loading raw data from %s", raw_path)
    raw = pd.read_parquet(raw_path)
    logger.info("Raw shape: %d rows x %d cols", *raw.shape)

    df = clean_raw(raw)
    df = pivot_wide(df)
    df = add_aggregate_features(df)
    df = encode_state(df)
    df = add_target(df)
    validate_features(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved feature matrix to %s", out_path)

    _report(df)
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the feature engineering pipeline."""
    parser = argparse.ArgumentParser(
        description="Build feature matrix from raw CMS HRRP parquet."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Path to raw parquet file (default: data/raw/hrrp_raw.parquet)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Output path for feature parquet (default: data/processed/features.parquet)",
    )
    args = parser.parse_args()

    try:
        build_features(raw_path=args.raw, out_path=args.out)
    except (FileNotFoundError, AssertionError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("Feature engineering failed unexpectedly")
        sys.exit(1)


if __name__ == "__main__":
    main()
