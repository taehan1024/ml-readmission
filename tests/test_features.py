"""tests/test_features.py

Unit tests for pipeline/features.py.
Uses a synthetic 3-hospital long-format DataFrame — no file I/O required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import (
    EXCESS_RATIO_THRESHOLD,
    MEASURE_MAP,
    add_aggregate_features,
    add_target,
    clean_raw,
    encode_state,
    pivot_wide,
    validate_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_raw(n_hospitals: int = 3) -> pd.DataFrame:
    """Build a minimal synthetic long-format HRRP DataFrame.

    Parameters
    ----------
    n_hospitals:
        Number of distinct hospitals to generate.

    Returns
    -------
    pd.DataFrame
        Synthetic raw data matching the CMS column schema.
    """
    measures = list(MEASURE_MAP.keys())
    rows = []
    for i in range(n_hospitals):
        fid = f"H{i:04d}"
        for measure in measures:
            # Hospital 0: all excess ratios = 0.9 (low risk)
            # Hospital 1: all excess ratios = 1.1 (high risk)
            # Hospital 2: mixed — two measures N/A
            if i == 0:
                ratio = "0.9"
                pred = "10.0"
                exp = "11.1"
                discharges = "200"
            elif i == 1:
                ratio = "1.1"
                pred = "12.0"
                exp = "10.9"
                discharges = "150"
            else:
                # For hospital 2, make two measures N/A
                if measure in ("READM-30-AMI-HRRP", "READM-30-CABG-HRRP"):
                    ratio = "N/A"
                    pred = "N/A"
                    exp = "N/A"
                    discharges = "Too Few to Report"
                else:
                    ratio = "0.95"
                    pred = "11.0"
                    exp = "11.5"
                    discharges = "120"
            rows.append(
                {
                    "facility_id": fid,
                    "facility_name": f"Hospital {i}",
                    "state": ["AL", "CA", "TX"][i],
                    "measure_name": measure,
                    "number_of_discharges": discharges,
                    "footnote": "",
                    "excess_readmission_ratio": ratio,
                    "predicted_readmission_rate": pred,
                    "expected_readmission_rate": exp,
                    "number_of_readmissions": "20",
                    "start_date": "07/01/2021",
                    "end_date": "06/30/2024",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Synthetic raw 3-hospital long-format DataFrame."""
    return _make_raw(n_hospitals=3)


@pytest.fixture
def feature_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Fully engineered wide-format DataFrame from synthetic data."""
    cleaned = clean_raw(raw_df)
    wide = pivot_wide(cleaned)
    wide = add_aggregate_features(wide)
    wide = encode_state(wide)
    wide = add_target(wide)
    return wide


# ── clean_raw ─────────────────────────────────────────────────────────────────

class TestCleanRaw:
    def test_numeric_coercion(self, raw_df: pd.DataFrame) -> None:
        """Non-numeric sentinel values become NaN after cleaning."""
        cleaned = clean_raw(raw_df)
        assert cleaned["excess_readmission_ratio"].dtype == float

    def test_na_strings_become_nan(self, raw_df: pd.DataFrame) -> None:
        """'N/A' and 'Too Few to Report' are coerced to NaN."""
        cleaned = clean_raw(raw_df)
        # Hospital 2 has 2 N/A rows
        h2 = cleaned[cleaned["facility_id"] == "H0002"]
        null_count = h2["excess_readmission_ratio"].isna().sum()
        assert null_count == 2

    def test_unknown_measures_dropped(self, raw_df: pd.DataFrame) -> None:
        """Rows with unrecognised measure names are removed."""
        raw_with_extra = raw_df.copy()
        extra = raw_with_extra.iloc[[0]].copy()
        extra["measure_name"] = "READM-30-UNKNOWN-HRRP"
        raw_with_extra = pd.concat([raw_with_extra, extra], ignore_index=True)
        cleaned = clean_raw(raw_with_extra)
        assert "READM-30-UNKNOWN-HRRP" not in cleaned["measure_name"].values

    def test_measure_short_column_added(self, raw_df: pd.DataFrame) -> None:
        """``measure_short`` column is added with abbreviated names."""
        cleaned = clean_raw(raw_df)
        assert "measure_short" in cleaned.columns
        assert set(cleaned["measure_short"].unique()) == set(MEASURE_MAP.values())


# ── pivot_wide ────────────────────────────────────────────────────────────────

class TestPivotWide:
    def test_one_row_per_hospital(self, raw_df: pd.DataFrame) -> None:
        """After pivoting, there is exactly one row per unique facility_id."""
        cleaned = clean_raw(raw_df)
        wide = pivot_wide(cleaned)
        assert len(wide) == raw_df["facility_id"].nunique()

    def test_expected_columns_present(self, raw_df: pd.DataFrame) -> None:
        """Wide DataFrame contains per-measure columns for excess_ratio."""
        cleaned = clean_raw(raw_df)
        wide = pivot_wide(cleaned)
        for short in MEASURE_MAP.values():
            assert f"{short}_excess_ratio" in wide.columns

    def test_metadata_columns_present(self, raw_df: pd.DataFrame) -> None:
        """facility_name and state are preserved in wide format."""
        cleaned = clean_raw(raw_df)
        wide = pivot_wide(cleaned)
        assert "facility_name" in wide.columns
        assert "state" in wide.columns


# ── add_aggregate_features ────────────────────────────────────────────────────

class TestAggregateFeatures:
    def test_mean_excess_ratio_computed(self, raw_df: pd.DataFrame) -> None:
        """mean_excess_ratio is non-null for all hospitals with data."""
        cleaned = clean_raw(raw_df)
        wide = pivot_wide(cleaned)
        agg = add_aggregate_features(wide)
        assert "mean_excess_ratio" in agg.columns
        # All 3 hospitals have at least some valid measures
        assert agg["mean_excess_ratio"].notna().all()

    def test_n_valid_measures_range(self, raw_df: pd.DataFrame) -> None:
        """n_valid_measures is between 1 and 6 for all hospitals."""
        cleaned = clean_raw(raw_df)
        wide = pivot_wide(cleaned)
        agg = add_aggregate_features(wide)
        assert (agg["n_valid_measures"] >= 1).all()
        assert (agg["n_valid_measures"] <= 6).all()

    def test_hospital2_has_fewer_valid_measures(
        self, raw_df: pd.DataFrame
    ) -> None:
        """Hospital 2 (with 2 N/A measures) has n_valid_measures == 4."""
        cleaned = clean_raw(raw_df)
        wide = pivot_wide(cleaned)
        agg = add_aggregate_features(wide)
        h2 = agg[agg["facility_id"] == "H0002"]
        assert int(h2["n_valid_measures"].iloc[0]) == 4


# ── encode_state ──────────────────────────────────────────────────────────────

class TestEncodeState:
    def test_state_encoded_is_integer(self, raw_df: pd.DataFrame) -> None:
        """state_encoded column contains integer dtype."""
        cleaned = clean_raw(raw_df)
        wide = pivot_wide(cleaned)
        encoded = encode_state(wide)
        assert pd.api.types.is_integer_dtype(encoded["state_encoded"])

    def test_n_unique_matches_states(self, raw_df: pd.DataFrame) -> None:
        """Number of unique encoded values matches number of unique states."""
        cleaned = clean_raw(raw_df)
        wide = pivot_wide(cleaned)
        encoded = encode_state(wide)
        assert encoded["state_encoded"].nunique() == encoded["state"].nunique()


# ── add_target ────────────────────────────────────────────────────────────────

class TestAddTarget:
    def test_target_is_binary(self, feature_df: pd.DataFrame) -> None:
        """Target column contains only 0 and 1."""
        assert set(feature_df["high_readmission_risk"].unique()).issubset({0, 1})

    def test_low_risk_hospital_label(self, feature_df: pd.DataFrame) -> None:
        """Hospital 0 (all ratios=0.9) is labelled low risk (0)."""
        h0 = feature_df[feature_df["facility_id"] == "H0000"]
        assert h0["high_readmission_risk"].iloc[0] == 0

    def test_high_risk_hospital_label(self, feature_df: pd.DataFrame) -> None:
        """Hospital 1 (all ratios=1.1) is labelled high risk (1)."""
        h1 = feature_df[feature_df["facility_id"] == "H0001"]
        assert h1["high_readmission_risk"].iloc[0] == 1

    def test_target_threshold(self, feature_df: pd.DataFrame) -> None:
        """mean_excess_ratio == threshold boundary maps to low risk (not >)."""
        assert EXCESS_RATIO_THRESHOLD == 1.0


# ── validate_features ─────────────────────────────────────────────────────────

class TestValidateFeatures:
    def test_valid_df_passes(self, feature_df: pd.DataFrame) -> None:
        """Fully engineered DataFrame passes all validation checks."""
        validate_features(feature_df)  # should not raise

    def test_null_in_key_column_raises(self, feature_df: pd.DataFrame) -> None:
        """Null in a key column triggers ValueError."""
        bad = feature_df.copy()
        bad.loc[bad.index[0], "state_encoded"] = np.nan
        with pytest.raises(ValueError, match="Unexpected nulls"):
            validate_features(bad)

    def test_out_of_range_ratio_raises(self, feature_df: pd.DataFrame) -> None:
        """mean_excess_ratio above upper bound triggers AssertionError."""
        bad = feature_df.copy()
        bad.loc[bad.index[0], "mean_excess_ratio"] = 99.0
        with pytest.raises(AssertionError):
            validate_features(bad)


# ── output shape ──────────────────────────────────────────────────────────────

class TestOutputShape:
    def test_row_count(self, feature_df: pd.DataFrame) -> None:
        """Output has one row per hospital (3 hospitals in synthetic data)."""
        assert len(feature_df) == 3

    def test_feature_columns_are_numeric(self, feature_df: pd.DataFrame) -> None:
        """All engineered numeric feature columns have float or int dtype."""
        skip = {"facility_id", "facility_name", "state", "high_readmission_risk"}
        numeric_cols = [c for c in feature_df.columns if c not in skip]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(feature_df[col]), (
                f"Column '{col}' is not numeric: {feature_df[col].dtype}"
            )
