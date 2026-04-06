"""Tests for insightml.intelligence.multicollinearity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from insightml.intelligence.multicollinearity import (
    compute_condition_number,
    compute_vif,
    removal_recommendations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _independent_df(n: int = 100) -> pd.DataFrame:
    """DataFrame with two truly independent numeric columns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
    })


def _correlated_df(n: int = 100) -> pd.DataFrame:
    """DataFrame where col_b is nearly equal to col_a (high multicollinearity)."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.001, n)
    return pd.DataFrame({
        "col_a": base,
        "col_b": base + noise,
        "col_c": rng.normal(0, 1, n),
    })


def _three_independent_df(n: int = 100) -> pd.DataFrame:
    """DataFrame with three independent numeric columns."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(5, 2, n),
        "x3": rng.uniform(0, 10, n),
    })


# ---------------------------------------------------------------------------
# compute_vif tests
# ---------------------------------------------------------------------------

class TestComputeVif:
    def test_returns_dataframe(self):
        df = _independent_df()
        result = compute_vif(df)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self):
        df = _independent_df()
        result = compute_vif(df)
        assert list(result.columns) == ["feature", "vif", "severity"]

    def test_row_count_equals_feature_count(self):
        df = _three_independent_df()
        result = compute_vif(df)
        assert len(result) == 3

    def test_independent_features_have_low_vif(self):
        """Independent features should have VIF close to 1.0."""
        df = _independent_df(200)
        result = compute_vif(df)
        # Both columns should be "low" severity
        assert (result["severity"] == "low").all()
        # VIF should be reasonably close to 1.0 for truly independent features
        assert (result["vif"] < 5.0).all()

    def test_high_collinearity_produces_high_vif(self):
        """Near-perfectly correlated features should produce a high VIF."""
        df = _correlated_df()
        result = compute_vif(df)
        high_vif_rows = result[result["feature"].isin(["col_a", "col_b"])]
        assert (high_vif_rows["vif"] > 10.0).any()

    def test_high_collinearity_severity_is_high(self):
        """Near-perfectly correlated features should be labelled 'high'."""
        df = _correlated_df()
        result = compute_vif(df)
        high_rows = result[result["feature"].isin(["col_a", "col_b"])]
        assert (high_rows["severity"] == "high").any()

    def test_severity_values_are_valid_strings(self):
        """severity must be one of the three allowed labels."""
        df = _three_independent_df()
        result = compute_vif(df)
        valid = {"low", "moderate", "high"}
        assert set(result["severity"].unique()).issubset(valid)

    def test_sorted_by_vif_descending(self):
        """Result should be sorted with highest VIF first."""
        df = _correlated_df()
        result = compute_vif(df)
        vif_vals = result["vif"].tolist()
        assert vif_vals == sorted(vif_vals, reverse=True)

    def test_explicit_numeric_cols_subset(self):
        """Passing numeric_cols should restrict which features are analysed."""
        df = _three_independent_df()
        result = compute_vif(df, numeric_cols=["x1", "x2"])
        assert set(result["feature"]) == {"x1", "x2"}

    def test_single_numeric_col_returns_empty(self):
        """Only one valid column — cannot compute regression; empty result."""
        df = pd.DataFrame({"solo": np.arange(20, dtype=float)})
        result = compute_vif(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["feature", "vif", "severity"]

    def test_with_missing_values(self):
        """Should handle NaNs via median imputation without raising."""
        rng = np.random.default_rng(7)
        n = 60
        a = rng.normal(0, 1, n)
        b = rng.normal(0, 1, n)
        a[[3, 7, 15]] = np.nan
        df = pd.DataFrame({"a": a, "b": b})
        result = compute_vif(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_vif_is_numeric(self):
        df = _three_independent_df()
        result = compute_vif(df)
        assert pd.api.types.is_numeric_dtype(result["vif"])


# ---------------------------------------------------------------------------
# compute_condition_number tests
# ---------------------------------------------------------------------------

class TestComputeConditionNumber:
    def test_returns_dict(self):
        df = _independent_df()
        result = compute_condition_number(df)
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = _independent_df()
        result = compute_condition_number(df)
        for key in ("condition_number", "severity", "eigenvalues", "n_near_zero"):
            assert key in result, f"Missing key: {key}"

    def test_condition_number_is_float(self):
        df = _independent_df()
        result = compute_condition_number(df)
        assert isinstance(result["condition_number"], float)

    def test_condition_number_is_positive(self):
        df = _independent_df()
        result = compute_condition_number(df)
        assert result["condition_number"] > 0

    def test_eigenvalues_is_list(self):
        df = _three_independent_df()
        result = compute_condition_number(df)
        assert isinstance(result["eigenvalues"], list)
        assert len(result["eigenvalues"]) == 3

    def test_n_near_zero_is_int(self):
        df = _independent_df()
        result = compute_condition_number(df)
        assert isinstance(result["n_near_zero"], int)

    def test_severity_values_are_valid(self):
        df = _three_independent_df()
        result = compute_condition_number(df)
        assert result["severity"] in ("low", "moderate", "severe")

    def test_low_severity_for_independent_features(self):
        """Independent features should yield a low condition number."""
        df = _independent_df(200)
        result = compute_condition_number(df)
        assert result["condition_number"] < 30.0
        assert result["severity"] == "low"

    def test_high_collinearity_raises_condition_number(self):
        """Near-singular matrix should have a very high condition number."""
        df = _correlated_df(200)
        result = compute_condition_number(df)
        assert result["condition_number"] > 30.0

    def test_single_col_returns_unknown(self):
        """Single column cannot form a matrix — should return sentinel."""
        df = pd.DataFrame({"solo": np.arange(20, dtype=float)})
        result = compute_condition_number(df)
        assert result["condition_number"] is None
        assert result["severity"] == "unknown"


# ---------------------------------------------------------------------------
# removal_recommendations tests
# ---------------------------------------------------------------------------

class TestRemovalRecommendations:
    def test_returns_list(self):
        df = _correlated_df()
        vif_df = compute_vif(df)
        recs = removal_recommendations(vif_df, df)
        assert isinstance(recs, list)

    def test_no_high_vif_returns_empty(self):
        """When all VIFs are below the threshold, no recommendations."""
        df = _independent_df(200)
        vif_df = compute_vif(df)
        # Force all VIF values below threshold
        vif_df["vif"] = 2.0
        recs = removal_recommendations(vif_df, df)
        assert recs == []

    def test_recommendation_keys(self):
        """Each recommendation dict must contain the expected keys."""
        df = _correlated_df()
        vif_df = compute_vif(df)
        recs = removal_recommendations(vif_df, df)
        assert len(recs) > 0
        for rec in recs:
            assert "feature" in rec
            assert "vif" in rec
            assert "recommendation" in rec
            assert "reason" in rec

    def test_recommendation_values_are_valid(self):
        """recommendation field must be one of the known string values."""
        df = _correlated_df()
        vif_df = compute_vif(df)
        recs = removal_recommendations(vif_df, df)
        valid = {"consider_removing", "keep_if_needed"}
        for rec in recs:
            assert rec["recommendation"] in valid

    def test_with_target_column(self):
        """Passing a target should add corr_with_target to each recommendation."""
        rng = np.random.default_rng(42)
        n = 100
        base = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "col_a": base,
            "col_b": base + rng.normal(0, 0.001, n),
            "col_c": rng.normal(0, 1, n),
            "target": rng.choice([0, 1], n),
        })
        vif_df = compute_vif(df[["col_a", "col_b", "col_c"]])
        recs = removal_recommendations(vif_df, df, target="target")
        # At least one rec should have corr_with_target
        assert any("corr_with_target" in r for r in recs)

    def test_without_target_column(self):
        """No target — recommendations still produced based on VIF alone."""
        df = _correlated_df()
        vif_df = compute_vif(df)
        recs = removal_recommendations(vif_df, df, target=None)
        assert isinstance(recs, list)
        for rec in recs:
            assert rec["recommendation"] == "consider_removing"

    def test_vif_field_is_float(self):
        df = _correlated_df()
        vif_df = compute_vif(df)
        recs = removal_recommendations(vif_df, df)
        for rec in recs:
            assert isinstance(rec["vif"], float)
