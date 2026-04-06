"""Tests for insightml.intelligence.feature_importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insightml.intelligence.feature_importance import compute_feature_importance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clf_df(n: int = 120) -> pd.DataFrame:
    """Numeric features with a clear binary classification target."""
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    # target is strongly influenced by x1
    y = (x1 + rng.normal(0, 0.2, n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


def _reg_df(n: int = 120) -> pd.DataFrame:
    """Numeric features with a continuous regression target."""
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 3.0 * x1 - 1.5 * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


def _cat_target_df(n: int = 120) -> pd.DataFrame:
    """Numeric features with a categorical string target."""
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    classes = np.where(x1 > 0.5, "high", np.where(x1 < -0.5, "low", "mid"))
    return pd.DataFrame({"x1": x1, "x2": x2, "label": classes})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeFeatureImportance:
    def test_returns_dataframe(self):
        df = _clf_df()
        result = compute_feature_importance(df, target="target")
        assert isinstance(result, pd.DataFrame)

    def test_has_feature_column(self):
        df = _clf_df()
        result = compute_feature_importance(df, target="target")
        assert "feature" in result.columns

    def test_has_composite_rank_column(self):
        df = _clf_df()
        result = compute_feature_importance(df, target="target")
        assert "composite_rank" in result.columns

    def test_has_expected_score_columns_classification(self):
        """At least mi, abs_corr, and f_stat should be present for classification."""
        df = _clf_df()
        result = compute_feature_importance(df, target="target", task="classification")
        for col in ("mi", "abs_corr", "f_stat"):
            assert col in result.columns, f"Expected column '{col}' not found"

    def test_has_expected_score_columns_regression(self):
        """At least mi, abs_corr, and f_stat should be present for regression."""
        df = _reg_df()
        result = compute_feature_importance(df, target="target", task="regression")
        for col in ("mi", "abs_corr", "f_stat"):
            assert col in result.columns, f"Expected column '{col}' not found"

    def test_n_rows_equals_n_features(self):
        """One row per feature column (excluding target)."""
        df = _clf_df()
        feature_cols = [c for c in df.columns if c != "target"]
        result = compute_feature_importance(df, target="target")
        assert len(result) == len(feature_cols)

    def test_composite_rank_is_numeric(self):
        df = _clf_df()
        result = compute_feature_importance(df, target="target")
        assert pd.api.types.is_numeric_dtype(result["composite_rank"])

    def test_sorted_ascending_by_composite_rank(self):
        """Rows should be sorted by composite_rank in ascending order (lower = more important)."""
        df = _clf_df()
        result = compute_feature_importance(df, target="target")
        ranks = result["composite_rank"].tolist()
        assert ranks == sorted(ranks)

    def test_most_important_feature_has_lowest_rank(self):
        """x1 drives the target; it should rank higher (lower composite_rank) than noise."""
        df = _clf_df()
        result = compute_feature_importance(df, target="target", task="classification")
        top_feature = result.iloc[0]["feature"]
        assert top_feature == "x1", (
            f"Expected 'x1' to be top feature, got '{top_feature}'"
        )

    def test_works_with_explicit_classification_task(self):
        df = _clf_df()
        result = compute_feature_importance(df, target="target", task="classification")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_works_with_explicit_regression_task(self):
        df = _reg_df()
        result = compute_feature_importance(df, target="target", task="regression")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_works_with_categorical_target(self):
        """String target should be label-encoded internally without raising."""
        df = _cat_target_df()
        result = compute_feature_importance(df, target="label")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # x1 and x2

    def test_works_with_all_numeric_features(self):
        """Regression dataset — all features are numeric; should work cleanly."""
        df = _reg_df()
        result = compute_feature_importance(df, target="target")
        feature_cols = [c for c in df.columns if c != "target"]
        assert set(result["feature"]) == set(feature_cols)

    def test_raises_on_missing_target(self):
        df = _clf_df()
        with pytest.raises(KeyError):
            compute_feature_importance(df, target="nonexistent_column")

    def test_no_target_column_in_feature_list(self):
        """Target column must not appear in the feature column of the result."""
        df = _clf_df()
        result = compute_feature_importance(df, target="target")
        assert "target" not in result["feature"].values
