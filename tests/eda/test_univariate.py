"""Tests for UnivariateAnalysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from insightml.eda.result import EDAResult
from insightml.eda.univariate import UnivariateAnalysis

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_numeric():
    """Small DataFrame with only numeric columns."""
    rng = np.random.default_rng(42)
    n = 40
    return pd.DataFrame({
        "age": rng.normal(35, 10, n),
        "income": rng.normal(50000, 15000, n),
        "score": rng.uniform(0, 100, n),
    })


@pytest.fixture
def df_categorical():
    """Small DataFrame with only categorical columns."""
    rng = np.random.default_rng(42)
    n = 40
    return pd.DataFrame({
        "color": rng.choice(["red", "blue", "green"], n),
        "size": rng.choice(["S", "M", "L", "XL"], n),
    })


@pytest.fixture
def df_mixed():
    """Small DataFrame with numeric and categorical columns."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "num1": rng.normal(0, 1, n),
        "num2": rng.normal(5, 2, n),
        "cat1": rng.choice(["a", "b", "c"], n),
        "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def univariate_numeric(df_numeric):
    return UnivariateAnalysis(df_numeric)


@pytest.fixture
def univariate_categorical(df_categorical):
    return UnivariateAnalysis(df_categorical)


@pytest.fixture
def univariate_mixed(df_mixed):
    return UnivariateAnalysis(df_mixed)


@pytest.fixture
def eda_result(df_mixed):
    return EDAResult(df_mixed, target="target")


# ---------------------------------------------------------------------------
# Numeric column tests
# ---------------------------------------------------------------------------

class TestNumericStats:
    def test_stats_returns_dict(self, univariate_numeric):
        s = univariate_numeric.stats("age")
        assert isinstance(s, dict)

    def test_numeric_keys_present(self, univariate_numeric):
        s = univariate_numeric.stats("age")
        expected = {"count", "missing", "mean", "median", "std",
                    "min", "max", "q1", "q3", "iqr", "skewness",
                    "kurtosis", "likely_normal", "type"}
        assert expected.issubset(s.keys())

    def test_mean_is_finite(self, univariate_numeric):
        s = univariate_numeric.stats("age")
        assert np.isfinite(s["mean"])

    def test_std_is_finite(self, univariate_numeric):
        s = univariate_numeric.stats("age")
        assert np.isfinite(s["std"])

    def test_skewness_is_finite(self, univariate_numeric):
        s = univariate_numeric.stats("age")
        assert np.isfinite(s["skewness"])

    def test_count_matches_rows(self, univariate_numeric, df_numeric):
        s = univariate_numeric.stats("age")
        assert s["count"] == len(df_numeric)
        assert s["missing"] == 0

    def test_min_le_max(self, univariate_numeric):
        s = univariate_numeric.stats("age")
        assert s["min"] <= s["max"]

    def test_q1_le_q3(self, univariate_numeric):
        s = univariate_numeric.stats("age")
        assert s["q1"] <= s["q3"]

    def test_iqr_matches_quartiles(self, univariate_numeric):
        s = univariate_numeric.stats("age")
        assert abs(s["iqr"] - (s["q3"] - s["q1"])) < 1e-9

    def test_type_is_numeric(self, univariate_numeric):
        s = univariate_numeric.stats("age")
        assert s["type"] == "numeric"


# ---------------------------------------------------------------------------
# Categorical column tests
# ---------------------------------------------------------------------------

class TestCategoricalStats:
    def test_stats_returns_dict(self, univariate_categorical):
        s = univariate_categorical.stats("color")
        assert isinstance(s, dict)

    def test_categorical_keys_present(self, univariate_categorical):
        s = univariate_categorical.stats("color")
        expected = {"n_unique", "missing", "top_value", "top_freq",
                    "entropy", "value_counts"}
        assert expected.issubset(s.keys())

    def test_n_unique(self, univariate_categorical):
        s = univariate_categorical.stats("color")
        assert s["n_unique"] == 3

    def test_entropy_non_negative(self, univariate_categorical):
        s = univariate_categorical.stats("color")
        assert s["entropy"] >= 0.0

    def test_top_value_is_string(self, univariate_categorical):
        s = univariate_categorical.stats("color")
        assert isinstance(s["top_value"], str)
        assert s["top_value"] in ("red", "blue", "green")

    def test_top_freq_is_positive(self, univariate_categorical):
        s = univariate_categorical.stats("color")
        assert s["top_freq"] > 0

    def test_value_counts_is_dict(self, univariate_categorical):
        s = univariate_categorical.stats("color")
        assert isinstance(s["value_counts"], dict)
        assert len(s["value_counts"]) == 3

    def test_value_counts_sum_equals_n(self, univariate_categorical, df_categorical):
        s = univariate_categorical.stats("color")
        total = sum(s["value_counts"].values())
        assert total == len(df_categorical)


# ---------------------------------------------------------------------------
# Mixed DataFrame tests
# ---------------------------------------------------------------------------

class TestMixed:
    def test_all_columns_have_stats(self, univariate_mixed, df_mixed):
        for col in df_mixed.columns:
            s = univariate_mixed.stats(col)
            assert isinstance(s, dict)
            assert "type" in s

    def test_raises_for_unknown_column(self, univariate_mixed):
        with pytest.raises(KeyError):
            univariate_mixed.stats("nonexistent")


# ---------------------------------------------------------------------------
# show() and plot() tests
# ---------------------------------------------------------------------------

class TestShowPlot:
    def test_show_does_not_crash(self, univariate_mixed, monkeypatch):
        """show() should not raise; monkeypatch display_html to no-op."""
        monkeypatch.setattr(
            "insightml.viz.display.display_html", lambda html: None
        )
        monkeypatch.setattr(
            "insightml.eda._base.display_html", lambda html: None
        )
        univariate_mixed.show()

    def test_plot_returns_dict_of_figures(self, univariate_mixed):
        result = univariate_mixed.plot()
        assert isinstance(result, dict)
        for fig in result.values():
            assert isinstance(fig, go.Figure)

    def test_plot_numeric_col_returns_figures(self, univariate_mixed):
        result = univariate_mixed.plot("num1")
        # numeric columns produce hist + box = 2 figures -> returned as dict
        assert isinstance(result, dict)
        assert len(result) == 2
        for fig in result.values():
            assert isinstance(fig, go.Figure)

    def test_plot_categorical_col_returns_single_figure(self, univariate_mixed):
        result = univariate_mixed.plot("cat1")
        assert isinstance(result, go.Figure)

    def test_plot_raises_for_unknown_column(self, univariate_mixed):
        with pytest.raises(KeyError):
            univariate_mixed.plot("nonexistent")


# ---------------------------------------------------------------------------
# to_dict() and to_dataframe()
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_returns_dict(self, univariate_mixed):
        d = univariate_mixed.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_stats_key(self, univariate_mixed):
        d = univariate_mixed.to_dict()
        assert "stats" in d

    def test_to_dataframe_returns_dataframe(self, univariate_mixed):
        df = univariate_mixed.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_returns_string(self, univariate_mixed):
        s = univariate_mixed.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_mentions_columns(self, univariate_mixed):
        s = univariate_mixed.summary()
        assert "column" in s.lower() or "4" in s or "numeric" in s.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_null_column(self):
        df = pd.DataFrame({
            "all_null": [np.nan] * 50,
            "normal": np.random.default_rng(0).normal(0, 1, 50),
        })
        eda = EDAResult(df)
        u = eda.univariate
        s = u.stats("all_null")
        assert s["missing"] == 50

    def test_single_value_column(self):
        df = pd.DataFrame({
            "constant": ["x"] * 60,
            "num": np.random.default_rng(1).normal(0, 1, 60),
        })
        eda = EDAResult(df)
        u = eda.univariate
        s = u.stats("constant")
        assert s["type"] == "constant"

    def test_boolean_column(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "flag": rng.choice([True, False], 40),
            "num": rng.normal(0, 1, 40),
        })
        u = UnivariateAnalysis(df)
        s = u.stats("flag")
        assert s["type"] in ("boolean", "categorical")
        assert "n_unique" in s
        assert s["n_unique"] == 2
