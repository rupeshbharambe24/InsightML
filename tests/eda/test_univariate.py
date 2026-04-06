"""Tests for UnivariateAnalysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from insightml.eda.result import EDAResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "num1": rng.normal(0, 1, n),
        "num2": rng.normal(5, 2, n),
        "cat1": rng.choice(["a", "b", "c"], n),
        "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def eda_result():
    return EDAResult(_make_df(), target="target")


@pytest.fixture
def univariate(eda_result):
    return eda_result.univariate


# ---------------------------------------------------------------------------
# Basic type and structure tests
# ---------------------------------------------------------------------------

def test_univariate_returns_object(eda_result):
    from insightml.eda.univariate import UnivariateAnalysis
    assert isinstance(eda_result.univariate, UnivariateAnalysis)


def test_univariate_is_cached(eda_result):
    """Accessing .univariate twice returns the same object."""
    assert eda_result.univariate is eda_result.univariate


# ---------------------------------------------------------------------------
# stats() accessor
# ---------------------------------------------------------------------------

def test_stats_returns_dict(univariate):
    result = univariate.stats("num1")
    assert isinstance(result, dict)


def test_numeric_stats_keys_present(univariate):
    s = univariate.stats("num1")
    expected_keys = {"count", "missing", "mean", "median", "std",
                     "min", "max", "q1", "q3", "iqr", "skewness",
                     "kurtosis", "likely_normal"}
    assert expected_keys.issubset(s.keys())


def test_numeric_stats_values_are_finite(univariate):
    s = univariate.stats("num1")
    for key in ("mean", "median", "std", "min", "max", "q1", "q3", "iqr"):
        assert np.isfinite(s[key]), f"{key} is not finite"


def test_numeric_count_matches_non_null(univariate):
    s = univariate.stats("num1")
    # _make_df has no missing values in num1
    assert s["count"] == 100
    assert s["missing"] == 0


def test_categorical_stats_keys_present(univariate):
    s = univariate.stats("cat1")
    expected_keys = {"n_unique", "missing", "top_value", "top_freq",
                     "entropy", "value_counts"}
    assert expected_keys.issubset(s.keys())


def test_categorical_n_unique(univariate):
    s = univariate.stats("cat1")
    assert s["n_unique"] == 3


def test_categorical_top_value_is_string(univariate):
    s = univariate.stats("cat1")
    assert isinstance(s["top_value"], str)
    assert s["top_value"] in ("a", "b", "c")


def test_categorical_entropy_non_negative(univariate):
    s = univariate.stats("cat1")
    assert s["entropy"] >= 0.0


def test_categorical_value_counts_is_dict(univariate):
    s = univariate.stats("cat1")
    assert isinstance(s["value_counts"], dict)
    assert len(s["value_counts"]) == 3


def test_stats_raises_for_unknown_column(univariate):
    with pytest.raises(KeyError):
        univariate.stats("does_not_exist")


# ---------------------------------------------------------------------------
# plot() accessor
# ---------------------------------------------------------------------------

def test_plot_numeric_returns_dict_of_figures(univariate):
    result = univariate.plot("num1")
    # numeric columns produce hist + box = 2 figures → returned as dict
    assert isinstance(result, dict)
    assert len(result) == 2


def test_plot_numeric_figures_are_go_figures(univariate):
    figures = univariate.plot("num1")
    for fig in figures.values():
        assert isinstance(fig, go.Figure)


def test_plot_categorical_returns_single_figure(univariate):
    # categorical column produces one freq bar chart → returned directly
    result = univariate.plot("cat1")
    assert isinstance(result, go.Figure)


def test_plot_raises_for_unknown_column(univariate):
    with pytest.raises(KeyError):
        univariate.plot("nonexistent_col")


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

def test_summary_returns_string(univariate):
    s = univariate.summary()
    assert isinstance(s, str)
    assert len(s) > 0


def test_summary_mentions_column_count(univariate):
    s = univariate.summary()
    # Should mention the number of columns analyzed
    assert "4" in s or "column" in s.lower()


# ---------------------------------------------------------------------------
# to_dict() and to_dataframe()
# ---------------------------------------------------------------------------

def test_to_dict_returns_dict(univariate):
    d = univariate.to_dict()
    assert isinstance(d, dict)


def test_to_dict_has_stats_key(univariate):
    d = univariate.to_dict()
    assert "stats" in d


def test_to_dataframe_returns_dataframe(univariate):
    df = univariate.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


# ---------------------------------------------------------------------------
# Edge case: all-null column
# ---------------------------------------------------------------------------

def test_all_null_numeric_column():
    df = pd.DataFrame({
        "all_null": [np.nan] * 50,
        "normal": np.random.default_rng(0).normal(0, 1, 50),
    })
    eda = EDAResult(df)
    u = eda.univariate
    # Should not raise; missing count should equal n
    s = u.stats("all_null")
    assert s["missing"] == 50


# ---------------------------------------------------------------------------
# Edge case: single-value column
# ---------------------------------------------------------------------------

def test_single_value_categorical_column():
    # A column with one unique value is classified as CONSTANT by infer_column_type,
    # so it stores only {'type': 'constant'} — no n_unique or top_value.
    df = pd.DataFrame({
        "constant": ["x"] * 60,
        "num": np.random.default_rng(1).normal(0, 1, 60),
    })
    eda = EDAResult(df)
    u = eda.univariate
    s = u.stats("constant")
    assert s["type"] == "constant"


def test_single_value_numeric_column():
    # A column with one unique numeric value is classified as CONSTANT,
    # so it stores only {'type': 'constant'}.
    df = pd.DataFrame({
        "one_val": [5.0] * 40,
        "other": np.random.default_rng(2).normal(0, 1, 40),
    })
    eda = EDAResult(df)
    u = eda.univariate
    s = u.stats("one_val")
    assert s["type"] == "constant"
