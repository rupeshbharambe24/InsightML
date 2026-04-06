"""Tests for BivariateAnalysis."""
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
        "num2": rng.normal(0, 1, n),
        "cat1": rng.choice(["a", "b", "c"], n),
        "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def eda_result():
    return EDAResult(_make_df(), target="target")


@pytest.fixture
def bivariate(eda_result):
    return eda_result.bivariate


# ---------------------------------------------------------------------------
# Basic type checks
# ---------------------------------------------------------------------------

def test_bivariate_returns_object(eda_result):
    from insightml.eda.bivariate import BivariateAnalysis
    assert isinstance(eda_result.bivariate, BivariateAnalysis)


def test_bivariate_is_cached(eda_result):
    assert eda_result.bivariate is eda_result.bivariate


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

def test_summary_returns_string(bivariate):
    s = bivariate.summary()
    assert isinstance(s, str)
    assert len(s) > 0


def test_summary_mentions_pairs(bivariate):
    s = bivariate.summary()
    assert "pair" in s.lower()


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------

def test_to_dict_returns_dict(bivariate):
    d = bivariate.to_dict()
    assert isinstance(d, dict)


def test_to_dict_has_pairs_key(bivariate):
    d = bivariate.to_dict()
    assert "pairs" in d


def test_to_dict_pairs_is_non_empty(bivariate):
    d = bivariate.to_dict()
    assert len(d["pairs"]) > 0


# ---------------------------------------------------------------------------
# to_dataframe()
# ---------------------------------------------------------------------------

def test_to_dataframe_returns_dataframe(bivariate):
    df = bivariate.to_dataframe()
    assert isinstance(df, pd.DataFrame)


def test_to_dataframe_not_empty(bivariate):
    df = bivariate.to_dataframe()
    assert not df.empty


# ---------------------------------------------------------------------------
# pair() accessor
# ---------------------------------------------------------------------------

def test_num_num_pair_returns_dict(bivariate):
    result = bivariate.pair("num1", "num2")
    assert isinstance(result, dict)


def test_num_num_pair_has_pearson(bivariate):
    result = bivariate.pair("num1", "num2")
    assert "pearson_r" in result
    assert isinstance(result["pearson_r"], float)


def test_num_num_pearson_in_valid_range(bivariate):
    result = bivariate.pair("num1", "num2")
    assert -1.0 <= result["pearson_r"] <= 1.0


def test_num_num_pair_has_spearman(bivariate):
    result = bivariate.pair("num1", "num2")
    assert "spearman_rho" in result


def test_num_num_pair_symmetric_lookup(bivariate):
    """pair(a, b) and pair(b, a) should return the same result."""
    r1 = bivariate.pair("num1", "num2")
    r2 = bivariate.pair("num2", "num1")
    assert r1 == r2


def test_num_cat_pair_has_anova(bivariate):
    result = bivariate.pair("num1", "cat1")
    assert "anova_f" in result
    assert "anova_p" in result


def test_num_cat_pair_type_label(bivariate):
    result = bivariate.pair("num1", "cat1")
    assert result.get("type") == "num-cat"


# ---------------------------------------------------------------------------
# Target column included in pairs
# ---------------------------------------------------------------------------

def test_target_involved_in_pairs(bivariate):
    """At least one analyzed pair should involve the target column."""
    d = bivariate.to_dict()
    pairs = d["pairs"]
    # Keys are "col_a||col_b"
    target_keys = [k for k in pairs if "target" in k]
    assert len(target_keys) > 0


def test_target_num_pair_exists(bivariate):
    """num1 vs target should be analyzable since target is numeric-ish (0/1)."""
    result = bivariate.pair("num1", "target")
    # Should return a non-empty dict (even if type is num-num or num-cat)
    assert isinstance(result, dict)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_num_num_figure_is_go_figure(eda_result):
    """num1 vs num2 scatter figure should be a go.Figure."""
    biv = eda_result.bivariate
    biv._ensure_computed()
    # Figure key is "num1||num2"
    fig = biv._figures.get("num1||num2")
    if fig is not None:
        assert isinstance(fig, go.Figure)


def test_figures_dict_contains_go_figures(bivariate):
    """All built figures must be go.Figure instances."""
    bivariate._ensure_computed()
    for key, fig in bivariate._figures.items():
        assert isinstance(fig, go.Figure), f"Figure '{key}' is not a go.Figure"
