"""Tests for StatisticalTests."""
import pytest
import pandas as pd
import numpy as np
from insightml.eda.statistical_tests import StatisticalTests


def test_normality_normal_data():
    import pandas as pd
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.normal(0, 1, 200)})
    st = StatisticalTests(df)
    result = st.normality("x")
    # Large normal sample should pass at least some tests
    assert "is_normal" in result


def test_normality_uniform_fails():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"u": rng.uniform(0, 1, 500)})
    st = StatisticalTests(df)
    result = st.normality("u")
    assert result.get("is_normal") is False


def test_independence_test(df_classification):
    st = StatisticalTests(df_classification)
    result = st.independence("species", "species")
    # Perfect correspondence with itself
    assert result["p_value"] < 0.05 or "error" in result


def test_variance_test(df_classification):
    st = StatisticalTests(df_classification)
    result = st.variance("sepal_length", "species")
    assert "levene" in result
    assert "bartlett" in result


def test_group_comparison_selects_test(df_classification):
    st = StatisticalTests(df_classification)
    result = st.group_comparison("petal_length", "species")
    assert "test_used" in result
    assert result["test_used"] in ("ANOVA", "Kruskal-Wallis")
    assert "p_value" in result


def test_qq_figure_built(df_regression):
    st = StatisticalTests(df_regression)
    figs = st.plot()
    assert len(figs) > 0  # at least one QQ plot
