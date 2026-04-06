"""Tests for CorrelationAnalysis."""
import pytest
import pandas as pd
import numpy as np
from insightml.eda.correlations import CorrelationAnalysis


def test_pearson_matrix_shape(df_regression):
    ca = CorrelationAnalysis(df_regression)
    pearson = ca.pearson()
    num_cols = df_regression.select_dtypes(include=[np.number]).columns
    assert pearson.shape == (len(num_cols), len(num_cols))


def test_diagonal_is_one(df_regression):
    ca = CorrelationAnalysis(df_regression)
    pearson = ca.pearson()
    np.testing.assert_allclose(np.diag(pearson.values), 1.0, atol=1e-10)


def test_unified_matrix_built(df_classification):
    ca = CorrelationAnalysis(df_classification)
    unified = ca.unified()
    assert not unified.empty
    assert unified.shape[0] == unified.shape[1]


def test_top_correlations_sorted(df_regression):
    ca = CorrelationAnalysis(df_regression)
    top = ca.top_correlations(n=5)
    assert len(top) <= 5
    if len(top) > 1:
        abs_vals = top["abs_value"].tolist()
        assert abs_vals == sorted(abs_vals, reverse=True)


def test_heatmap_returns_figure(df_regression):
    import plotly.graph_objects as go
    ca = CorrelationAnalysis(df_regression)
    fig = ca.heatmap("pearson")
    assert isinstance(fig, go.Figure)


def test_cramers_v_for_categorical(df_classification):
    ca = CorrelationAnalysis(df_classification)
    cv = ca.cramers_v_matrix()
    # Only one categorical column (species), so trivial matrix
    assert cv is not None
