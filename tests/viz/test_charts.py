"""Tests for viz.charts — chart factory functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from insightml.viz.charts import (
    box_plot,
    frequency_bar,
    gauge,
    heatmap,
    histogram,
    scatter,
    violin,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_series():
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(10, 3, 50), name="values")


@pytest.fixture
def categorical_series():
    rng = np.random.default_rng(42)
    return pd.Series(rng.choice(["A", "B", "C", "D"], 50), name="category")


@pytest.fixture
def corr_matrix():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (50, 4))
    df = pd.DataFrame(data, columns=["a", "b", "c", "d"])
    return df.corr()


@pytest.fixture
def two_numeric_series():
    rng = np.random.default_rng(42)
    x = pd.Series(rng.normal(0, 1, 50), name="x")
    y = pd.Series(rng.normal(0, 1, 50), name="y")
    return x, y


@pytest.fixture
def groups_series():
    rng = np.random.default_rng(42)
    return pd.Series(rng.choice(["G1", "G2", "G3"], 50), name="group")


# ---------------------------------------------------------------------------
# histogram tests
# ---------------------------------------------------------------------------

class TestHistogram:
    def test_returns_figure(self, numeric_series):
        fig = histogram(numeric_series)
        assert isinstance(fig, go.Figure)

    def test_with_title(self, numeric_series):
        fig = histogram(numeric_series, title="My Histogram")
        assert isinstance(fig, go.Figure)

    def test_without_kde(self, numeric_series):
        fig = histogram(numeric_series, kde=False)
        assert isinstance(fig, go.Figure)

    def test_with_custom_color(self, numeric_series):
        fig = histogram(numeric_series, color="#ff0000")
        assert isinstance(fig, go.Figure)

    def test_small_series_no_kde(self):
        """KDE requires >= 5 data points; fewer should still work."""
        s = pd.Series([1.0, 2.0, 3.0], name="tiny")
        fig = histogram(s, kde=True)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# box_plot tests
# ---------------------------------------------------------------------------

class TestBoxPlot:
    def test_returns_figure(self, numeric_series):
        fig = box_plot(numeric_series)
        assert isinstance(fig, go.Figure)

    def test_with_title(self, numeric_series):
        fig = box_plot(numeric_series, title="Box Plot")
        assert isinstance(fig, go.Figure)

    def test_with_color(self, numeric_series):
        fig = box_plot(numeric_series, color="#00ff00")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# frequency_bar tests
# ---------------------------------------------------------------------------

class TestFrequencyBar:
    def test_returns_figure(self, categorical_series):
        fig = frequency_bar(categorical_series)
        assert isinstance(fig, go.Figure)

    def test_with_top_n(self, categorical_series):
        fig = frequency_bar(categorical_series, top_n=2)
        assert isinstance(fig, go.Figure)

    def test_with_title(self, categorical_series):
        fig = frequency_bar(categorical_series, title="Freq Bar")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# heatmap tests
# ---------------------------------------------------------------------------

class TestHeatmap:
    def test_returns_figure(self, corr_matrix):
        fig = heatmap(corr_matrix)
        assert isinstance(fig, go.Figure)

    def test_with_title(self, corr_matrix):
        fig = heatmap(corr_matrix, title="Corr Heatmap")
        assert isinstance(fig, go.Figure)

    def test_no_annotations(self, corr_matrix):
        fig = heatmap(corr_matrix, annotate=False)
        assert isinstance(fig, go.Figure)

    def test_custom_zmid(self, corr_matrix):
        fig = heatmap(corr_matrix, zmid=0.5)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# scatter tests
# ---------------------------------------------------------------------------

class TestScatter:
    def test_returns_figure(self, two_numeric_series):
        x, y = two_numeric_series
        fig = scatter(x, y)
        assert isinstance(fig, go.Figure)

    def test_with_title(self, two_numeric_series):
        x, y = two_numeric_series
        fig = scatter(x, y, title="Scatter Plot")
        assert isinstance(fig, go.Figure)

    def test_no_trendline(self, two_numeric_series):
        x, y = two_numeric_series
        fig = scatter(x, y, trendline=False)
        assert isinstance(fig, go.Figure)

    def test_color_by(self, two_numeric_series, groups_series):
        x, y = two_numeric_series
        fig = scatter(x, y, color_by=groups_series)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# violin tests
# ---------------------------------------------------------------------------

class TestViolin:
    def test_returns_figure(self, numeric_series, groups_series):
        fig = violin(numeric_series, groups_series)
        assert isinstance(fig, go.Figure)

    def test_with_title(self, numeric_series, groups_series):
        fig = violin(numeric_series, groups_series, title="Violin Plot")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# gauge tests
# ---------------------------------------------------------------------------

class TestGauge:
    def test_returns_figure(self):
        fig = gauge(75.0)
        assert isinstance(fig, go.Figure)

    def test_with_title(self):
        fig = gauge(85.0, title="Data Readiness")
        assert isinstance(fig, go.Figure)

    def test_custom_range(self):
        fig = gauge(0.5, min_val=0, max_val=1.0, title="AUC")
        assert isinstance(fig, go.Figure)

    def test_boundary_values(self):
        fig_low = gauge(0.0)
        fig_high = gauge(100.0)
        assert isinstance(fig_low, go.Figure)
        assert isinstance(fig_high, go.Figure)
