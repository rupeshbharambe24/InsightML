"""Tests for FeatureInteractions."""
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
def interactions(eda_result):
    return eda_result.interactions


# ---------------------------------------------------------------------------
# Basic type checks
# ---------------------------------------------------------------------------

def test_interactions_returns_object(eda_result):
    from insightml.eda.interactions import FeatureInteractions
    assert isinstance(eda_result.interactions, FeatureInteractions)


def test_interactions_is_cached(eda_result):
    assert eda_result.interactions is eda_result.interactions


# ---------------------------------------------------------------------------
# strengths()
# ---------------------------------------------------------------------------

def test_strengths_returns_dataframe(interactions):
    df = interactions.strengths()
    assert isinstance(df, pd.DataFrame)


def test_strengths_not_empty(interactions):
    df = interactions.strengths()
    assert not df.empty


def test_strengths_has_col_a_col_b(interactions):
    df = interactions.strengths()
    assert "col_a" in df.columns
    assert "col_b" in df.columns


def test_strengths_has_interaction_strength_column(interactions):
    df = interactions.strengths()
    assert "interaction_strength" in df.columns


def test_strengths_col_a_col_b_are_numeric_features(interactions):
    """col_a and col_b should only reference numeric non-target columns."""
    df = interactions.strengths()
    numeric_feature_cols = {"num1", "num2"}
    all_cols_in_pairs = set(df["col_a"].tolist()) | set(df["col_b"].tolist())
    assert all_cols_in_pairs.issubset(numeric_feature_cols)


def test_strengths_no_duplicate_pairs(interactions):
    """Each unordered pair should appear at most once."""
    df = interactions.strengths()
    pair_set = set()
    for _, row in df.iterrows():
        pair = frozenset([row["col_a"], row["col_b"]])
        assert pair not in pair_set, f"Duplicate pair: {row['col_a']}, {row['col_b']}"
        pair_set.add(pair)


# ---------------------------------------------------------------------------
# top_interactions()
# ---------------------------------------------------------------------------

def test_top_interactions_returns_dataframe(interactions):
    df = interactions.top_interactions(n=5)
    assert isinstance(df, pd.DataFrame)


def test_top_interactions_respects_n(interactions):
    full = interactions.strengths()
    top5 = interactions.top_interactions(n=5)
    assert len(top5) <= min(5, len(full))


def test_top_interactions_default_n(interactions):
    top = interactions.top_interactions()
    full = interactions.strengths()
    assert len(top) <= min(10, len(full))


def test_top_interactions_is_head_of_strengths(interactions):
    """top_interactions(n) should equal the first n rows of strengths()."""
    n = 1
    top = interactions.top_interactions(n=n)
    full = interactions.strengths()
    pd.testing.assert_frame_equal(top.reset_index(drop=True),
                                  full.head(n).reset_index(drop=True))


# ---------------------------------------------------------------------------
# nonlinear_pairs()
# ---------------------------------------------------------------------------

def test_nonlinear_pairs_returns_dataframe(interactions):
    df = interactions.nonlinear_pairs()
    assert isinstance(df, pd.DataFrame)


def test_nonlinear_pairs_is_subset_of_strengths(interactions):
    """Every row in nonlinear_pairs() should have is_nonlinear=True."""
    df = interactions.nonlinear_pairs()
    if not df.empty:
        assert "is_nonlinear" in df.columns
        assert df["is_nonlinear"].all()


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

def test_summary_returns_string(interactions):
    s = interactions.summary()
    assert isinstance(s, str)
    assert len(s) > 0


def test_summary_mentions_pairs(interactions):
    s = interactions.summary()
    assert "pair" in s.lower() or "feature" in s.lower() or "interaction" in s.lower()


# ---------------------------------------------------------------------------
# interaction_plot()
# ---------------------------------------------------------------------------

def test_interaction_plot_returns_figure(interactions):
    fig = interactions.interaction_plot("num1", "num2")
    assert isinstance(fig, go.Figure)


def test_interaction_plot_has_scatter3d_trace(interactions):
    fig = interactions.interaction_plot("num1", "num2")
    trace_types = [type(t).__name__ for t in fig.data]
    assert "Scatter3d" in trace_types


def test_interaction_plot_title_contains_column_names(interactions):
    fig = interactions.interaction_plot("num1", "num2")
    title_text = fig.layout.title.text or ""
    assert "num1" in title_text
    assert "num2" in title_text


# ---------------------------------------------------------------------------
# Edge case: fewer than 2 numeric columns
# ---------------------------------------------------------------------------

def test_skipped_when_one_numeric_column():
    """With only one numeric feature, interactions should be gracefully skipped."""
    df = pd.DataFrame({
        "num1": np.random.default_rng(7).normal(0, 1, 50),
        "cat1": ["a", "b"] * 25,
        "target": np.random.default_rng(7).choice([0, 1], 50),
    })
    eda = EDAResult(df, target="target")
    interactions = eda.interactions
    s = interactions.summary()
    assert isinstance(s, str)
    # strengths() should return empty DataFrame (no pairs)
    result = interactions.strengths()
    assert isinstance(result, pd.DataFrame)
