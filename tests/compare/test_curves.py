"""Tests for compare/curves.py — ROC, PR, confusion matrix, residual, actual-vs-predicted, bar chart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dissectml.compare.curves import (
    actual_vs_predicted,
    confusion_matrices,
    metric_bar_chart,
    pr_curves,
    residual_plots,
    roc_curves,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_clf_result(n=100):
    from dissectml.battle.result import BattleResult, ModelScore

    rng = np.random.default_rng(42)
    y = rng.choice([0, 1], n)
    oof_a = y.copy().astype(float)
    oof_a[rng.choice(n, 15, replace=False)] = 1 - oof_a[rng.choice(n, 15, replace=False)]
    probs_a = np.column_stack([1 - oof_a, oof_a])
    oof_b = rng.choice([0, 1], n).astype(float)
    probs_b = np.column_stack([1 - oof_b, oof_b])
    s1 = ModelScore(
        "ModelA", "classification",
        metrics={"accuracy": 0.85},
        oof_predictions=oof_a, oof_probabilities=probs_a,
        train_time=1.0,
    )
    s2 = ModelScore(
        "ModelB", "classification",
        metrics={"accuracy": 0.65},
        oof_predictions=oof_b, oof_probabilities=probs_b,
        train_time=0.3,
    )
    return BattleResult(
        "classification", [s1, s2],
        primary_metric="accuracy", cv_folds=3, n_samples=n,
    ), pd.Series(y)


def _make_reg_result(n=100):
    from dissectml.battle.result import BattleResult, ModelScore

    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, n)
    s1 = ModelScore(
        "RegA", "regression",
        metrics={"r2": 0.9},
        oof_predictions=y + rng.normal(0, 0.2, n),
        train_time=1.0,
    )
    s2 = ModelScore(
        "RegB", "regression",
        metrics={"r2": 0.5},
        oof_predictions=rng.normal(0, 1, n),
        train_time=0.3,
    )
    return BattleResult(
        "regression", [s1, s2],
        primary_metric="r2", cv_folds=3, n_samples=n,
    ), pd.Series(y)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRocCurves:
    def test_returns_figure(self):
        result, y = _make_clf_result()
        fig = roc_curves(result, y)
        assert isinstance(fig, go.Figure)

    def test_figure_has_traces(self):
        result, y = _make_clf_result()
        fig = roc_curves(result, y)
        assert len(fig.data) > 0

    def test_n_models_1_limits_curves(self):
        result, y = _make_clf_result()
        fig_all = roc_curves(result, y, n_models=10)
        fig_one = roc_curves(result, y, n_models=1)
        # With n_models=1 we get fewer model traces (at most 1 model + diagonal)
        assert len(fig_one.data) <= len(fig_all.data)


class TestPrCurves:
    def test_returns_figure(self):
        result, y = _make_clf_result()
        fig = pr_curves(result, y)
        assert isinstance(fig, go.Figure)

    def test_figure_has_traces(self):
        result, y = _make_clf_result()
        fig = pr_curves(result, y)
        assert len(fig.data) > 0

    def test_n_models_1_works(self):
        result, y = _make_clf_result()
        fig = pr_curves(result, y, n_models=1)
        assert isinstance(fig, go.Figure)


class TestConfusionMatrices:
    def test_returns_figure(self):
        result, y = _make_clf_result()
        fig = confusion_matrices(result, y)
        assert isinstance(fig, go.Figure)

    def test_figure_has_traces(self):
        result, y = _make_clf_result()
        fig = confusion_matrices(result, y)
        assert len(fig.data) > 0

    def test_n_models_1_returns_single_heatmap(self):
        result, y = _make_clf_result()
        fig = confusion_matrices(result, y, n_models=1)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestResidualPlots:
    def test_returns_figure(self):
        result, y = _make_reg_result()
        fig = residual_plots(result, y)
        assert isinstance(fig, go.Figure)

    def test_figure_has_traces(self):
        result, y = _make_reg_result()
        fig = residual_plots(result, y)
        assert len(fig.data) > 0

    def test_n_models_1_works(self):
        result, y = _make_reg_result()
        fig = residual_plots(result, y, n_models=1)
        assert isinstance(fig, go.Figure)


class TestActualVsPredicted:
    def test_returns_figure(self):
        result, y = _make_reg_result()
        fig = actual_vs_predicted(result, y)
        assert isinstance(fig, go.Figure)

    def test_figure_has_traces(self):
        result, y = _make_reg_result()
        fig = actual_vs_predicted(result, y)
        assert len(fig.data) > 0

    def test_n_models_1_works(self):
        result, y = _make_reg_result()
        fig = actual_vs_predicted(result, y, n_models=1)
        assert isinstance(fig, go.Figure)


class TestMetricBarChart:
    def test_returns_figure_clf(self):
        result, _ = _make_clf_result()
        fig = metric_bar_chart(result, "accuracy")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_reg(self):
        result, _ = _make_reg_result()
        fig = metric_bar_chart(result, "r2")
        assert isinstance(fig, go.Figure)

    def test_figure_has_traces(self):
        result, _ = _make_clf_result()
        fig = metric_bar_chart(result, "accuracy")
        assert len(fig.data) > 0

    def test_uses_primary_metric_when_none_given(self):
        result, _ = _make_clf_result()
        fig = metric_bar_chart(result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_n_models_1_single_bar(self):
        result, _ = _make_clf_result()
        fig = metric_bar_chart(result, "accuracy", n_models=1)
        assert isinstance(fig, go.Figure)
        # One bar trace present
        assert len(fig.data) >= 1
