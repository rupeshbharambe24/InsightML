"""Tests for compare/pareto.py — additional coverage beyond test_significance.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dissectml.compare.pareto import _compute_pareto, get_pareto_models, pareto_front

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


def _make_single_model_result():
    from dissectml.battle.result import BattleResult, ModelScore

    rng = np.random.default_rng(7)
    y = rng.choice([0, 1], 50)
    oof = y.copy().astype(float)
    probs = np.column_stack([1 - oof, oof])
    s = ModelScore(
        "OnlyModel", "classification",
        metrics={"accuracy": 0.80},
        oof_predictions=oof, oof_probabilities=probs,
        train_time=0.5,
    )
    return BattleResult(
        "classification", [s],
        primary_metric="accuracy", cv_folds=3, n_samples=50,
    ), pd.Series(y)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParetoFrontFigure:
    def test_returns_figure(self):
        result, _ = _make_clf_result()
        fig = pareto_front(result)
        assert isinstance(fig, go.Figure)

    def test_figure_has_at_least_one_trace(self):
        result, _ = _make_clf_result()
        fig = pareto_front(result)
        assert len(fig.data) >= 1

    def test_figure_title_contains_pareto(self):
        result, _ = _make_clf_result()
        fig = pareto_front(result)
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert "Pareto" in title_text

    def test_single_model_still_returns_figure(self):
        result, _ = _make_single_model_result()
        fig = pareto_front(result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_regression_result_figure(self):
        result, _ = _make_reg_result()
        fig = pareto_front(result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_figure_title_contains_primary_metric(self):
        result, _ = _make_clf_result()
        fig = pareto_front(result)
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert result.primary_metric in title_text


class TestGetParetoModels:
    def test_returns_list(self):
        result, _ = _make_clf_result()
        pareto = get_pareto_models(result)
        assert isinstance(pareto, list)

    def test_all_names_are_strings(self):
        result, _ = _make_clf_result()
        pareto = get_pareto_models(result)
        assert all(isinstance(n, str) for n in pareto)

    def test_regression_result(self):
        result, _ = _make_reg_result()
        pareto = get_pareto_models(result)
        assert isinstance(pareto, list)
        assert len(pareto) >= 1

    def test_pareto_models_subset_of_successful(self):
        result, _ = _make_clf_result()
        pareto = get_pareto_models(result)
        successful_names = {s.name for s in result.successful}
        assert set(pareto).issubset(successful_names)

    def test_single_model_is_pareto(self):
        result, _ = _make_single_model_result()
        pareto = get_pareto_models(result)
        assert len(pareto) == 1
        assert pareto[0] == "OnlyModel"


class TestComputeParetoEqualSpeed:
    def test_all_models_same_time_pareto_by_metric(self):
        # All models equally fast — Pareto is determined by metric only
        # Only the best metric model should survive (others are dominated on metric)
        metrics = [0.9, 0.7, 0.5]
        times = [1.0, 1.0, 1.0]
        mask = _compute_pareto(metrics, times)
        # Model with 0.9 dominates all others (same time, better metric)
        assert mask[0] is True
        assert mask[1] is False
        assert mask[2] is False

    def test_tied_best_metric_same_time_both_pareto(self):
        # Two models with identical metric and time — neither dominates the other
        metrics = [0.9, 0.9]
        times = [1.0, 1.0]
        mask = _compute_pareto(metrics, times)
        assert mask[0] is True
        assert mask[1] is True

    def test_dominated_on_both_axes_not_pareto(self):
        metrics = [0.9, 0.7]
        times = [1.0, 2.0]
        mask = _compute_pareto(metrics, times)
        # model 1 (0.9, 1.0s) dominates model 2 (0.7, 2.0s)
        assert mask[0] is True
        assert mask[1] is False
