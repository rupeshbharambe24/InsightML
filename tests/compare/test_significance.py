"""Tests for compare/significance.py and compare/pareto.py."""

import numpy as np
import pandas as pd

from insightml.compare.pareto import _compute_pareto, get_pareto_models, pareto_front
from insightml.compare.significance import corrected_ttest_matrix, mcnemar_matrix

# ---------------------------------------------------------------------------
# Shared fixtures — build a minimal BattleResult with OOF predictions
# ---------------------------------------------------------------------------

def _make_clf_result(n=200):
    """Build a minimal classification BattleResult for testing."""
    from insightml.battle.result import BattleResult, ModelScore

    rng = np.random.default_rng(99)
    y = rng.choice([0, 1], n)

    # Perfect model
    oof_perfect = y.copy().astype(float)
    probs_perfect = np.column_stack([1 - oof_perfect, oof_perfect])

    # Random model
    oof_random = rng.choice([0, 1], n).astype(float)
    probs_random = rng.dirichlet([1, 1], n)

    s1 = ModelScore(
        name="PerfectModel", task="classification",
        metrics={"accuracy": 1.0}, metrics_std={"accuracy": 0.0},
        oof_predictions=oof_perfect, oof_probabilities=probs_perfect,
        train_time=1.0,
    )
    s2 = ModelScore(
        name="RandomModel", task="classification",
        metrics={"accuracy": 0.5}, metrics_std={"accuracy": 0.05},
        oof_predictions=oof_random, oof_probabilities=probs_random,
        train_time=0.5,
    )
    return BattleResult(
        task="classification", scores=[s1, s2],
        primary_metric="accuracy", cv_folds=5, n_samples=n,
    ), pd.Series(y)


def _make_reg_result(n=200):
    from insightml.battle.result import BattleResult, ModelScore

    rng = np.random.default_rng(77)
    y = rng.normal(0, 1, n)

    oof_good = y + rng.normal(0, 0.1, n)
    oof_bad = rng.normal(0, 1, n)

    s1 = ModelScore(
        name="GoodRegressor", task="regression",
        metrics={"r2": 0.95}, metrics_std={"r2": 0.01},
        oof_predictions=oof_good, train_time=2.0,
    )
    s2 = ModelScore(
        name="BadRegressor", task="regression",
        metrics={"r2": 0.10}, metrics_std={"r2": 0.05},
        oof_predictions=oof_bad, train_time=0.5,
    )
    return BattleResult(
        task="regression", scores=[s1, s2],
        primary_metric="r2", cv_folds=5, n_samples=n,
    ), pd.Series(y)


class TestMcNemar:
    def test_returns_dict(self):
        result, y = _make_clf_result()
        out = mcnemar_matrix(result, y)
        assert "p_matrix" in out
        assert "figure" in out

    def test_p_matrix_shape(self):
        result, y = _make_clf_result()
        out = mcnemar_matrix(result, y)
        p = out["p_matrix"]
        assert p.shape == (2, 2)

    def test_diagonal_is_one(self):
        result, y = _make_clf_result()
        out = mcnemar_matrix(result, y)
        p = out["p_matrix"]
        assert all(p.loc[m, m] == 1.0 for m in p.index)

    def test_significant_pair_detected(self):
        """Perfect vs random should give very low p-value."""
        result, y = _make_clf_result(n=500)
        out = mcnemar_matrix(result, y)
        p = out["p_matrix"]
        assert float(p.loc["PerfectModel", "RandomModel"]) < 0.05

    def test_figure_returned(self):
        import plotly.graph_objects as go
        result, y = _make_clf_result()
        out = mcnemar_matrix(result, y)
        assert isinstance(out["figure"], go.Figure)


class TestCorrectedTTest:
    def test_returns_dict(self):
        result, y = _make_reg_result()
        out = corrected_ttest_matrix(result, y)
        assert "p_matrix" in out
        assert "figure" in out

    def test_symmetric(self):
        result, y = _make_reg_result()
        out = corrected_ttest_matrix(result, y)
        p = out["p_matrix"]
        assert abs(float(p.iloc[0, 1]) - float(p.iloc[1, 0])) < 1e-9

    def test_classification_also_works(self):
        result, y = _make_clf_result()
        out = corrected_ttest_matrix(result, y)
        assert "p_matrix" in out
        assert out["p_matrix"].shape == (2, 2)


class TestParetoFront:
    def test_compute_pareto_simple(self):
        # Model A: high accuracy, slow  → Pareto
        # Model B: low accuracy, fast   → Pareto
        # Model C: low accuracy, slow   → dominated
        metrics = [0.9, 0.6, 0.5]
        times   = [10.0, 1.0, 8.0]
        mask = _compute_pareto(metrics, times)
        assert mask[0] is True   # A: Pareto
        assert mask[1] is True   # B: Pareto
        assert mask[2] is False  # C: dominated by A on metric AND by B on time

    def test_all_pareto_if_tradeoff(self):
        # Each model uniquely better on one dimension
        metrics = [1.0, 0.5]
        times   = [10.0, 1.0]
        mask = _compute_pareto(metrics, times)
        assert all(mask)

    def test_get_pareto_models(self):
        result, _ = _make_clf_result()
        pareto = get_pareto_models(result)
        assert isinstance(pareto, list)
        assert len(pareto) >= 1

    def test_pareto_figure(self):
        import plotly.graph_objects as go
        result, _ = _make_clf_result()
        fig = pareto_front(result)
        assert isinstance(fig, go.Figure)
