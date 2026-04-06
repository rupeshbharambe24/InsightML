"""Tests for compare/error_analysis.py and compare/comparator.py."""

import numpy as np
import pandas as pd

from insightml.compare.comparator import ModelComparator
from insightml.compare.error_analysis import ErrorAnalysisResult, analyze_errors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_battle_result(n=200, task="classification"):
    from insightml.battle.result import BattleResult, ModelScore
    rng = np.random.default_rng(42)

    if task == "classification":
        y = rng.choice([0, 1], n)
        # Model A: mostly correct
        oof_a = y.copy().astype(float)
        idx_wrong = rng.choice(n, 30, replace=False)
        oof_a[idx_wrong] = 1 - oof_a[idx_wrong]
        # Model B: different errors
        oof_b = y.copy().astype(float)
        idx_wrong_b = rng.choice(n, 50, replace=False)
        oof_b[idx_wrong_b] = 1 - oof_b[idx_wrong_b]

        probs_a = np.column_stack([1 - oof_a, oof_a])
        probs_b = np.column_stack([1 - oof_b, oof_b])

        s1 = ModelScore("ModelA", "classification",
                        metrics={"accuracy": 0.85}, metrics_std={"accuracy": 0.02},
                        oof_predictions=oof_a, oof_probabilities=probs_a, train_time=1.0)
        s2 = ModelScore("ModelB", "classification",
                        metrics={"accuracy": 0.75}, metrics_std={"accuracy": 0.03},
                        oof_predictions=oof_b, oof_probabilities=probs_b, train_time=0.5)
        result = BattleResult(task="classification", scores=[s1, s2],
                              primary_metric="accuracy", cv_folds=5, n_samples=n)
        return result, pd.Series(y)
    else:
        y = rng.normal(0, 1, n)
        oof_a = y + rng.normal(0, 0.2, n)
        oof_b = y + rng.normal(0, 0.5, n)
        s1 = ModelScore("ModelA", "regression",
                        metrics={"r2": 0.90}, oof_predictions=oof_a, train_time=1.5)
        s2 = ModelScore("ModelB", "regression",
                        metrics={"r2": 0.70}, oof_predictions=oof_b, train_time=0.3)
        result = BattleResult(task="regression", scores=[s1, s2],
                              primary_metric="r2", cv_folds=5, n_samples=n)
        return result, pd.Series(y)


class TestAnalyzeErrors:
    def test_returns_result(self):
        result, y = _make_battle_result()
        ea = analyze_errors(result, y)
        assert isinstance(ea, ErrorAnalysisResult)

    def test_disagreement_shape(self):
        result, y = _make_battle_result()
        ea = analyze_errors(result, y)
        assert ea.disagreement.shape == (2, 2)

    def test_disagreement_diagonal_zero(self):
        result, y = _make_battle_result()
        ea = analyze_errors(result, y)
        for m in ea.models:
            assert ea.disagreement.loc[m, m] == 0.0

    def test_hard_indices_nonempty(self):
        result, y = _make_battle_result()
        ea = analyze_errors(result, y)
        assert len(ea.hard_indices) > 0

    def test_hard_sample_profile_with_X(self):
        result, y = _make_battle_result()
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"f1": rng.normal(0, 1, len(y)), "f2": rng.normal(0, 1, len(y))})
        ea = analyze_errors(result, y, X=X)
        assert isinstance(ea.hard_sample_profile, pd.DataFrame)
        if not ea.hard_sample_profile.empty:
            assert "feature" in ea.hard_sample_profile.columns

    def test_regression_also_works(self):
        result, y = _make_battle_result(task="regression")
        ea = analyze_errors(result, y)
        assert isinstance(ea, ErrorAnalysisResult)
        assert len(ea.models) == 2

    def test_ensemble_candidates(self):
        result, y = _make_battle_result()
        ea = analyze_errors(result, y)
        candidates = ea.ensemble_candidates()
        assert isinstance(candidates, list)

    def test_disagreement_figure(self):
        import plotly.graph_objects as go
        result, y = _make_battle_result()
        ea = analyze_errors(result, y)
        fig = ea.disagreement_figure()
        assert isinstance(fig, go.Figure)

    def test_hard_sample_figure(self):
        import plotly.graph_objects as go
        result, y = _make_battle_result()
        ea = analyze_errors(result, y)
        fig = ea.hard_sample_figure()
        assert isinstance(fig, go.Figure)

    def test_summary_string(self):
        result, y = _make_battle_result()
        ea = analyze_errors(result, y)
        s = ea.summary()
        assert "ErrorAnalysis" in s


class TestModelComparator:
    def test_table(self):
        from insightml.compare.metrics_table import ComparisonTable
        result, y = _make_battle_result()
        comp = ModelComparator(result, y=y)
        assert isinstance(comp.table, ComparisonTable)

    def test_pareto_figure(self):
        import plotly.graph_objects as go
        result, y = _make_battle_result()
        comp = ModelComparator(result)
        assert isinstance(comp.pareto, go.Figure)

    def test_metric_bar(self):
        import plotly.graph_objects as go
        result, y = _make_battle_result()
        comp = ModelComparator(result)
        assert isinstance(comp.metric_bar, go.Figure)

    def test_roc_curves(self):
        import plotly.graph_objects as go
        result, y = _make_battle_result(task="classification")
        comp = ModelComparator(result, y=y)
        fig = comp.roc_curves
        assert isinstance(fig, go.Figure)

    def test_residual_plots(self):
        import plotly.graph_objects as go
        result, y = _make_battle_result(task="regression")
        comp = ModelComparator(result, y=y)
        fig = comp.residual_plots
        assert isinstance(fig, go.Figure)

    def test_significance_has_ttest(self):
        result, y = _make_battle_result()
        comp = ModelComparator(result, y=y)
        sig = comp.significance
        assert "ttest" in sig
        assert "p_matrix" in sig["ttest"]

    def test_significance_has_mcnemar_for_clf(self):
        result, y = _make_battle_result(task="classification")
        comp = ModelComparator(result, y=y)
        sig = comp.significance
        assert "mcnemar" in sig

    def test_error_analysis(self):
        result, y = _make_battle_result()
        comp = ModelComparator(result, y=y)
        ea = comp.error_analysis
        assert isinstance(ea, ErrorAnalysisResult)

    def test_repr(self):
        result, y = _make_battle_result()
        comp = ModelComparator(result, y=y)
        r = repr(comp)
        assert "ModelComparator" in r
