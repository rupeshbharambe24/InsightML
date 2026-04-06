"""Tests for compare/comparator.py — ModelComparator (new functionality only).

Tests in test_error_analysis.py already cover:
  - comp.table is a ComparisonTable
  - comp.pareto (go.Figure)
  - comp.metric_bar (go.Figure)
  - comp.roc_curves for classification
  - comp.residual_plots for regression
  - comp.significance has "ttest" key
  - comp.significance has "mcnemar" key for classification
  - comp.error_analysis
  - repr(comp) contains "ModelComparator"

This file covers DIFFERENT functionality:
  - comp.table is specifically a ComparisonTable instance (type-checked)
  - comp.metric_bar type guard
  - comp.pareto_models is a list of strings
  - significance dict structure for regression (ttest only, no mcnemar)
  - comp.significance["ttest"] key details
  - comp.table.dataframe() models match result.successful
  - comp.roc_curves for classification is go.Figure (via comparator property)
  - comp.residual_plots for regression is go.Figure (via comparator property)
  - repr(comp) format details
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from insightml.compare.comparator import ModelComparator
from insightml.compare.metrics_table import ComparisonTable


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_clf_result(n=100):
    from insightml.battle.result import BattleResult, ModelScore

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
    from insightml.battle.result import BattleResult, ModelScore

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

class TestModelComparatorTable:
    def test_table_is_comparison_table_instance(self):
        result, y = _make_clf_result()
        comp = ModelComparator(result, y=y)
        assert isinstance(comp.table, ComparisonTable)

    def test_table_dataframe_has_same_models_as_successful(self):
        result, y = _make_clf_result()
        comp = ModelComparator(result, y=y)
        df = comp.table.dataframe()
        successful_names = {s.name for s in result.successful}
        df_names = set(df["model"].tolist())
        assert df_names == successful_names

    def test_table_dataframe_row_count_equals_successful(self):
        result, y = _make_clf_result()
        comp = ModelComparator(result, y=y)
        df = comp.table.dataframe()
        assert len(df) == len(result.successful)

    def test_table_cached_property_returns_same_object(self):
        result, y = _make_clf_result()
        comp = ModelComparator(result, y=y)
        t1 = comp.table
        t2 = comp.table
        assert t1 is t2


class TestModelComparatorMetricBar:
    def test_metric_bar_is_figure(self):
        result, _ = _make_clf_result()
        comp = ModelComparator(result)
        assert isinstance(comp.metric_bar, go.Figure)

    def test_metric_bar_has_traces(self):
        result, _ = _make_clf_result()
        comp = ModelComparator(result)
        assert len(comp.metric_bar.data) > 0


class TestModelComparatorParetoModels:
    def test_pareto_models_is_list(self):
        result, _ = _make_clf_result()
        comp = ModelComparator(result)
        assert isinstance(comp.pareto_models, list)

    def test_pareto_models_are_strings(self):
        result, _ = _make_clf_result()
        comp = ModelComparator(result)
        assert all(isinstance(m, str) for m in comp.pareto_models)

    def test_pareto_models_subset_of_successful(self):
        result, _ = _make_clf_result()
        comp = ModelComparator(result)
        successful_names = {s.name for s in result.successful}
        assert set(comp.pareto_models).issubset(successful_names)

    def test_pareto_models_nonempty_when_models_exist(self):
        result, _ = _make_clf_result()
        comp = ModelComparator(result)
        assert len(comp.pareto_models) >= 1


class TestModelComparatorSignificanceRegression:
    def test_significance_has_ttest_for_regression(self):
        result, y = _make_reg_result()
        comp = ModelComparator(result, y=y)
        sig = comp.significance
        assert "ttest" in sig

    def test_significance_ttest_has_p_matrix(self):
        result, y = _make_reg_result()
        comp = ModelComparator(result, y=y)
        sig = comp.significance
        assert "p_matrix" in sig["ttest"]

    def test_significance_ttest_has_figure(self):
        result, y = _make_reg_result()
        comp = ModelComparator(result, y=y)
        sig = comp.significance
        assert "figure" in sig["ttest"]

    def test_significance_no_mcnemar_for_regression(self):
        result, y = _make_reg_result()
        comp = ModelComparator(result, y=y)
        sig = comp.significance
        assert "mcnemar" not in sig

    def test_significance_empty_when_no_y(self):
        result, _ = _make_reg_result()
        comp = ModelComparator(result)
        sig = comp.significance
        assert sig == {}

    def test_ttest_p_matrix_shape(self):
        result, y = _make_reg_result()
        comp = ModelComparator(result, y=y)
        p = comp.significance["ttest"]["p_matrix"]
        assert p.shape == (2, 2)


class TestModelComparatorCurves:
    def test_roc_curves_is_figure_for_classification(self):
        result, y = _make_clf_result()
        comp = ModelComparator(result, y=y)
        assert isinstance(comp.roc_curves, go.Figure)

    def test_roc_curves_is_none_for_regression(self):
        result, y = _make_reg_result()
        comp = ModelComparator(result, y=y)
        assert comp.roc_curves is None

    def test_residual_plots_is_figure_for_regression(self):
        result, y = _make_reg_result()
        comp = ModelComparator(result, y=y)
        assert isinstance(comp.residual_plots, go.Figure)

    def test_residual_plots_is_none_for_classification(self):
        result, y = _make_clf_result()
        comp = ModelComparator(result, y=y)
        assert comp.residual_plots is None


class TestModelComparatorRepr:
    def test_repr_contains_model_comparator(self):
        result, y = _make_clf_result()
        comp = ModelComparator(result, y=y)
        r = repr(comp)
        assert "ModelComparator" in r

    def test_repr_contains_task(self):
        result, y = _make_clf_result()
        comp = ModelComparator(result, y=y)
        r = repr(comp)
        assert "classification" in r

    def test_repr_contains_model_count(self):
        result, y = _make_clf_result()
        comp = ModelComparator(result, y=y)
        r = repr(comp)
        # The repr format is ModelComparator(task=..., models=N, pareto=[...])
        assert "models=" in r

    def test_repr_html_contains_model_comparator_heading(self):
        result, y = _make_clf_result()
        comp = ModelComparator(result, y=y)
        html = comp._repr_html_()
        assert "ModelComparator" in html
