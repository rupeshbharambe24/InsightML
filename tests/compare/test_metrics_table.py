"""Tests for compare/metrics_table.py — ComparisonTable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dissectml.compare.metrics_table import ComparisonTable

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

class TestComparisonTable:
    def test_dataframe_returns_dataframe(self):
        result, _ = _make_clf_result()
        ct = ComparisonTable(result)
        df = ct.dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_model_names(self):
        result, _ = _make_clf_result()
        ct = ComparisonTable(result)
        df = ct.dataframe()
        assert "model" in df.columns
        model_names = df["model"].tolist()
        assert "ModelA" in model_names
        assert "ModelB" in model_names

    def test_dataframe_has_primary_metric_clf(self):
        result, _ = _make_clf_result()
        ct = ComparisonTable(result)
        df = ct.dataframe()
        assert "accuracy" in df.columns

    def test_dataframe_has_primary_metric_reg(self):
        result, _ = _make_reg_result()
        ct = ComparisonTable(result)
        df = ct.dataframe()
        assert "r2" in df.columns

    def test_dataframe_has_train_time_column(self):
        result, _ = _make_clf_result()
        ct = ComparisonTable(result)
        df = ct.dataframe()
        assert "train_time_s" in df.columns

    def test_include_std_adds_std_columns(self):
        result, _ = _make_clf_result()
        # Add metrics_std to scores so the column actually appears
        for score in result.scores:
            score.metrics_std = {"accuracy": 0.02}
        ct = ComparisonTable(result)
        df = ct.dataframe(include_std=True)
        assert any("_std" in col for col in df.columns)

    def test_include_std_false_no_std_columns(self):
        result, _ = _make_clf_result()
        ct = ComparisonTable(result)
        df = ct.dataframe(include_std=False)
        assert not any("_std" in col for col in df.columns)

    def test_to_latex_returns_string_with_tabular(self):
        result, _ = _make_clf_result()
        ct = ComparisonTable(result)
        latex = ct.to_latex()
        assert isinstance(latex, str)
        assert "tabular" in latex

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("tabulate") is None,
        reason="tabulate not installed",
    )
    def test_to_markdown_returns_string_with_pipes(self):
        result, _ = _make_clf_result()
        ct = ComparisonTable(result)
        md = ct.to_markdown()
        assert isinstance(md, str)
        assert "|" in md

    def test_repr_html_returns_string_with_table_tag(self):
        result, _ = _make_clf_result()
        ct = ComparisonTable(result)
        html = ct._repr_html_()
        assert isinstance(html, str)
        assert "<table" in html.lower()

    def test_repr_contains_comparison_table(self):
        result, _ = _make_clf_result()
        ct = ComparisonTable(result)
        r = repr(ct)
        assert "ComparisonTable" in r

    def test_dataframe_sorted_by_primary_metric_descending(self):
        result, _ = _make_clf_result()
        ct = ComparisonTable(result)
        df = ct.dataframe()
        vals = df["accuracy"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_dataframe_row_count_matches_successful_models(self):
        result, _ = _make_clf_result()
        ct = ComparisonTable(result)
        df = ct.dataframe()
        assert len(df) == len(result.successful)

    def test_regression_to_latex_not_empty(self):
        result, _ = _make_reg_result()
        ct = ComparisonTable(result)
        latex = ct.to_latex()
        assert len(latex) > 0
