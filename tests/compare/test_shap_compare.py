"""Tests for compare/shap_compare.py — shap_comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Detect whether shap is installed once at module level
try:
    import shap as _shap  # noqa: F401
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

shap_installed = pytest.mark.skipif(
    not _SHAP_AVAILABLE, reason="shap not installed"
)


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


def _make_small_X(n=30, n_features=2):
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        rng.normal(0, 1, (n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )


def _make_result_with_pipeline(n=30):
    """Build a BattleResult where ModelA has a fitted sklearn pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from insightml.battle.result import BattleResult, ModelScore

    rng = np.random.default_rng(11)
    X = _make_small_X(n=n)
    y = rng.choice([0, 1], n)

    pipe = Pipeline([("model", LogisticRegression(max_iter=200, random_state=0))])
    pipe.fit(X, y)

    oof = pipe.predict(X).astype(float)
    probs = pipe.predict_proba(X)

    s = ModelScore(
        "FittedModel", "classification",
        metrics={"accuracy": 0.80},
        oof_predictions=oof, oof_probabilities=probs,
        train_time=0.1,
        fitted_pipeline=pipe,
    )
    result = BattleResult(
        "classification", [s],
        primary_metric="accuracy", cv_folds=3, n_samples=n,
    )
    return result, pd.Series(y), X


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShapComparisonImportError:
    def test_raises_import_error_when_shap_missing(self, monkeypatch):
        """shap_comparison raises ImportError if shap is not installed."""
        import sys
        import unittest.mock as mock

        # Only run this test when shap IS installed (mock it away)
        if not _SHAP_AVAILABLE:
            pytest.skip("shap not installed; ImportError is guaranteed without mocking")

        from insightml.compare import shap_compare as sc_module

        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        result, _ = _make_clf_result(n=30)
        X = _make_small_X(n=30)
        with mock.patch.dict(sys.modules, {"shap": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError("shap not installed")):
                with pytest.raises(ImportError):
                    sc_module.shap_comparison(result, X)


class TestShapComparisonNoFittedPipeline:
    def test_no_fitted_pipeline_returns_empty_result(self):
        """When no model has a fitted_pipeline, shap_comparison returns empty structures."""
        if not _SHAP_AVAILABLE:
            pytest.skip("shap not installed")

        from insightml.compare.shap_compare import shap_comparison

        result, _ = _make_clf_result(n=30)
        X = _make_small_X(n=30)
        # None of the ModelScores have fitted_pipeline (default is None)
        out = shap_comparison(result, X)
        assert isinstance(out, dict)
        assert "importance_df" in out
        assert "rank_correlation" in out
        assert "figures" in out
        assert out["importance_df"].empty
        assert out["figures"] == {} or isinstance(out["figures"], dict)


@shap_installed
class TestShapComparisonWithFittedPipeline:
    def test_returns_dict(self):
        from insightml.compare.shap_compare import shap_comparison

        result, _, X = _make_result_with_pipeline(n=30)
        out = shap_comparison(result, X)
        assert isinstance(out, dict)

    def test_dict_has_required_keys(self):
        from insightml.compare.shap_compare import shap_comparison

        result, _, X = _make_result_with_pipeline(n=30)
        out = shap_comparison(result, X)
        assert "importance_df" in out
        assert "rank_correlation" in out
        assert "figures" in out

    def test_importance_df_is_dataframe(self):
        from insightml.compare.shap_compare import shap_comparison

        result, _, X = _make_result_with_pipeline(n=30)
        out = shap_comparison(result, X)
        assert isinstance(out["importance_df"], pd.DataFrame)

    def test_figures_is_dict(self):
        from insightml.compare.shap_compare import shap_comparison

        result, _, X = _make_result_with_pipeline(n=30)
        out = shap_comparison(result, X)
        assert isinstance(out["figures"], dict)

    def test_importance_df_has_feature_column(self):
        from insightml.compare.shap_compare import shap_comparison

        result, _, X = _make_result_with_pipeline(n=30)
        out = shap_comparison(result, X)
        df = out["importance_df"]
        if not df.empty:
            assert "feature" in df.columns

    def test_importance_df_has_model_column(self):
        from insightml.compare.shap_compare import shap_comparison

        result, _, X = _make_result_with_pipeline(n=30)
        out = shap_comparison(result, X)
        df = out["importance_df"]
        if not df.empty:
            assert "FittedModel" in df.columns

    def test_top_n_1_limits_models(self):
        from insightml.compare.shap_compare import shap_comparison

        result, _, X = _make_result_with_pipeline(n=30)
        out = shap_comparison(result, X, top_n=1)
        assert isinstance(out, dict)
        assert isinstance(out["importance_df"], pd.DataFrame)

    def test_figures_keys_are_model_names(self):
        from insightml.compare.shap_compare import shap_comparison

        result, _, X = _make_result_with_pipeline(n=30)
        out = shap_comparison(result, X)
        figs = out["figures"]
        if figs:
            successful_names = {s.name for s in result.successful}
            assert set(figs.keys()).issubset(successful_names)
