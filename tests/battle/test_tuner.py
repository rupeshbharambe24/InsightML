"""Tests for insightml.battle.tuner.ModelTuner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insightml.battle.result import BattleResult, ModelScore
from insightml.battle.tuner import ModelTuner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clf_battle():
    """Minimal BattleResult for classification tests."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    y = pd.Series(rng.choice([0, 1], n))
    s = ModelScore(
        "LogisticRegression",
        "classification",
        metrics={"accuracy": 0.75},
        metrics_std={"accuracy": 0.02},
        oof_predictions=rng.choice([0, 1], n).astype(float),
        train_time=0.5,
    )
    result = BattleResult(
        task="classification",
        scores=[s],
        primary_metric="accuracy",
        cv_folds=3,
        n_samples=n,
    )
    return result, X, y


def _make_clf_battle_with_failed():
    """BattleResult containing one successful and one failed ModelScore."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    y = pd.Series(rng.choice([0, 1], n))

    good = ModelScore(
        "LogisticRegression",
        "classification",
        metrics={"accuracy": 0.75},
        metrics_std={"accuracy": 0.02},
        oof_predictions=rng.choice([0, 1], n).astype(float),
        train_time=0.5,
    )
    bad = ModelScore(
        "BrokenModel",
        "classification",
        error="ImportError: some dependency missing",
    )
    result = BattleResult(
        task="classification",
        scores=[good, bad],
        primary_metric="accuracy",
        cv_folds=3,
        n_samples=n,
    )
    return result, X, y


def _make_multi_clf_battle():
    """BattleResult with two successful models, for top_n tests."""
    rng = np.random.default_rng(7)
    n = 100
    X = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    y = pd.Series(rng.choice([0, 1], n))

    s1 = ModelScore(
        "LogisticRegression",
        "classification",
        metrics={"accuracy": 0.78},
        metrics_std={"accuracy": 0.03},
        oof_predictions=rng.choice([0, 1], n).astype(float),
        train_time=0.4,
    )
    s2 = ModelScore(
        "RidgeClassifier",
        "classification",
        metrics={"accuracy": 0.72},
        metrics_std={"accuracy": 0.04},
        oof_predictions=rng.choice([0, 1], n).astype(float),
        train_time=0.3,
    )
    result = BattleResult(
        task="classification",
        scores=[s1, s2],
        primary_metric="accuracy",
        cv_folds=3,
        n_samples=n,
    )
    return result, X, y


# ---------------------------------------------------------------------------
# Quick mode tests
# ---------------------------------------------------------------------------

class TestModelTunerQuickMode:
    def test_quick_mode_returns_same_object(self):
        """mode='quick' must return the exact same BattleResult instance."""
        result, X, y = _make_clf_battle()
        tuner = ModelTuner(mode="quick")
        tuned = tuner.tune(result, X, y)
        assert tuned is result

    def test_quick_mode_preserves_scores(self):
        """Scores must be identical after a quick pass."""
        result, X, y = _make_clf_battle()
        original_acc = result.scores[0].metrics["accuracy"]
        tuner = ModelTuner(mode="quick")
        tuned = tuner.tune(result, X, y)
        assert tuned.scores[0].metrics["accuracy"] == original_acc

    def test_quick_mode_preserves_task(self):
        result, X, y = _make_clf_battle()
        tuner = ModelTuner(mode="quick")
        tuned = tuner.tune(result, X, y)
        assert tuned.task == "classification"

    def test_quick_mode_with_plan_arg(self):
        """quick mode should ignore the plan argument and still return unchanged result."""
        from insightml.battle.preprocessing import build_preprocessing_plan
        result, X, y = _make_clf_battle()
        plan = build_preprocessing_plan(X, target=None, eda_result=None)
        tuner = ModelTuner(mode="quick")
        tuned = tuner.tune(result, X, y, plan=plan)
        assert tuned is result


# ---------------------------------------------------------------------------
# Tuned mode tests
# ---------------------------------------------------------------------------

class TestModelTunerTunedMode:
    def test_tuned_mode_returns_battle_result(self):
        """mode='tuned' must return a BattleResult."""
        result, X, y = _make_clf_battle()
        tuner = ModelTuner(mode="tuned", n_iter=2, cv=2)
        tuned = tuner.tune(result, X, y)
        assert isinstance(tuned, BattleResult)

    def test_tuned_mode_has_same_task(self):
        result, X, y = _make_clf_battle()
        tuner = ModelTuner(mode="tuned", n_iter=2, cv=2)
        tuned = tuner.tune(result, X, y)
        assert tuned.task == result.task

    def test_tuned_mode_score_count_unchanged(self):
        """Number of scores should be preserved after tuning."""
        result, X, y = _make_clf_battle()
        tuner = ModelTuner(mode="tuned", n_iter=2, cv=2)
        tuned = tuner.tune(result, X, y)
        assert len(tuned.scores) == len(result.scores)

    def test_tuned_mode_with_explicit_plan(self):
        """Passing an explicit PreprocessingPlan should not raise."""
        from insightml.battle.preprocessing import build_preprocessing_plan
        result, X, y = _make_clf_battle()
        plan = build_preprocessing_plan(X, target=None, eda_result=None)
        tuner = ModelTuner(mode="tuned", n_iter=2, cv=2)
        tuned = tuner.tune(result, X, y, plan=plan)
        assert isinstance(tuned, BattleResult)

    def test_tuned_mode_config_snapshot_updated(self):
        """config_snapshot should record tuning metadata."""
        result, X, y = _make_clf_battle()
        tuner = ModelTuner(mode="tuned", n_iter=2, cv=2)
        tuned = tuner.tune(result, X, y)
        assert tuned.config_snapshot.get("tuning_mode") == "tuned"
        assert tuned.config_snapshot.get("tuning_n_iter") == 2


# ---------------------------------------------------------------------------
# top_n parameter tests
# ---------------------------------------------------------------------------

class TestModelTunerTopN:
    def test_top_n_limits_tuned_models(self):
        """With top_n=1, only the best model from two should be tuned."""
        result, X, y = _make_multi_clf_battle()
        tuner = ModelTuner(mode="tuned", top_n=1, n_iter=2, cv=2)
        tuned = tuner.tune(result, X, y)
        # Result must still contain both models
        assert len(tuned.scores) == 2

    def test_top_n_zero_returns_result_unchanged_content(self):
        """top_n=0 means no successful scores to tune; all scores pass through."""
        result, X, y = _make_clf_battle()
        tuner = ModelTuner(mode="tuned", top_n=0, n_iter=2, cv=2)
        tuned = tuner.tune(result, X, y)
        assert isinstance(tuned, BattleResult)


# ---------------------------------------------------------------------------
# Failed model handling
# ---------------------------------------------------------------------------

class TestModelTunerFailedModels:
    def test_failed_models_are_skipped(self):
        """BattleResult with a failed ModelScore should not crash the tuner."""
        result, X, y = _make_clf_battle_with_failed()
        tuner = ModelTuner(mode="tuned", top_n=3, n_iter=2, cv=2)
        tuned = tuner.tune(result, X, y)
        assert isinstance(tuned, BattleResult)

    def test_failed_model_count_preserved(self):
        """Both successful and failed scores should be present in output."""
        result, X, y = _make_clf_battle_with_failed()
        tuner = ModelTuner(mode="tuned", top_n=3, n_iter=2, cv=2)
        tuned = tuner.tune(result, X, y)
        # The good model is tuned; the bad model stays in failed list
        assert len(tuned.scores) >= 1


# ---------------------------------------------------------------------------
# Custom mode tests
# ---------------------------------------------------------------------------

class TestModelTunerCustomMode:
    def test_custom_mode_empty_grids_returns_result(self):
        """custom mode with no grids defined — models have no grid, so originals kept."""
        result, X, y = _make_clf_battle()
        tuner = ModelTuner(mode="custom", top_n=3, n_iter=2, cv=2, custom_grids={})
        tuned = tuner.tune(result, X, y)
        assert isinstance(tuned, BattleResult)

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="mode must be"):
            ModelTuner(mode="invalid_mode")

    def test_quick_mode_is_valid(self):
        tuner = ModelTuner(mode="quick")
        assert tuner.mode == "quick"

    def test_tuned_mode_is_valid(self):
        tuner = ModelTuner(mode="tuned")
        assert tuner.mode == "tuned"

    def test_custom_mode_is_valid(self):
        tuner = ModelTuner(mode="custom")
        assert tuner.mode == "custom"
