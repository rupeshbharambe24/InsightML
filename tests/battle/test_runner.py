"""Tests for BattleRunner — integration tests using small DataFrames."""

import numpy as np
import pandas as pd
import pytest

from insightml.battle.result import BattleResult, ModelScore
from insightml.battle.runner import BattleRunner


@pytest.fixture
def clf_df():
    rng = np.random.default_rng(42)
    n = 150
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(1, 1, n),
        "x3": rng.choice(["a", "b", "c"], n),
        "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def reg_df():
    rng = np.random.default_rng(42)
    n = 150
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(1, 1, n)
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": rng.choice(["a", "b"], n),
        "target": x1 * 2 + x2 + rng.normal(0, 0.1, n),
    })


class TestBattleRunner:
    def test_classification_returns_battle_result(self, clf_df):
        runner = BattleRunner()
        result = runner.run(
            clf_df, target="target", task="classification",
            families=["linear", "tree"],
            exclude=["GradientBoostingClassifier"],
            cv=3,
            n_jobs=1,
        )
        assert isinstance(result, BattleResult)
        assert result.task == "classification"

    def test_regression_returns_battle_result(self, reg_df):
        runner = BattleRunner()
        result = runner.run(
            reg_df, target="target", task="regression",
            families=["linear"],
            cv=3,
            n_jobs=1,
        )
        assert isinstance(result, BattleResult)
        assert result.task == "regression"

    def test_leaderboard_has_rows(self, clf_df):
        runner = BattleRunner()
        result = runner.run(
            clf_df, target="target", task="classification",
            families=["linear"],
            cv=3, n_jobs=1,
        )
        lb = result.leaderboard()
        assert isinstance(lb, pd.DataFrame)
        assert len(lb) > 0

    def test_best_model_not_none(self, clf_df):
        runner = BattleRunner()
        result = runner.run(
            clf_df, target="target", task="classification",
            families=["linear"],
            cv=3, n_jobs=1,
        )
        assert result.best is not None
        assert isinstance(result.best, ModelScore)

    def test_oof_predictions_shape(self, clf_df):
        runner = BattleRunner()
        result = runner.run(
            clf_df, target="target", task="classification",
            models=["LogisticRegression"],
            cv=3, n_jobs=1,
        )
        assert len(result.successful) >= 1
        score = result.successful[0]
        assert score.oof_predictions is not None
        assert len(score.oof_predictions) == len(clf_df)

    def test_model_score_metrics_present(self, clf_df):
        runner = BattleRunner()
        result = runner.run(
            clf_df, target="target", task="classification",
            models=["LogisticRegression"],
            cv=3, n_jobs=1,
        )
        score = result.get("LogisticRegression")
        assert "accuracy" in score.metrics
        assert 0.0 <= score.metrics["accuracy"] <= 1.0

    def test_filter_by_names(self, clf_df):
        runner = BattleRunner()
        result = runner.run(
            clf_df, target="target", task="classification",
            models=["LogisticRegression", "DecisionTreeClassifier"],
            cv=3, n_jobs=1,
        )
        names = {s.name for s in result.scores}
        assert names == {"LogisticRegression", "DecisionTreeClassifier"}

    def test_missing_target_raises(self, clf_df):
        runner = BattleRunner()
        with pytest.raises(KeyError):
            runner.run(clf_df, target="nonexistent")

    def test_repr(self, clf_df):
        runner = BattleRunner()
        result = runner.run(
            clf_df, target="target", task="classification",
            families=["linear"], cv=3, n_jobs=1,
        )
        r = repr(result)
        assert "BattleResult" in r

    def test_html_repr(self, clf_df):
        runner = BattleRunner()
        result = runner.run(
            clf_df, target="target", task="classification",
            families=["linear"], cv=3, n_jobs=1,
        )
        html = result._repr_html_()
        assert "<table" in html.lower() or "<h3" in html.lower()


class TestBattleResultHelpers:
    def test_to_dict(self, clf_df):
        runner = BattleRunner()
        result = runner.run(
            clf_df, target="target", task="classification",
            families=["linear"], cv=3, n_jobs=1,
        )
        d = result.to_dict()
        assert d["task"] == "classification"
        assert "n_models_ok" in d
        assert "best_model" in d

    def test_get_missing_raises(self, clf_df):
        runner = BattleRunner()
        result = runner.run(
            clf_df, target="target", task="classification",
            families=["linear"], cv=3, n_jobs=1,
        )
        with pytest.raises(KeyError):
            result.get("DoesNotExist")
