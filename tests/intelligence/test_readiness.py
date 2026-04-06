"""Tests for intelligence/readiness.py, multicollinearity.py, feature_importance.py, recommendations.py."""

import numpy as np
import pandas as pd
import pytest

from dissectml.intelligence.feature_importance import compute_feature_importance
from dissectml.intelligence.multicollinearity import compute_condition_number, compute_vif
from dissectml.intelligence.readiness import ReadinessResult, compute_readiness
from dissectml.intelligence.recommendations import recommend_algorithms

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_df():
    rng = np.random.default_rng(7)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.choice(["a", "b", "c"], n)
    target = (x1 + x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target})


@pytest.fixture
def messy_df():
    rng = np.random.default_rng(8)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = x1 * 0.999 + rng.normal(0, 0.01, n)  # near-duplicate -> high VIF
    x3 = np.full(n, 5.0)  # constant
    target = rng.choice([0, 1, 1, 1, 1, 1, 1, 1], n)  # imbalanced
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target})
    # Introduce missing
    df.loc[:40, "x1"] = np.nan
    return df


# ---------------------------------------------------------------------------
# ReadinessResult / compute_readiness
# ---------------------------------------------------------------------------

class TestComputeReadiness:
    def test_returns_readiness_result(self, clean_df):
        result = compute_readiness(clean_df, target="target", task="classification")
        assert isinstance(result, ReadinessResult)

    def test_score_in_range(self, clean_df):
        result = compute_readiness(clean_df, target="target")
        assert 0 <= result.score <= 100

    def test_grade_assigned(self, clean_df):
        result = compute_readiness(clean_df, target="target")
        assert result.grade in ("A", "B", "C", "D", "F")

    def test_clean_data_high_score(self, clean_df):
        result = compute_readiness(clean_df, target="target", task="classification")
        assert result.score >= 60  # clean data should score reasonably well

    def test_messy_data_lower_score(self, messy_df, clean_df):
        clean = compute_readiness(clean_df, target="target", task="classification")
        messy = compute_readiness(messy_df, target="target", task="classification")
        assert messy.score <= clean.score

    def test_breakdown_has_expected_keys(self, clean_df):
        result = compute_readiness(clean_df, target="target")
        assert "missing_values" in result.breakdown
        assert "outliers" in result.breakdown

    def test_recommendations_list(self, messy_df):
        result = compute_readiness(messy_df, target="target", task="classification")
        assert isinstance(result.recommendations, list)

    def test_summary_string(self, clean_df):
        result = compute_readiness(clean_df, target="target")
        s = result.summary()
        assert "Readiness" in s

    def test_html_repr(self, clean_df):
        result = compute_readiness(clean_df, target="target")
        html = result._repr_html_()
        assert "<table" in html.lower()


# ---------------------------------------------------------------------------
# VIF / multicollinearity
# ---------------------------------------------------------------------------

class TestComputeVIF:
    def test_returns_dataframe(self, clean_df):
        vif = compute_vif(clean_df.drop(columns=["target", "x3"]))
        assert isinstance(vif, pd.DataFrame)

    def test_columns_present(self, clean_df):
        vif = compute_vif(clean_df.drop(columns=["target", "x3"]))
        assert set(vif.columns) >= {"feature", "vif", "severity"}

    def test_high_vif_for_duplicates(self, messy_df):
        vif = compute_vif(messy_df.drop(columns=["target", "x3"]).dropna())
        max_vif = vif["vif"].max()
        assert max_vif > 10  # near-duplicate columns should have high VIF

    def test_too_few_columns_returns_empty(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        vif = compute_vif(df)
        assert vif.empty or len(vif) == 0

    def test_condition_number_returns_dict(self, clean_df):
        result = compute_condition_number(clean_df.drop(columns=["target", "x3"]))
        assert "condition_number" in result
        assert "severity" in result
        assert result["severity"] in ("low", "moderate", "severe", "unknown")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class TestComputeFeatureImportance:
    def test_returns_dataframe(self, clean_df):
        result = compute_feature_importance(clean_df, target="target")
        assert isinstance(result, pd.DataFrame)

    def test_has_composite_rank(self, clean_df):
        result = compute_feature_importance(clean_df, target="target")
        assert "composite_rank" in result.columns

    def test_sorted_ascending(self, clean_df):
        result = compute_feature_importance(clean_df, target="target")
        ranks = result["composite_rank"].tolist()
        assert ranks == sorted(ranks)

    def test_correlated_feature_ranks_high(self):
        rng = np.random.default_rng(5)
        n = 200
        important = rng.normal(0, 1, n)
        noise = rng.normal(0, 1, n)
        target = (important > 0).astype(int)
        df = pd.DataFrame({"important": important, "noise": noise, "target": target})
        result = compute_feature_importance(df, target="target")
        top_feature = result.iloc[0]["feature"]
        assert top_feature == "important"

    def test_missing_target_raises(self, clean_df):
        with pytest.raises(KeyError):
            compute_feature_importance(clean_df, target="nonexistent")


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

class TestRecommendAlgorithms:
    def test_returns_result(self):
        result = recommend_algorithms(n_samples=500, n_features=10, task="classification")
        from dissectml.intelligence.recommendations import RecommendationResult
        assert isinstance(result, RecommendationResult)

    def test_all_families_ranked(self):
        result = recommend_algorithms(n_samples=500, n_features=10, task="classification")
        assert len(result.ranked) >= 5

    def test_nonlinear_boosts_trees(self):
        linear = recommend_algorithms(n_samples=500, n_features=10, task="classification",
                                       has_nonlinear=False)
        nonlinear = recommend_algorithms(n_samples=500, n_features=10, task="classification",
                                          has_nonlinear=True)
        # With non-linear, tree ensembles should rank higher
        linear_rank = next(r["rank"] for r in linear.ranked if r["algorithm"] == "TreeEnsembles")
        nonlinear_rank = next(r["rank"] for r in nonlinear.ranked if r["algorithm"] == "TreeEnsembles")
        assert nonlinear_rank <= linear_rank

    def test_small_dataset_penalises_nn(self):
        result = recommend_algorithms(n_samples=100, n_features=5, task="classification",
                                       is_small_dataset=True)
        nn_score = next(r["score"] for r in result.ranked if r["algorithm"] == "NeuralNetwork")
        linear_score = next(r["score"] for r in result.ranked if r["algorithm"] == "LinearModels")
        assert linear_score >= nn_score

    def test_top_returns_list(self):
        result = recommend_algorithms(n_samples=500, n_features=10, task="regression")
        top3 = result.top(3)
        assert len(top3) == 3
        assert all(isinstance(x, str) for x in top3)

    def test_html_repr(self):
        result = recommend_algorithms(n_samples=500, n_features=10, task="classification")
        html = result._repr_html_()
        assert "<table" in html.lower()
