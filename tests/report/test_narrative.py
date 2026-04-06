"""Additional / different tests for insightml.report.narrative.

test_html_renderer.py::TestNarrative already covers:
  - executive_summary returns str, contains sample count + best model
  - model_narrative basic format
  - data_recommendations leakage branch
  - data_recommendations clean-data ("No major …") branch
  - ensemble_recommendation two-candidate path

These tests cover the remaining cases that are NOT tested there.
"""

from __future__ import annotations

import pytest

from insightml.report.narrative import (
    data_recommendations,
    ensemble_recommendation,
    executive_summary,
    model_narrative,
)


# ---------------------------------------------------------------------------
# executive_summary — regression variant
# ---------------------------------------------------------------------------

class TestExecutiveSummaryRegression:
    """executive_summary called with task="regression"."""

    @pytest.fixture
    def reg_summary(self):
        return executive_summary(
            task="regression",
            target="price",
            n_samples=2000,
            n_features=15,
            best_model="GradientBoosting",
            best_score=0.923,
            primary_metric="r2",
            readiness_score=85,
            readiness_grade="B",
        )

    def test_regression_summary_contains_task(self, reg_summary):
        assert "regression" in reg_summary

    def test_regression_summary_contains_best_model(self, reg_summary):
        assert "GradientBoosting" in reg_summary


# ---------------------------------------------------------------------------
# model_narrative — rank differences
# ---------------------------------------------------------------------------

class TestModelNarrativeRank:
    def _make_narrative(self, rank: int) -> str:
        return model_narrative(
            model_name="SVC",
            metrics={"accuracy": 0.75},
            primary_metric="accuracy",
            rank=rank,
            n_models=5,
        )

    def test_rank1_vs_rank3_differ(self):
        n1 = self._make_narrative(rank=1)
        n3 = self._make_narrative(rank=3)
        assert n1 != n3

    def test_best_model_narrative_different_from_worst(self):
        best = model_narrative(
            model_name="RandomForest",
            metrics={"accuracy": 0.91},
            primary_metric="accuracy",
            rank=1,
            n_models=4,
        )
        worst = model_narrative(
            model_name="DummyClassifier",
            metrics={"accuracy": 0.55},
            primary_metric="accuracy",
            rank=4,
            n_models=4,
        )
        assert best != worst


# ---------------------------------------------------------------------------
# data_recommendations — VIF, missing, and clean branches
# ---------------------------------------------------------------------------

class TestDataRecommendations:
    def test_high_vif_columns_mentioned(self):
        recs = data_recommendations(
            readiness_score=80,
            leakage_columns=[],
            high_vif_columns=["col_a", "col_b"],
            missing_pct=0.0,
        )
        combined = " ".join(recs)
        assert "VIF" in combined

    def test_high_missing_pct_triggers_recommendation(self):
        recs = data_recommendations(
            readiness_score=70,
            leakage_columns=[],
            high_vif_columns=[],
            missing_pct=0.25,  # 25 % > 10 % threshold
        )
        combined = " ".join(recs)
        assert "missing" in combined.lower() or "Missing" in combined

    def test_all_clean_returns_single_no_major_item(self):
        recs = data_recommendations(
            readiness_score=95,
            leakage_columns=[],
            high_vif_columns=[],
            missing_pct=0.0,
        )
        assert len(recs) == 1
        assert "No major" in recs[0]


# ---------------------------------------------------------------------------
# ensemble_recommendation
# ---------------------------------------------------------------------------

class TestEnsembleRecommendation:
    def test_empty_candidates_still_returns_string(self):
        result = ensemble_recommendation(
            ensemble_candidates=[],
            best_model="RandomForest",
            pareto_models=["RandomForest", "LogisticRegression"],
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_single_candidate_contains_candidate_names(self):
        result = ensemble_recommendation(
            ensemble_candidates=[("ExtraTrees", "GradientBoosting", 0.18)],
            best_model="ExtraTrees",
            pareto_models=["ExtraTrees", "GradientBoosting"],
        )
        assert "ExtraTrees" in result
        assert "GradientBoosting" in result
