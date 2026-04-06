"""Integration tests: full pipeline on a California Housing regression dataset.

Exercises the end-to-end flow:
    iml.explore() → iml.analyze(run_battle=False) → iml.analyze(task="regression", ...)

All tests are marked as slow integration tests and share module-scoped fixtures
to avoid re-running the pipeline for every test function.  The housing DataFrame
is down-sampled to 200 rows so these tests remain fast in CI.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

import insightml as iml
from insightml.eda.result import EDAResult
from insightml.report.builder import AnalysisReport

# ---------------------------------------------------------------------------
# Module-level marks
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Target column for the California Housing dataset
_TARGET = "MedHouseVal"


# ---------------------------------------------------------------------------
# Module-scoped fixtures — pipeline runs only once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def housing_df() -> pd.DataFrame:
    """Load California Housing dataset and down-sample to 200 rows for CI speed."""
    full_df = iml.load_housing()
    return full_df.sample(n=200, random_state=42).reset_index(drop=True)


@pytest.fixture(scope="module")
def eda_result(housing_df: pd.DataFrame) -> EDAResult:
    """EDAResult from iml.explore() on the housing sample."""
    return iml.explore(housing_df, target=_TARGET)


@pytest.fixture(scope="module")
def report_no_battle(housing_df: pd.DataFrame) -> AnalysisReport:
    """AnalysisReport with EDA + Intelligence only (run_battle=False)."""
    return iml.analyze(housing_df, target=_TARGET, run_battle=False)


@pytest.fixture(scope="module")
def report_with_battle(housing_df: pd.DataFrame) -> AnalysisReport:
    """AnalysisReport with the linear model family battle."""
    return iml.analyze(
        housing_df,
        target=_TARGET,
        task="regression",
        battle_families=["linear"],
        cv=3,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Stage 1 — EDA
# ---------------------------------------------------------------------------


def test_explore_housing_returns_eda(eda_result: EDAResult) -> None:
    """iml.explore() on a housing DataFrame must return an EDAResult instance."""
    assert isinstance(eda_result, EDAResult)


# ---------------------------------------------------------------------------
# Stage 2 — Intelligence (no-battle report)
# ---------------------------------------------------------------------------


def test_analyze_regression_no_battle(report_no_battle: AnalysisReport) -> None:
    """analyze(run_battle=False) must return an AnalysisReport with eda + intelligence."""
    assert isinstance(report_no_battle, AnalysisReport)
    assert report_no_battle.eda is not None, "report.eda should be set"
    assert report_no_battle.intelligence is not None, "report.intelligence should be set"


def test_regression_intelligence(report_no_battle: AnalysisReport) -> None:
    """report.intelligence.readiness.score must be greater than 0 for a clean dataset."""
    score = report_no_battle.intelligence.readiness.score
    assert isinstance(score, (int, float)), (
        f"readiness.score should be numeric, got {type(score).__name__}"
    )
    assert score > 0, (
        f"Expected readiness.score > 0 for a clean housing dataset, got {score}"
    )


# ---------------------------------------------------------------------------
# Stage 3+4 — Battle + Compare (with linear family, task=regression)
# ---------------------------------------------------------------------------


def test_analyze_regression_with_linear(report_with_battle: AnalysisReport) -> None:
    """analyze() with task='regression' must return an AnalysisReport with models."""
    assert isinstance(report_with_battle, AnalysisReport)
    assert report_with_battle.task == "regression", (
        f"Expected task='regression', got {report_with_battle.task!r}"
    )
    assert report_with_battle.models is not None, "report.models should be set after battle"


def test_regression_report_primary_metric_is_r2(report_with_battle: AnalysisReport) -> None:
    """The primary ranking metric for a regression battle should be 'r2'."""
    primary = report_with_battle.models.primary_metric
    assert primary == "r2", (
        f"Expected primary_metric='r2' for regression, got {primary!r}"
    )


def test_regression_best_model_has_r2(report_with_battle: AnalysisReport) -> None:
    """The best model's metrics dict must contain an 'r2' key."""
    best = report_with_battle.models.best
    assert best is not None, (
        "report.models.best is None — no model trained successfully. "
        f"Failed models: {[s.name for s in report_with_battle.models.failed]}"
    )
    assert "r2" in best.metrics, (
        f"Expected 'r2' in best model metrics, got keys: {list(best.metrics.keys())}"
    )


def test_regression_leaderboard_sorted(report_with_battle: AnalysisReport) -> None:
    """The leaderboard DataFrame should be sorted by r2 in descending order."""
    lb = report_with_battle.models.leaderboard()
    assert isinstance(lb, pd.DataFrame), (
        f"leaderboard() should return a DataFrame, got {type(lb).__name__}"
    )
    assert not lb.empty, "leaderboard() returned an empty DataFrame"

    if "r2" in lb.columns and len(lb) > 1:
        r2_values = lb["r2"].tolist()
        assert r2_values == sorted(r2_values, reverse=True), (
            f"Leaderboard is not sorted by r2 descending: {r2_values}"
        )


def test_regression_comparator_residuals(report_with_battle: AnalysisReport) -> None:
    """report.compare.residual_plots should be a plotly Figure for regression."""
    assert report_with_battle.compare is not None, "report.compare should be set"
    fig = report_with_battle.compare.residual_plots
    assert isinstance(fig, go.Figure), (
        f"compare.residual_plots should be a go.Figure, got {type(fig).__name__}"
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    False,  # always run; heavy only in terms of disk I/O
    reason="HTML export skipped",
)
def test_regression_export(
    report_with_battle: AnalysisReport,
    tmp_path,
) -> None:
    """report.export() must create an HTML file starting with '<!DOCTYPE html>'."""
    out_path = tmp_path / "housing.html"
    returned_path = report_with_battle.export(str(out_path))

    assert out_path.exists(), (
        f"Export did not create a file at {out_path} (returned path: {returned_path})"
    )

    content = out_path.read_text(encoding="utf-8")
    assert content.lstrip().lower().startswith("<!doctype html"), (
        "Exported file does not start with '<!DOCTYPE html'. "
        f"First 80 chars: {content[:80]!r}"
    )
