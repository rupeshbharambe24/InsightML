"""Integration tests: full pipeline on a Titanic-like classification dataset.

Exercises the end-to-end flow:
    iml.explore() → iml.analyze(run_battle=False) → iml.analyze(battle_families=["linear"])

All tests are marked as slow integration tests and share a single module-scoped
fixture to avoid re-running the pipeline for every test function.
"""

from __future__ import annotations

import pandas as pd
import pytest

import insightml as iml
from insightml.eda.result import EDAResult
from insightml.report.builder import AnalysisReport

# ---------------------------------------------------------------------------
# Module-level marks
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Module-scoped fixtures — pipeline runs only once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def titanic_df() -> pd.DataFrame:
    """Load Titanic dataset (seaborn or synthetic fallback)."""
    return iml.load_titanic()


@pytest.fixture(scope="module")
def eda_result(titanic_df: pd.DataFrame) -> EDAResult:
    """EDAResult from iml.explore()."""
    return iml.explore(titanic_df, target="survived")


@pytest.fixture(scope="module")
def report_no_battle(titanic_df: pd.DataFrame) -> AnalysisReport:
    """AnalysisReport with EDA + Intelligence only (run_battle=False)."""
    return iml.analyze(titanic_df, target="survived", run_battle=False)


@pytest.fixture(scope="module")
def report_with_battle(titanic_df: pd.DataFrame) -> AnalysisReport:
    """AnalysisReport with the linear model family battle."""
    return iml.analyze(
        titanic_df,
        target="survived",
        battle_families=["linear"],
        cv=3,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Stage 1 — EDA
# ---------------------------------------------------------------------------


def test_explore_returns_eda_result(eda_result: EDAResult) -> None:
    """iml.explore() must return an EDAResult instance."""
    assert isinstance(eda_result, EDAResult)


def test_eda_overview_runs(eda_result: EDAResult) -> None:
    """eda.overview.show() should not raise any exception."""
    # show() renders HTML to display; we call it but swallow any display output
    try:
        eda_result.overview.show()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"eda.overview.show() raised an unexpected exception: {exc}")


def test_eda_correlations_runs(eda_result: EDAResult) -> None:
    """eda.correlations.unified() should return a non-empty DataFrame."""
    result = eda_result.correlations.unified()
    assert isinstance(result, pd.DataFrame), (
        f"Expected a DataFrame, got {type(result).__name__}"
    )
    assert not result.empty, "correlations.unified() returned an empty DataFrame"


# ---------------------------------------------------------------------------
# Stage 2 — Intelligence (no-battle report)
# ---------------------------------------------------------------------------


def test_analyze_no_battle_returns_report(report_no_battle: AnalysisReport) -> None:
    """analyze(run_battle=False) must return an AnalysisReport with eda + intelligence."""
    assert isinstance(report_no_battle, AnalysisReport)
    assert report_no_battle.eda is not None, "report.eda should be set"
    assert report_no_battle.intelligence is not None, "report.intelligence should be set"


def test_intelligence_readiness_score(report_no_battle: AnalysisReport) -> None:
    """report.intelligence.readiness.score must be a float in [0, 100]."""
    score = report_no_battle.intelligence.readiness.score
    assert isinstance(score, (int, float)), (
        f"readiness.score should be numeric, got {type(score).__name__}"
    )
    assert 0 <= score <= 100, (
        f"readiness.score={score} is outside the expected [0, 100] range"
    )


def test_intelligence_leakage_list(report_no_battle: AnalysisReport) -> None:
    """report.intelligence.leakage must be a list (possibly empty)."""
    leakage = report_no_battle.intelligence.leakage
    assert isinstance(leakage, list), (
        f"intelligence.leakage should be a list, got {type(leakage).__name__}"
    )


# ---------------------------------------------------------------------------
# Stage 3+4 — Battle + Compare (with linear family)
# ---------------------------------------------------------------------------


def test_analyze_with_linear_family(report_with_battle: AnalysisReport) -> None:
    """analyze() with battle_families=['linear'] must return an AnalysisReport with models."""
    assert isinstance(report_with_battle, AnalysisReport)
    assert report_with_battle.models is not None, "report.models should be set after battle"


def test_report_has_best_model(report_with_battle: AnalysisReport) -> None:
    """report.models.best should be a non-None ModelScore after a successful battle."""
    best = report_with_battle.models.best
    assert best is not None, (
        "report.models.best is None — no model trained successfully. "
        f"Failed models: {[s.name for s in report_with_battle.models.failed]}"
    )


def test_comparator_table_has_models(report_with_battle: AnalysisReport) -> None:
    """report.compare.table.dataframe() must be a non-empty DataFrame."""
    assert report_with_battle.compare is not None, "report.compare should be set"
    df = report_with_battle.compare.table.dataframe()
    assert isinstance(df, pd.DataFrame), (
        f"compare.table.dataframe() should return a DataFrame, got {type(df).__name__}"
    )
    assert not df.empty, "compare.table.dataframe() returned an empty DataFrame"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    False,  # always run; heavy only in terms of disk I/O
    reason="HTML export skipped",
)
def test_report_export_produces_html(
    report_with_battle: AnalysisReport,
    tmp_path,
) -> None:
    """report.export() must create an HTML file starting with '<!DOCTYPE html>'."""
    out_path = tmp_path / "out.html"
    returned_path = report_with_battle.export(str(out_path))

    assert out_path.exists(), (
        f"Export did not create a file at {out_path} (returned path: {returned_path})"
    )

    content = out_path.read_text(encoding="utf-8")
    assert content.lstrip().lower().startswith("<!doctype html"), (
        "Exported file does not start with '<!DOCTYPE html'. "
        f"First 80 chars: {content[:80]!r}"
    )
