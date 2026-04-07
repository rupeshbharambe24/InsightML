"""Tests for report/html_renderer.py, report/builder.py, and dml.analyze()."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dissectml.report.builder import AnalysisReport
from dissectml.report.html_renderer import render_html_report
from dissectml.report.narrative import (
    data_recommendations,
    ensemble_recommendation,
    executive_summary,
    model_narrative,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_report():
    """A minimal AnalysisReport with no sub-modules."""
    return AnalysisReport(
        task="classification",
        target="survived",
        n_samples=500,
        n_features=8,
    )


@pytest.fixture
def clf_df():
    rng = np.random.default_rng(42)
    n = 120
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "cat": rng.choice(["a", "b"], n),
        "target": rng.choice([0, 1], n),
    })


# ---------------------------------------------------------------------------
# Narrative tests
# ---------------------------------------------------------------------------

class TestNarrative:
    def test_executive_summary_returns_str(self):
        s = executive_summary(
            task="classification", target="survived",
            n_samples=891, n_features=7,
            best_model="RandomForest", best_score=0.84,
            primary_metric="accuracy",
            readiness_score=78, readiness_grade="C",
        )
        assert isinstance(s, str)
        assert "891" in s
        assert "RandomForest" in s

    def test_model_narrative(self):
        s = model_narrative(
            model_name="LogisticRegression",
            metrics={"accuracy": 0.80},
            primary_metric="accuracy",
            rank=1, n_models=5,
        )
        assert "LogisticRegression" in s
        assert "0.8" in s

    def test_data_recommendations_leakage(self):
        recs = data_recommendations(
            readiness_score=50,
            leakage_columns=["fare"],
            high_vif_columns=[],
            missing_pct=0.0,
        )
        assert any("Leakage" in r or "leakage" in r.lower() for r in recs)

    def test_data_recommendations_clean(self):
        recs = data_recommendations(
            readiness_score=95,
            leakage_columns=[],
            high_vif_columns=[],
            missing_pct=0.0,
        )
        assert len(recs) == 1
        assert "No major" in recs[0]

    def test_ensemble_recommendation(self):
        s = ensemble_recommendation(
            ensemble_candidates=[("RF", "LR", 0.15)],
            best_model="RF",
            pareto_models=["RF", "LR"],
        )
        assert "RF" in s
        assert "LR" in s


# ---------------------------------------------------------------------------
# HTML renderer tests
# ---------------------------------------------------------------------------

class TestHtmlRenderer:
    def test_renders_minimal_report(self, minimal_report):
        html = render_html_report(minimal_report)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "survived" in html

    def test_html_has_sections(self, minimal_report):
        html = render_html_report(minimal_report)
        assert "Executive Summary" in html

    def test_html_has_plotly_cdn(self, minimal_report):
        html = render_html_report(minimal_report)
        assert "plotly" in html.lower()

    def test_html_has_target_name(self, minimal_report):
        html = render_html_report(minimal_report)
        assert "survived" in html

    def test_html_has_sample_count(self, minimal_report):
        html = render_html_report(minimal_report)
        assert "500" in html


# ---------------------------------------------------------------------------
# AnalysisReport tests
# ---------------------------------------------------------------------------

class TestAnalysisReport:
    def test_summary_string(self, minimal_report):
        s = minimal_report.summary()
        assert "classification" in s
        assert "survived" in s

    def test_repr(self, minimal_report):
        r = repr(minimal_report)
        assert "AnalysisReport" in r

    def test_export_writes_file(self, minimal_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_report.html")
            out = minimal_report.export(path)
            assert Path(out).exists()
            content = Path(out).read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
            assert len(content) > 1000

    def test_export_default_path(self, minimal_report, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = minimal_report.export("test.html")
        assert Path(out).exists()

    def test_repr_html_returns_html(self, minimal_report):
        html = minimal_report._repr_html_()
        assert "<!DOCTYPE html>" in html


# ---------------------------------------------------------------------------
# Integration: dml.analyze() end-to-end
# ---------------------------------------------------------------------------

class TestAnalyzeIntegration:
    def test_analyze_returns_report(self, clf_df):
        import dissectml as dml
        report = dml.analyze(
            clf_df, target="target",
            task="classification",
            battle_families=["linear"],
            cv=3,
            n_jobs=1,
        )
        assert isinstance(report, AnalysisReport)
        assert report.task == "classification"

    def test_analyze_has_eda(self, clf_df):
        import dissectml as dml
        report = dml.analyze(
            clf_df, target="target",
            battle_families=["linear"], cv=3, n_jobs=1,
        )
        assert report.eda is not None

    def test_analyze_has_models(self, clf_df):
        import dissectml as dml
        from dissectml.battle.result import BattleResult
        report = dml.analyze(
            clf_df, target="target",
            battle_families=["linear"], cv=3, n_jobs=1,
        )
        assert isinstance(report.models, BattleResult)
        assert report.models.best is not None

    def test_analyze_has_compare(self, clf_df):
        import dissectml as dml
        from dissectml.compare.comparator import ModelComparator
        report = dml.analyze(
            clf_df, target="target",
            battle_families=["linear"], cv=3, n_jobs=1,
        )
        assert isinstance(report.compare, ModelComparator)

    def test_analyze_export_works(self, clf_df):
        import dissectml as dml
        report = dml.analyze(
            clf_df, target="target",
            battle_families=["linear"], cv=3, n_jobs=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "full_report.html")
            out = report.export(path)
            assert Path(out).exists()
            content = Path(out).read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content

    def test_analyze_no_battle(self, clf_df):
        import dissectml as dml
        report = dml.analyze(clf_df, target="target", run_battle=False)
        assert report.models is None
        assert report.eda is not None
        assert report.intelligence is not None

    def test_analyze_wrong_target_raises(self, clf_df):
        import dissectml as dml
        with pytest.raises(KeyError):
            dml.analyze(clf_df, target="nonexistent")
