"""Additional tests for insightml.report.builder.AnalysisReport.

test_html_renderer.py already covers basic export/repr/summary via
TestAnalysisReport.  These tests focus on attribute initialisation,
default values, and fine-grained contract of summary() / export().
"""

from __future__ import annotations

from pathlib import Path

import pytest

from insightml.report.builder import AnalysisReport

# ---------------------------------------------------------------------------
# Construction and attribute access
# ---------------------------------------------------------------------------

class TestAnalysisReportAttributes:
    """Verify dataclass fields are stored and defaulted correctly."""

    def test_construction_stores_task(self):
        report = AnalysisReport(task="classification", target="y",
                                n_samples=100, n_features=5)
        assert report.task == "classification"

    def test_construction_stores_target(self):
        report = AnalysisReport(task="classification", target="y",
                                n_samples=100, n_features=5)
        assert report.target == "y"

    def test_construction_stores_n_samples(self):
        report = AnalysisReport(task="classification", target="y",
                                n_samples=100, n_features=5)
        assert report.n_samples == 100

    def test_construction_stores_n_features(self):
        report = AnalysisReport(task="classification", target="y",
                                n_samples=100, n_features=5)
        assert report.n_features == 5

    def test_eda_defaults_to_none(self):
        report = AnalysisReport(task="classification", target="y",
                                n_samples=100, n_features=5)
        assert report.eda is None

    def test_models_defaults_to_none(self):
        report = AnalysisReport(task="classification", target="y",
                                n_samples=100, n_features=5)
        assert report.models is None

    def test_compare_defaults_to_none(self):
        report = AnalysisReport(task="classification", target="y",
                                n_samples=100, n_features=5)
        assert report.compare is None

    def test_intelligence_defaults_to_none(self):
        report = AnalysisReport(task="classification", target="y",
                                n_samples=100, n_features=5)
        assert report.intelligence is None


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

class TestSummary:
    """summary() should embed key dataset facts in plain text."""

    @pytest.fixture
    def report(self):
        return AnalysisReport(task="classification", target="y",
                              n_samples=100, n_features=5)

    def test_summary_contains_task(self, report):
        assert "classification" in report.summary()

    def test_summary_contains_target(self, report):
        assert "y" in report.summary()

    def test_summary_contains_sample_count_as_string(self, report):
        # summary() uses {:,} formatting; 100 -> "100"
        assert "100" in report.summary()


# ---------------------------------------------------------------------------
# _repr_html_()
# ---------------------------------------------------------------------------

class TestReprHtml:
    def test_repr_html_contains_doctype(self):
        report = AnalysisReport(task="regression", target="price",
                                n_samples=200, n_features=10)
        html = report._repr_html_()
        assert "<!DOCTYPE html>" in html


# ---------------------------------------------------------------------------
# export()
# ---------------------------------------------------------------------------

class TestExport:
    @pytest.fixture
    def report(self):
        return AnalysisReport(task="classification", target="y",
                              n_samples=100, n_features=5)

    def test_export_creates_file(self, report, tmp_path):
        out_path = tmp_path / "report.html"
        report.export(str(out_path))
        assert out_path.exists()

    def test_export_returns_absolute_path_string(self, report, tmp_path):
        out_path = tmp_path / "report.html"
        result = report.export(str(out_path))
        assert isinstance(result, str)
        assert Path(result).is_absolute()

    def test_exported_content_contains_doctype(self, report, tmp_path):
        out_path = tmp_path / "report.html"
        result = report.export(str(out_path))
        content = Path(result).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
