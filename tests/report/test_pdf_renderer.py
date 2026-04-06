"""Tests for insightml.report.pdf_renderer.

weasyprint is an optional dependency (pip install insightml[report]).
In the standard test environment it is not installed.

Strategy
--------
- Tests that need weasyprint use ``pytest.importorskip`` and are skipped
  when it is absent.
- Tests that verify the graceful ImportError path patch sys.modules to
  simulate a missing weasyprint, then reload the module so the top-level
  import attempt happens again inside the functions under test.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from insightml.report.builder import AnalysisReport


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_report():
    return AnalysisReport(
        task="classification",
        target="survived",
        n_samples=500,
        n_features=8,
    )


# ---------------------------------------------------------------------------
# ImportError path — weasyprint absent
# ---------------------------------------------------------------------------

def _reload_pdf_renderer_without_weasyprint(monkeypatch):
    """Patch sys.modules so weasyprint looks absent, then reload pdf_renderer."""
    monkeypatch.setitem(sys.modules, "weasyprint", None)
    import insightml.report.pdf_renderer as mod
    importlib.reload(mod)
    return mod


class TestPdfRendererWithoutWeasyprint:
    def test_render_pdf_raises_import_error(self, minimal_report, monkeypatch):
        mod = _reload_pdf_renderer_without_weasyprint(monkeypatch)
        with pytest.raises(ImportError):
            mod.render_pdf_report(minimal_report)

    def test_render_pdf_error_mentions_weasyprint(self, minimal_report, monkeypatch):
        mod = _reload_pdf_renderer_without_weasyprint(monkeypatch)
        with pytest.raises(ImportError, match="WeasyPrint"):
            mod.render_pdf_report(minimal_report)

    def test_export_pdf_raises_import_error(self, minimal_report, monkeypatch, tmp_path):
        mod = _reload_pdf_renderer_without_weasyprint(monkeypatch)
        with pytest.raises(ImportError):
            mod.export_pdf(minimal_report, tmp_path / "out.pdf")

    def test_error_message_mentions_pip_install(self, minimal_report, monkeypatch):
        mod = _reload_pdf_renderer_without_weasyprint(monkeypatch)
        with pytest.raises(ImportError, match="pip install insightml"):
            mod.render_pdf_report(minimal_report)


# ---------------------------------------------------------------------------
# Happy-path tests — only run when weasyprint IS installed AND loads cleanly
# ---------------------------------------------------------------------------

def _weasyprint_available() -> bool:
    try:
        import weasyprint  # noqa: F401
        return True
    except Exception:
        return False


_weasyprint_skip = pytest.mark.skipif(
    not _weasyprint_available(),
    reason="weasyprint not installed or native libraries missing",
)


@_weasyprint_skip
class TestPdfRendererWithWeasyprint:
    def test_render_pdf_returns_bytes(self, minimal_report):
        from insightml.report.pdf_renderer import render_pdf_report
        result = render_pdf_report(minimal_report)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_export_pdf_creates_file(self, minimal_report, tmp_path):
        from insightml.report.pdf_renderer import export_pdf
        out = export_pdf(minimal_report, tmp_path / "report.pdf")
        assert out.exists()
        assert out.stat().st_size > 0
