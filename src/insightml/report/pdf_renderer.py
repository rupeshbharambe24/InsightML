"""Optional PDF export for InsightML reports.

Requires the ``report`` extra::

    pip install insightml[report]

which installs ``weasyprint`` and ``kaleido``.

Usage::

    from insightml.report.pdf_renderer import render_pdf_report

    pdf_bytes = render_pdf_report(report)
    Path("report.pdf").write_bytes(pdf_bytes)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from insightml.report.builder import AnalysisReport


def render_pdf_report(report: "AnalysisReport") -> bytes:
    """Render *report* as a PDF and return the raw bytes.

    Internally calls :func:`~insightml.report.html_renderer.render_html_report`
    to produce an HTML string, then converts it to PDF via
    `weasyprint <https://weasyprint.org/>`_.  Static chart images are embedded
    using kaleido.

    Args:
        report: The :class:`~insightml.report.builder.AnalysisReport` to render.

    Returns:
        Raw PDF bytes.

    Raises:
        ImportError: If ``weasyprint`` is not installed.
    """
    try:
        import weasyprint  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PDF export requires WeasyPrint. "
            "Install it with: pip install insightml[report]"
        ) from exc

    from insightml.report.html_renderer import render_html_report

    html_string = render_html_report(report, inline_plotly=True)
    pdf = weasyprint.HTML(string=html_string).write_pdf()
    return pdf


def export_pdf(report: "AnalysisReport", path: str | Path) -> Path:
    """Export *report* as a PDF file.

    Args:
        report: The :class:`~insightml.report.builder.AnalysisReport` to export.
        path: Destination file path.

    Returns:
        Absolute :class:`~pathlib.Path` of the written file.

    Raises:
        ImportError: If ``weasyprint`` is not installed.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(render_pdf_report(report))
    return out
