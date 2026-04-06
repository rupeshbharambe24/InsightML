"""EDA section builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from insightml.report.builder import AnalysisReport


def build_eda_section(report: AnalysisReport) -> dict[str, Any] | None:
    """Build the EDA section data dict.

    Returns None if no EDA result is available.
    """
    if report.eda is None:
        return None

    eda = report.eda
    parts: list[str] = []

    # Overview table
    try:
        overview_df = eda.overview.to_dataframe()
        if not overview_df.empty:
            parts.append("<h3>Dataset Overview</h3>")
            parts.append(_df_to_html(overview_df.head(30)))
    except Exception:
        pass

    # Correlation heatmap
    try:
        fig = eda.correlations.heatmap()
        parts.append("<h3>Correlation Heatmap</h3>")
        parts.append(_fig_html(fig))
    except Exception:
        pass

    # Missing data summary
    try:
        missing_df = eda.missing.summary()
        if isinstance(missing_df, dict):
            import pandas as pd
            missing_df = pd.DataFrame([missing_df]).T
        if hasattr(missing_df, "empty") and not missing_df.empty:
            parts.append("<h3>Missing Data</h3>")
            parts.append(_df_to_html(missing_df))
    except Exception:
        pass

    # Outliers
    try:
        outlier_fig = eda.outliers.comparison()
        parts.append("<h3>Outlier Comparison</h3>")
        parts.append(_fig_html(outlier_fig))
    except Exception:
        pass

    if not parts:
        parts.append("<p>EDA completed — access sub-modules for detailed results.</p>")

    return {
        "title": "EDA Findings",
        "anchor": "eda",
        "content": "\n".join(parts),
    }


def _fig_html(fig, height: int = 400) -> str:
    try:
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": False},
            default_height=height,
        )
    except Exception:
        return "<p><em>Chart unavailable.</em></p>"


def _df_to_html(df) -> str:
    try:
        return (
            '<div class="table-wrap">'
            + df.to_html(classes="table", border=0, index=True)
            + "</div>"
        )
    except Exception:
        return ""
