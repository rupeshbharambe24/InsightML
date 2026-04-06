"""Intelligence section builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from insightml.report.builder import AnalysisReport


def build_intelligence_section(report: "AnalysisReport") -> dict[str, Any] | None:
    """Build the pre-model intelligence section."""
    if report.intelligence is None:
        return None

    intel = report.intelligence
    parts: list[str] = []

    # Readiness gauge
    try:
        rs = intel.readiness
        gauge = rs.gauge_figure()
        parts.append("<h3>Data Readiness Score</h3>")
        parts.append(_fig_html(gauge, height=300))
        waterfall = rs.waterfall_figure()
        parts.append(_fig_html(waterfall, height=350))
    except Exception:
        pass

    # Leakage warnings
    try:
        warnings = intel.leakage
        if warnings:
            parts.append("<h3>Leakage Warnings</h3>")
            parts.append('<ul class="rec-list">')
            for w in warnings:
                parts.append(
                    f'<li class="danger"><strong>{w.column}</strong>: '
                    f'{w.method} (score={w.score:.3f}, severity={w.severity})</li>'
                )
            parts.append("</ul>")
    except Exception:
        pass

    # Feature importance
    try:
        fi = intel.feature_importance
        if fi is not None and not fi.empty:
            parts.append("<h3>Feature Importance Ranking</h3>")
            display_cols = [c for c in ["feature", "composite_rank", "mi", "abs_corr"] if c in fi.columns]
            parts.append(_df_to_html(fi[display_cols].head(15)))
    except Exception:
        pass

    # VIF
    try:
        vif_df = intel.vif
        if vif_df is not None and not vif_df.empty:
            high_vif = vif_df[vif_df["vif"] > 5]
            if not high_vif.empty:
                parts.append("<h3>Multicollinearity (VIF &gt; 5)</h3>")
                parts.append(_df_to_html(high_vif))
    except Exception:
        pass

    # Algorithm recommendations
    try:
        recs = intel.recommendations
        if recs is not None:
            parts.append("<h3>Algorithm Recommendations</h3>")
            parts.append('<ul class="rec-list">')
            for rec in recs.ranked[:5]:
                parts.append(
                    f"<li><strong>{rec.algorithm}</strong> "
                    f"(score={rec.score:.0f}): {rec.reasoning}</li>"
                )
            parts.append("</ul>")
    except Exception:
        pass

    if not parts:
        parts.append("<p>Intelligence analysis completed.</p>")

    return {
        "title": "Pre-Model Intelligence",
        "anchor": "intelligence",
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
            + df.to_html(classes="table", border=0, index=False)
            + "</div>"
        )
    except Exception:
        return ""
