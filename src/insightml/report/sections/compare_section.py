"""Comparative analysis section builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from insightml.report.builder import AnalysisReport


def build_compare_section(report: "AnalysisReport") -> dict[str, Any] | None:
    """Build the comparative analysis section."""
    if report.compare is None:
        return None

    comp = report.compare
    parts: list[str] = []

    # Pareto front
    try:
        fig = comp.pareto
        parts.append("<h3>Accuracy vs Speed Pareto Front</h3>")
        parts.append(_fig_html(fig))
    except Exception:
        pass

    # Significance heatmap
    try:
        sig = comp.significance
        if "ttest" in sig:
            fig = sig["ttest"]["figure"]
            parts.append("<h3>Statistical Significance (Corrected t-test)</h3>")
            parts.append(_fig_html(fig, height=350))
        if "mcnemar" in sig:
            fig = sig["mcnemar"]["figure"]
            parts.append("<h3>McNemar Test</h3>")
            parts.append(_fig_html(fig, height=350))
    except Exception:
        pass

    # Error analysis — disagreement
    try:
        ea = comp.error_analysis
        fig = ea.disagreement_figure()
        parts.append("<h3>Cross-Model Disagreement</h3>")
        parts.append(_fig_html(fig, height=350))

        # Hard samples summary
        n_hard = len(ea.hard_indices)
        if n_hard > 0:
            n_total = len(ea.hard_indices)
            parts.append(
                f"<p><strong>{n_total}</strong> hard samples identified "
                f"(misclassified by multiple models).</p>"
            )
    except Exception:
        pass

    # ROC or residual curves
    try:
        if report.task == "classification":
            fig = comp.roc_curves
            parts.append("<h3>ROC Curves</h3>")
        else:
            fig = comp.residual_plots
            parts.append("<h3>Residual Plots</h3>")
        parts.append(_fig_html(fig))
    except Exception:
        pass

    if not parts:
        parts.append("<p>Comparative analysis completed.</p>")

    return {
        "title": "Comparative Analysis",
        "anchor": "compare",
        "content": "\n".join(parts),
    }


def _fig_html(fig, height: int = 420) -> str:
    try:
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": False},
            default_height=height,
        )
    except Exception:
        return "<p><em>Chart unavailable.</em></p>"
