"""Executive summary section builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from insightml.report.builder import AnalysisReport


def build_summary_section(report: AnalysisReport) -> dict[str, Any]:
    """Build the executive summary section data dict.

    Returns a dict with keys: title, anchor, content (HTML string).
    """
    from insightml.report.narrative import (
        data_recommendations,
        ensemble_recommendation,
        executive_summary,
    )

    # Gather key metrics
    best_model = None
    best_score = None
    primary_metric = "accuracy" if report.task == "classification" else "r2"

    if report.models is not None and report.models.best is not None:
        best_model = report.models.best.name
        best_score = report.models.best.metrics.get(primary_metric)
        primary_metric = report.models.primary_metric

    readiness_score = 75
    readiness_grade = "C"
    if report.intelligence is not None:
        try:
            rs = report.intelligence.readiness
            readiness_score = rs.score
            readiness_grade = rs.grade
        except Exception:
            pass

    summary_text = executive_summary(
        task=report.task,
        target=report.target,
        n_samples=report.n_samples,
        n_features=report.n_features,
        best_model=best_model or "N/A",
        best_score=best_score or 0.0,
        primary_metric=primary_metric,
        readiness_score=readiness_score,
        readiness_grade=readiness_grade,
    )

    # Data quality recommendations
    leakage_cols: list[str] = []
    high_vif_cols: list[str] = []
    missing_pct = 0.0

    if report.intelligence is not None:
        try:
            leakage_cols = [w.column for w in report.intelligence.leakage]
        except Exception:
            pass
        try:
            vif_df = report.intelligence.vif
            if vif_df is not None and not vif_df.empty:
                high_vif_cols = list(
                    vif_df.loc[vif_df["vif"] > 10, "feature"].values
                )
        except Exception:
            pass

    recs = data_recommendations(
        readiness_score=readiness_score,
        leakage_columns=leakage_cols,
        high_vif_columns=high_vif_cols,
        missing_pct=missing_pct,
    )

    # Ensemble recommendation
    ensemble_text = ""
    if report.compare is not None:
        try:
            ea = report.compare.error_analysis
            candidates = ea.ensemble_candidates()
            pareto = report.compare.pareto_models
            ensemble_text = ensemble_recommendation(
                ensemble_candidates=candidates,
                best_model=best_model or "",
                pareto_models=pareto,
            )
        except Exception:
            pass

    # Build HTML content
    lines = [
        f'<p class="summary-text">{summary_text}</p>',
    ]
    if recs:
        lines.append("<h3>Recommendations</h3>")
        lines.append('<ul class="rec-list">')
        for rec in recs:
            lines.append(f"<li>{rec}</li>")
        lines.append("</ul>")
    if ensemble_text:
        lines.append(f'<p class="ensemble-note">{ensemble_text}</p>')

    return {
        "title": "Executive Summary",
        "anchor": "summary",
        "content": "\n".join(lines),
    }
