"""Stage 5: Report generation — AnalysisReport with HTML export."""

from dissectml.report.builder import AnalysisReport
from dissectml.report.html_renderer import render_html_report
from dissectml.report.narrative import (
    data_recommendations,
    ensemble_recommendation,
    executive_summary,
    model_narrative,
)

__all__ = [
    "AnalysisReport",
    "render_html_report",
    "executive_summary",
    "model_narrative",
    "data_recommendations",
    "ensemble_recommendation",
]
