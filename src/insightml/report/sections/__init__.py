"""Section builders for the InsightML HTML report."""

from insightml.report.sections.summary_section import build_summary_section
from insightml.report.sections.eda_section import build_eda_section
from insightml.report.sections.intelligence_section import build_intelligence_section
from insightml.report.sections.battle_section import build_battle_section
from insightml.report.sections.compare_section import build_compare_section

__all__ = [
    "build_summary_section",
    "build_eda_section",
    "build_intelligence_section",
    "build_battle_section",
    "build_compare_section",
]
