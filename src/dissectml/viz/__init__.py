"""Visualization infrastructure for InsightML."""

from dissectml.viz.display import detect_environment, display_html, show_in_browser
from dissectml.viz.theme import INSIGHTML_TEMPLATE, apply_theme

__all__ = [
    "INSIGHTML_TEMPLATE",
    "apply_theme",
    "detect_environment",
    "display_html",
    "show_in_browser",
]
