"""InsightML Plotly theme and template."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

PALETTE = {
    "primary": "#4c78a8",
    "secondary": "#f58518",
    "success": "#54a24b",
    "warning": "#e45756",
    "info": "#72b7b2",
    "light": "#f8f9fa",
    "dark": "#2d3748",
    "muted": "#888888",
}

# Sequential and diverging scales
SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"
QUALITATIVE = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756",
    "#72b7b2", "#b279a2", "#ff9da6", "#9d755d",
    "#bab0ac", "#eeca3b",
]

# ---------------------------------------------------------------------------
# Plotly layout template
# ---------------------------------------------------------------------------

_LAYOUT = go.Layout(
    paper_bgcolor="white",
    plot_bgcolor="#f8f9fa",
    font={"family": "Inter, system-ui, sans-serif", "size": 12, "color": "#2d3748"},
    title={"font": {"size": 16, "color": "#2d3748"}, "x": 0.01, "xanchor": "left"},
    colorway=QUALITATIVE,
    margin={"t": 50, "r": 20, "b": 50, "l": 60},
    legend={
        "bgcolor": "rgba(255,255,255,0.9)",
        "bordercolor": "#e2e8f0",
        "borderwidth": 1,
    },
    xaxis={
        "gridcolor": "#e2e8f0",
        "linecolor": "#cbd5e0",
        "zerolinecolor": "#cbd5e0",
    },
    yaxis={
        "gridcolor": "#e2e8f0",
        "linecolor": "#cbd5e0",
        "zerolinecolor": "#cbd5e0",
    },
)

INSIGHTML_TEMPLATE = go.layout.Template(layout=_LAYOUT)

# Register the template with Plotly so `template="dissectml"` works
pio.templates["dissectml"] = INSIGHTML_TEMPLATE


def apply_theme() -> None:
    """Set InsightML as the default Plotly template globally."""
    pio.templates.default = "dissectml"


def make_figure(title: str = "", **layout_kwargs) -> go.Figure:
    """Create a blank Figure with the InsightML theme applied.

    Args:
        title: Chart title.
        **layout_kwargs: Additional layout overrides.

    Returns:
        A new go.Figure with InsightML template.
    """
    fig = go.Figure()
    fig.update_layout(template="dissectml", title=title, **layout_kwargs)
    return fig
