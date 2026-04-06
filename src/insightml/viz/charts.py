"""Chart factory functions for InsightML visualizations (Plotly only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from insightml.viz.theme import DIVERGING, QUALITATIVE, make_figure


def histogram(
    series: pd.Series,
    title: str = "",
    kde: bool = True,
    color: str | None = None,
) -> go.Figure:
    """Histogram with optional KDE overlay for a numeric series."""
    fig = make_figure(title=title or series.name or "Distribution")
    data = series.dropna()
    color = color or QUALITATIVE[0]

    fig.add_trace(go.Histogram(
        x=data,
        name="Count",
        marker_color=color,
        opacity=0.75,
        showlegend=False,
    ))

    if kde and len(data) >= 5:
        try:
            from scipy.stats import gaussian_kde
            kde_x = np.linspace(data.min(), data.max(), 200)
            kde_y = gaussian_kde(data)(kde_x)
            # Scale KDE to histogram height
            bin_count = min(50, len(data) // 5)
            counts, bin_edges = np.histogram(data, bins=bin_count)
            scale = counts.max() / (kde_y.max() + 1e-10)
            fig.add_trace(go.Scatter(
                x=kde_x, y=kde_y * scale,
                mode="lines", name="KDE",
                line={"color": QUALITATIVE[1], "width": 2},
            ))
        except ImportError:
            pass

    fig.update_layout(xaxis_title=series.name or "Value", yaxis_title="Count")
    return fig


def box_plot(
    series: pd.Series,
    title: str = "",
    color: str | None = None,
) -> go.Figure:
    """Box plot with outlier points for a numeric series."""
    fig = make_figure(title=title or series.name or "Box Plot")
    fig.add_trace(go.Box(
        y=series.dropna(),
        name=series.name or "",
        marker_color=color or QUALITATIVE[0],
        boxpoints="outliers",
        jitter=0.3,
        pointpos=-1.8,
    ))
    return fig


def frequency_bar(
    series: pd.Series,
    top_n: int = 20,
    title: str = "",
    color: str | None = None,
) -> go.Figure:
    """Horizontal frequency bar chart for a categorical series."""
    counts = series.value_counts().head(top_n)
    fig = make_figure(title=title or f"{series.name} — Top {top_n}")
    fig.add_trace(go.Bar(
        y=counts.index.astype(str),
        x=counts.values,
        orientation="h",
        marker_color=color or QUALITATIVE[0],
    ))
    fig.update_layout(
        yaxis={"autorange": "reversed"},
        xaxis_title="Count",
        height=max(300, len(counts) * 28),
    )
    return fig


def heatmap(
    matrix: pd.DataFrame,
    title: str = "",
    colorscale: str = DIVERGING,
    zmid: float | None = 0,
    annotate: bool = True,
) -> go.Figure:
    """Annotated heatmap for a correlation or contingency matrix."""
    n = len(matrix.columns)
    size = max(500, 40 * n)
    fig = make_figure(title=title)
    fig.add_trace(go.Heatmap(
        z=matrix.values,
        x=list(matrix.columns),
        y=list(matrix.index),
        colorscale=colorscale,
        zmid=zmid,
        text=matrix.round(2).astype(str).values if annotate else None,
        texttemplate="%{text}" if annotate else None,
        hoverongaps=False,
    ))
    fig.update_layout(width=size, height=size)
    return fig


def scatter(
    x: pd.Series,
    y: pd.Series,
    title: str = "",
    trendline: bool = True,
    color_by: pd.Series | None = None,
) -> go.Figure:
    """Scatter plot, optionally colored by a third variable."""
    fig = make_figure(title=title or f"{x.name} vs {y.name}")

    if color_by is not None:
        # Color by categorical group
        for group in color_by.unique():
            mask = color_by == group
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask],
                mode="markers",
                name=str(group),
                marker={"size": 5, "opacity": 0.6},
            ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker={"size": 5, "color": QUALITATIVE[0], "opacity": 0.6},
            showlegend=False,
        ))
        if trendline:
            valid = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(valid) >= 2:
                coeffs = np.polyfit(valid["x"], valid["y"], 1)
                x_line = np.linspace(valid["x"].min(), valid["x"].max(), 100)
                fig.add_trace(go.Scatter(
                    x=x_line, y=np.polyval(coeffs, x_line),
                    mode="lines",
                    name="OLS trend",
                    line={"color": QUALITATIVE[1], "width": 2, "dash": "dash"},
                ))

    fig.update_layout(xaxis_title=x.name, yaxis_title=y.name)
    return fig


def violin(
    values: pd.Series,
    groups: pd.Series,
    title: str = "",
) -> go.Figure:
    """Grouped violin plot (numeric values by categorical groups)."""
    fig = make_figure(title=title or f"{values.name} by {groups.name}")
    for i, group in enumerate(groups.unique()):
        mask = groups == group
        fig.add_trace(go.Violin(
            y=values[mask],
            name=str(group),
            marker_color=QUALITATIVE[i % len(QUALITATIVE)],
            box_visible=True,
            meanline_visible=True,
        ))
    fig.update_layout(xaxis_title=groups.name, yaxis_title=values.name)
    return fig


def gauge(
    value: float,
    title: str = "Score",
    min_val: float = 0,
    max_val: float = 100,
) -> go.Figure:
    """Gauge chart for a single numeric score (e.g., Data Readiness Score)."""
    fig = make_figure()
    # Color thresholds: red < 60, yellow < 80, green >= 80
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickwidth": 1},
            "bar": {"color": QUALITATIVE[0]},
            "steps": [
                {"range": [0, 60], "color": "#fed7d7"},
                {"range": [60, 80], "color": "#fefcbf"},
                {"range": [80, 100], "color": "#c6f6d5"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 60,
            },
        },
    ))
    fig.update_layout(height=300)
    return fig
