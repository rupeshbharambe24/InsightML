"""Pareto front — accuracy vs training speed."""

from __future__ import annotations

import plotly.graph_objects as go

from dissectml.battle.result import BattleResult
from dissectml.viz.theme import QUALITATIVE, make_figure


def pareto_front(battle_result: BattleResult) -> go.Figure:
    """Interactive Pareto front: primary metric vs training time.

    Pareto-optimal models (not dominated on both axes) are highlighted
    and connected by a step line.

    Args:
        battle_result: Output of BattleRunner.

    Returns:
        Plotly Figure.
    """
    scores = battle_result.successful
    if not scores:
        return make_figure("No models to plot")

    primary = battle_result.primary_metric
    names = [s.name for s in scores]
    metric_vals = [s.metrics.get(primary, 0.0) for s in scores]
    train_times = [s.train_time for s in scores]

    # Compute Pareto front (maximise metric, minimise time)
    pareto_mask = _compute_pareto(metric_vals, train_times)

    # Pareto points sorted by time for step line
    pareto_idx = [i for i, m in enumerate(pareto_mask) if m]
    pareto_idx.sort(key=lambda i: train_times[i])
    pareto_times = [train_times[i] for i in pareto_idx]
    pareto_metrics = [metric_vals[i] for i in pareto_idx]

    fig = make_figure(title=f"Pareto Front — {primary} vs Training Time")

    # Non-Pareto models
    non_pareto = [i for i in range(len(scores)) if not pareto_mask[i]]
    if non_pareto:
        fig.add_trace(go.Scatter(
            x=[train_times[i] for i in non_pareto],
            y=[metric_vals[i] for i in non_pareto],
            mode="markers",
            marker={"color": "lightgray", "size": 10, "symbol": "circle"},
            text=[names[i] for i in non_pareto],
            hovertemplate="%{text}<br>Time: %{x:.2f}s<br>Score: %{y:.4f}<extra></extra>",
            name="Sub-optimal",
        ))

    # Pareto step line
    if pareto_times:
        fig.add_trace(go.Scatter(
            x=pareto_times, y=pareto_metrics,
            mode="lines",
            line={"color": "#4c78a8", "width": 1.5, "dash": "dot"},
            showlegend=False,
        ))

    # Pareto models
    if pareto_idx:
        fig.add_trace(go.Scatter(
            x=[train_times[i] for i in pareto_idx],
            y=[metric_vals[i] for i in pareto_idx],
            mode="markers+text",
            marker={"color": QUALITATIVE[0], "size": 14, "symbol": "star",
                        "line": {"color": "black", "width": 1}},
            text=[names[i] for i in pareto_idx],
            textposition="top center",
            textfont={"size": 10},
            hovertemplate="%{text}<br>Time: %{x:.2f}s<br>Score: %{y:.4f}<extra></extra>",
            name="Pareto optimal",
        ))

    fig.update_layout(
        xaxis_title="Training Time (s)",
        yaxis_title=primary,
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def get_pareto_models(battle_result: BattleResult) -> list[str]:
    """Return names of Pareto-optimal models (not dominated)."""
    scores = battle_result.successful
    if not scores:
        return []
    primary = battle_result.primary_metric
    metric_vals = [s.metrics.get(primary, 0.0) for s in scores]
    train_times = [s.train_time for s in scores]
    mask = _compute_pareto(metric_vals, train_times)
    return [scores[i].name for i, m in enumerate(mask) if m]


def _compute_pareto(metrics: list[float], times: list[float]) -> list[bool]:
    """Return boolean mask; True = Pareto-optimal (maximise metric, minimise time)."""
    n = len(metrics)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is at least as good on both and strictly better on one
            if (metrics[j] >= metrics[i] and times[j] <= times[i]
                    and (metrics[j] > metrics[i] or times[j] < times[i])):
                dominated[i] = True
                break
    return [not d for d in dominated]
