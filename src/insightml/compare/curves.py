"""ROC, PR, confusion matrix, residual, and calibration curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from insightml.battle.result import BattleResult
from insightml.viz.theme import QUALITATIVE, make_figure

# ---------------------------------------------------------------------------
# Classification curves
# ---------------------------------------------------------------------------


def roc_curves(
    battle_result: BattleResult,
    y_true: pd.Series | np.ndarray,
    n_models: int = 10,
) -> go.Figure:
    """Plot ROC curves for all successful classification models.

    Args:
        battle_result: Output of BattleRunner (classification).
        y_true: True labels (same order as training data, OOF aligned).
        n_models: Max number of models to plot.

    Returns:
        Plotly Figure with all ROC curves overlaid.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import LabelBinarizer

    fig = make_figure(title="ROC Curves (OOF)")
    scores = battle_result.successful[:n_models]
    y_arr = np.asarray(y_true)
    classes = np.unique(y_arr[~pd.isna(y_arr)])
    binary = len(classes) == 2

    for i, score in enumerate(scores):
        if score.oof_probabilities is None:
            continue
        probs = score.oof_probabilities
        mask = ~np.isnan(probs[:, 0])
        if mask.sum() < 10:
            continue
        try:
            if binary:
                fpr, tpr, _ = roc_curve(y_arr[mask], probs[mask, 1])
                auc = roc_auc_score(y_arr[mask], probs[mask, 1])
                fig.add_trace(go.Scatter(
                    x=list(fpr), y=list(tpr),
                    mode="lines", name=f"{score.name} (AUC={auc:.3f})",
                    line={"color": QUALITATIVE[i % len(QUALITATIVE)], "width": 2},
                ))
            else:
                lb = LabelBinarizer().fit(y_arr[mask])
                y_bin = lb.transform(y_arr[mask])
                for j, cls in enumerate(lb.classes_):
                    fpr, tpr, _ = roc_curve(y_bin[:, j], probs[mask, j])
                    auc = roc_auc_score(y_bin[:, j], probs[mask, j])
                    fig.add_trace(go.Scatter(
                        x=list(fpr), y=list(tpr),
                        mode="lines",
                        name=f"{score.name} cls={cls} (AUC={auc:.3f})",
                        line={"color": QUALITATIVE[(i * len(lb.classes_) + j) % len(QUALITATIVE)], "width": 1.5},
                    ))
        except Exception:
            continue

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line={"color": "gray", "width": 1, "dash": "dash"}, showlegend=False,
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        xaxis={"range": [0, 1]}, yaxis={"range": [0, 1.02]},
        height=500,
    )
    return fig


def pr_curves(
    battle_result: BattleResult,
    y_true: pd.Series | np.ndarray,
    n_models: int = 10,
) -> go.Figure:
    """Plot Precision-Recall curves for all successful classification models."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig = make_figure(title="Precision-Recall Curves (OOF)")
    scores = battle_result.successful[:n_models]
    y_arr = np.asarray(y_true)
    classes = np.unique(y_arr[~pd.isna(y_arr)])
    binary = len(classes) == 2

    for i, score in enumerate(scores):
        if score.oof_probabilities is None:
            continue
        probs = score.oof_probabilities
        mask = ~np.isnan(probs[:, 0])
        if mask.sum() < 10 or not binary:
            continue
        try:
            prec, rec, _ = precision_recall_curve(y_arr[mask], probs[mask, 1])
            ap = average_precision_score(y_arr[mask], probs[mask, 1])
            fig.add_trace(go.Scatter(
                x=list(rec), y=list(prec),
                mode="lines", name=f"{score.name} (AP={ap:.3f})",
                line={"color": QUALITATIVE[i % len(QUALITATIVE)], "width": 2},
            ))
        except Exception:
            continue

    fig.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        xaxis={"range": [0, 1]}, yaxis={"range": [0, 1.02]},
        height=500,
    )
    return fig


def confusion_matrices(
    battle_result: BattleResult,
    y_true: pd.Series | np.ndarray,
    n_models: int = 6,
) -> go.Figure:
    """Grid of confusion-matrix heatmaps for top-N models."""
    from sklearn.metrics import confusion_matrix

    scores = [s for s in battle_result.successful[:n_models] if s.oof_predictions is not None]
    if not scores:
        return make_figure(title="No OOF predictions available")

    n = len(scores)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[s.name for s in scores],
    )
    y_arr = np.asarray(y_true)
    classes = np.unique(y_arr[~pd.isna(y_arr)])

    for idx, score in enumerate(scores):
        row = idx // cols + 1
        col = idx % cols + 1
        preds = score.oof_predictions
        mask = ~np.isnan(preds)
        if mask.sum() < 5:
            continue
        try:
            cm = confusion_matrix(y_arr[mask], preds[mask].round(), labels=classes)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            labels = [str(c) for c in classes]
            fig.add_trace(
                go.Heatmap(
                    z=cm_norm, x=labels, y=labels,
                    colorscale="Blues", showscale=False,
                    text=cm, texttemplate="%{text}",
                    hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>",
                ),
                row=row, col=col,
            )
        except Exception:
            continue

    fig.update_layout(height=max(350, rows * 300), title_text="Confusion Matrices (OOF)")
    return fig


# ---------------------------------------------------------------------------
# Regression curves
# ---------------------------------------------------------------------------


def residual_plots(
    battle_result: BattleResult,
    y_true: pd.Series | np.ndarray,
    n_models: int = 6,
) -> go.Figure:
    """Grid of residual plots (predicted vs residual) for top-N regression models."""
    scores = [s for s in battle_result.successful[:n_models] if s.oof_predictions is not None]
    if not scores:
        return make_figure(title="No OOF predictions available")

    n = len(scores)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[s.name for s in scores])
    y_arr = np.asarray(y_true, dtype=float)

    for idx, score in enumerate(scores):
        row = idx // cols + 1
        col = idx % cols + 1
        preds = score.oof_predictions.astype(float)
        mask = ~(np.isnan(preds) | np.isnan(y_arr))
        if mask.sum() < 5:
            continue
        residuals = y_arr[mask] - preds[mask]
        fig.add_trace(
            go.Scatter(
                x=preds[mask].tolist(), y=residuals.tolist(),
                mode="markers",
                marker={"color": QUALITATIVE[idx % len(QUALITATIVE)], "size": 4, "opacity": 0.5},
                showlegend=False,
            ),
            row=row, col=col,
        )
        # Zero line
        fig.add_hline(y=0, line={"color": "gray", "dash": "dash"}, row=row, col=col)

    fig.update_layout(height=max(350, rows * 300), title_text="Residual Plots (OOF)")
    return fig


def actual_vs_predicted(
    battle_result: BattleResult,
    y_true: pd.Series | np.ndarray,
    n_models: int = 6,
) -> go.Figure:
    """Grid of actual vs predicted scatter plots for top-N regression models."""
    scores = [s for s in battle_result.successful[:n_models] if s.oof_predictions is not None]
    if not scores:
        return make_figure(title="No OOF predictions available")

    n = len(scores)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[s.name for s in scores])
    y_arr = np.asarray(y_true, dtype=float)

    for idx, score in enumerate(scores):
        row = idx // cols + 1
        col = idx % cols + 1
        preds = score.oof_predictions.astype(float)
        mask = ~(np.isnan(preds) | np.isnan(y_arr))
        if mask.sum() < 5:
            continue
        fig.add_trace(
            go.Scatter(
                x=y_arr[mask].tolist(), y=preds[mask].tolist(),
                mode="markers",
                marker={"color": QUALITATIVE[idx % len(QUALITATIVE)], "size": 4, "opacity": 0.5},
                showlegend=False,
            ),
            row=row, col=col,
        )
        # Perfect prediction line
        mn = float(min(y_arr[mask].min(), preds[mask].min()))
        mx = float(max(y_arr[mask].max(), preds[mask].max()))
        fig.add_trace(
            go.Scatter(
                x=[mn, mx], y=[mn, mx], mode="lines",
                line={"color": "gray", "dash": "dash"}, showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_layout(height=max(350, rows * 300), title_text="Actual vs Predicted (OOF)")
    return fig


# ---------------------------------------------------------------------------
# Metric comparison bar chart
# ---------------------------------------------------------------------------


def metric_bar_chart(
    battle_result: BattleResult,
    metric: str | None = None,
    n_models: int = 15,
) -> go.Figure:
    """Horizontal bar chart comparing models by a single metric."""
    metric = metric or battle_result.primary_metric
    scores = battle_result.successful[:n_models]
    if not scores:
        return make_figure(title="No models")

    names = [s.name for s in scores]
    vals = [s.metrics.get(metric, 0) for s in scores]
    errs = [s.metrics_std.get(metric, 0) for s in scores]

    fig = make_figure(title=f"Model Comparison — {metric}")
    fig.add_trace(go.Bar(
        y=names, x=vals,
        orientation="h",
        error_x={"type": "data", "array": errs, "visible": True},
        marker_color=[QUALITATIVE[i % len(QUALITATIVE)] for i in range(len(names))],
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title=metric, yaxis_title="Model",
        height=max(300, len(names) * 35 + 100),
        yaxis={"autorange": "reversed"},
    )
    return fig
