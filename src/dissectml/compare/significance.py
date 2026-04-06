"""Statistical significance tests for model comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dissectml.battle.result import BattleResult
from dissectml.viz.theme import make_figure


def mcnemar_matrix(
    battle_result: BattleResult,
    y_true: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Compute McNemar's test p-values for all model pairs (classification).

    For each pair (A, B):
    - n01 = samples where A wrong, B right
    - n10 = samples where A right, B wrong
    - chi2 = (|n01 - n10| - 1)² / (n01 + n10) with continuity correction

    Args:
        battle_result: BattleResult from classification run.
        y_true: Ground-truth labels.

    Returns:
        Dict with keys: p_matrix (DataFrame), figure (heatmap).
    """
    from scipy.stats import chi2

    scores = [s for s in battle_result.successful if s.oof_predictions is not None]
    if len(scores) < 2:
        return {"p_matrix": pd.DataFrame(), "figure": make_figure("Not enough models")}

    y_arr = np.asarray(y_true)
    names = [s.name for s in scores]
    n = len(names)
    p_matrix = np.ones((n, n))

    for i, s1 in enumerate(scores):
        for j, s2 in enumerate(scores):
            if i >= j:
                continue
            mask = ~(np.isnan(s1.oof_predictions) | np.isnan(s2.oof_predictions))
            if mask.sum() < 10:
                continue
            try:
                y_sub = y_arr[mask]
                p1 = s1.oof_predictions[mask].round()
                p2 = s2.oof_predictions[mask].round()
                n01 = int(np.sum((p1 != y_sub) & (p2 == y_sub)))
                n10 = int(np.sum((p1 == y_sub) & (p2 != y_sub)))
                denom = n01 + n10
                if denom == 0:
                    p_val = 1.0
                else:
                    stat = max(0, (abs(n01 - n10) - 1) ** 2 / denom)
                    p_val = float(1 - chi2.cdf(stat, df=1))
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val
            except Exception:
                pass

    p_df = pd.DataFrame(p_matrix, index=names, columns=names).round(4)

    fig = make_figure(title="McNemar p-value Matrix (lower = significantly different)")
    fig.add_trace(go.Heatmap(
        z=p_matrix, x=names, y=names,
        colorscale="RdYlGn_r", zmin=0, zmax=0.1,
        text=p_df.values,
        texttemplate="%{text:.3f}",
        hovertemplate="Model A: %{y}<br>Model B: %{x}<br>p-value: %{z:.4f}<extra></extra>",
        colorbar={"title": "p-value"},
    ))
    fig.update_layout(height=max(350, n * 50 + 100))
    return {"p_matrix": p_df, "figure": fig}


def corrected_ttest_matrix(
    battle_result: BattleResult,
    y_true: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Corrected paired t-test (Nadeau & Bengio) for all model pairs.

    Corrects variance for overlapping training sets in k-fold CV:
        var_corrected = (1/k + n_test/n_train) * var(differences)

    Args:
        battle_result: BattleResult (classification or regression).
        y_true: Ground-truth labels/values.

    Returns:
        Dict with keys: p_matrix (DataFrame), figure (heatmap).
    """
    from scipy.stats import t as t_dist

    scores = [s for s in battle_result.successful if s.oof_predictions is not None]
    if len(scores) < 2:
        return {"p_matrix": pd.DataFrame(), "figure": make_figure("Not enough models")}

    y_arr = np.asarray(y_true, dtype=float)
    task = battle_result.task
    k = battle_result.cv_folds
    n = len(y_arr)
    n_test = n // k
    n_train = n - n_test
    correction_factor = 1 / k + n_test / max(n_train, 1)

    names = [s.name for s in scores]
    p_matrix = np.ones((len(names), len(names)))

    for i, s1 in enumerate(scores):
        for j, s2 in enumerate(scores):
            if i >= j:
                continue
            mask = ~(np.isnan(s1.oof_predictions) | np.isnan(s2.oof_predictions))
            if mask.sum() < 10:
                continue
            try:
                if task == "classification":
                    e1 = (s1.oof_predictions[mask].round() != y_arr[mask]).astype(float)
                    e2 = (s2.oof_predictions[mask].round() != y_arr[mask]).astype(float)
                else:
                    e1 = (s1.oof_predictions[mask] - y_arr[mask]) ** 2
                    e2 = (s2.oof_predictions[mask] - y_arr[mask]) ** 2

                diffs = e1 - e2
                mean_diff = float(np.mean(diffs))
                var_diff = float(np.var(diffs, ddof=1))
                var_corrected = correction_factor * var_diff
                if var_corrected <= 0:
                    p_val = 1.0
                else:
                    t_stat = mean_diff / np.sqrt(var_corrected / k)
                    df = k - 1
                    p_val = float(2 * t_dist.sf(abs(t_stat), df=df))

                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val
            except Exception:
                pass

    p_df = pd.DataFrame(p_matrix, index=names, columns=names).round(4)

    fig = make_figure(title="Corrected Paired t-test p-value Matrix")
    fig.add_trace(go.Heatmap(
        z=p_matrix, x=names, y=names,
        colorscale="RdYlGn_r", zmin=0, zmax=0.1,
        text=p_df.values,
        texttemplate="%{text:.3f}",
        hovertemplate="Model A: %{y}<br>Model B: %{x}<br>p-value: %{z:.4f}<extra></extra>",
        colorbar={"title": "p-value"},
    ))
    fig.update_layout(height=max(350, len(names) * 50 + 100))
    return {"p_matrix": p_df, "figure": fig}
