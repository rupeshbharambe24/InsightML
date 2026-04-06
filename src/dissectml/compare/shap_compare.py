"""SHAP-based feature importance comparison across top-N models (optional)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dissectml.battle.result import BattleResult
from dissectml.viz.theme import QUALITATIVE, make_figure

_SHAP_MAX_SAMPLES = 500  # subsample for KernelExplainer


def shap_comparison(
    battle_result: BattleResult,
    X: pd.DataFrame,
    top_n: int = 3,
    max_samples: int = _SHAP_MAX_SAMPLES,
) -> dict[str, Any]:
    """Compute and compare SHAP values for the top-N models.

    Requires ``pip install dissectml[explain]`` (shap>=0.44).

    Args:
        battle_result: BattleResult with fitted_pipeline per ModelScore.
        X: Feature DataFrame (same columns as training).
        top_n: Number of top models to explain.
        max_samples: Subsample rows for KernelExplainer (slow models).

    Returns:
        Dict with keys:
        - importance_df: DataFrame of mean |SHAP| per feature per model
        - rank_correlation: Spearman rank correlation matrix between models
        - figures: dict of model_name -> bar chart figure
    """
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install with: pip install dissectml[explain]"
        ) from exc

    scores = [s for s in battle_result.successful[:top_n] if s.fitted_pipeline is not None]
    if not scores:
        return {"importance_df": pd.DataFrame(), "rank_correlation": pd.DataFrame(), "figures": {}}

    # Subsample X
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(X), size=min(max_samples, len(X)))
    X_sub = X.iloc[idx].reset_index(drop=True)

    importance_dict: dict[str, np.ndarray] = {}
    figures: dict[str, go.Figure] = {}

    for score in scores:
        pipeline = score.fitted_pipeline
        try:
            shap_vals = _compute_shap(pipeline, X_sub, score, batch_size=max_samples)
            if shap_vals is None:
                continue
            mean_abs = np.abs(shap_vals).mean(axis=0)
            importance_dict[score.name] = mean_abs

            # Bar chart for this model
            fig = _shap_bar(X_sub.columns.tolist(), mean_abs, score.name)
            figures[score.name] = fig
        except Exception:
            continue

    if not importance_dict:
        return {"importance_df": pd.DataFrame(), "rank_correlation": pd.DataFrame(), "figures": {}}

    # Build importance DataFrame
    importance_df = pd.DataFrame(
        importance_dict, index=X_sub.columns
    ).reset_index().rename(columns={"index": "feature"})

    # Spearman rank correlation between model importances
    rank_corr = pd.DataFrame(index=importance_dict.keys(), columns=importance_dict.keys(), dtype=float)
    models = list(importance_dict.keys())
    for _i, m1 in enumerate(models):
        for _j, m2 in enumerate(models):
            from scipy.stats import spearmanr
            r, _ = spearmanr(importance_dict[m1], importance_dict[m2])
            rank_corr.loc[m1, m2] = round(float(r), 4)

    return {
        "importance_df": importance_df,
        "rank_correlation": rank_corr,
        "figures": figures,
    }


def _compute_shap(pipeline, X, score, batch_size: int = 500):
    """Dispatch to appropriate SHAP explainer based on model type."""
    import shap

    model_step = pipeline.named_steps.get("model")
    if model_step is None:
        return None

    # Transform X through preprocessor only
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is not None:
        try:
            X_transformed = preprocessor.transform(X)
        except Exception:
            return None
    else:
        X_transformed = X.values

    # Choose explainer
    model_type = type(model_step).__name__
    try:
        if any(k in model_type for k in ("Forest", "Tree", "Boosting", "XGB", "LGBM", "CatBoost", "Bagging")):
            explainer = shap.TreeExplainer(model_step)
            shap_vals = explainer.shap_values(X_transformed)
            # For classifiers, shap_values may return list (one per class)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
        elif any(k in model_type for k in ("Linear", "Ridge", "Lasso", "Logistic", "SGD")):
            explainer = shap.LinearExplainer(model_step, X_transformed)
            shap_vals = explainer.shap_values(X_transformed)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
        else:
            # KernelExplainer — expensive, use subsample
            background = shap.kmeans(X_transformed, min(50, len(X_transformed)))
            explainer = shap.KernelExplainer(model_step.predict, background)
            shap_vals = explainer.shap_values(X_transformed[:min(100, len(X_transformed))], silent=True)

        return np.array(shap_vals)
    except Exception:
        return None


def _shap_bar(feature_names: list[str], mean_abs_shap: np.ndarray, model_name: str) -> go.Figure:
    """Horizontal bar chart of mean |SHAP| values."""
    order = np.argsort(mean_abs_shap)[-20:]  # top 20
    names = [feature_names[i] for i in order]
    vals = mean_abs_shap[order].tolist()

    fig = make_figure(title=f"SHAP Feature Importance — {model_name}")
    fig.add_trace(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=QUALITATIVE[0],
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Mean |SHAP value|",
        height=max(300, len(names) * 22 + 100),
    )
    return fig
