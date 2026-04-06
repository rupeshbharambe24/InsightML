"""TargetAnalysis — class balance, distribution, and feature-target relationships."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import shapiro

from dissectml._types import ColumnType, TaskType
from dissectml.core.validators import infer_column_type, infer_task
from dissectml.eda._base import BaseAnalysisModule
from dissectml.viz.theme import QUALITATIVE, make_figure


class TargetAnalysis(BaseAnalysisModule):
    """Target-specific analysis.

    Only available when a ``target`` column is provided to ``explore()``.

    **Classification**:
    - Class distribution, counts, percentages
    - Imbalance ratio and severity label
    - Recommendation: SMOTE / class_weight if severe

    **Regression**:
    - Mean, median, std, skew, kurtosis
    - Shapiro-Wilk normality test
    - Log/sqrt transform recommendation if highly skewed

    **Feature-target relationships** (top-N features):
    - Numeric features vs target: scatter (regression) or violin (classification)
    - Categorical features vs target: grouped bar

    Access::

        eda.target.balance()                 # classification only
        eda.target.distribution()            # regression only
        eda.target.feature_target_plots()    # top-N feature vs target figures
        eda.target.show()
    """

    def _compute(self) -> None:
        df = self._df
        config = self._config
        target = self._target

        if target is None:
            raise ValueError("TargetAnalysis requires a target column.")

        target_series = df[target]
        task = infer_task(target_series)

        if task == TaskType.CLASSIFICATION:
            result = _classification_analysis(target_series, config.significance_level)
        else:
            result = _regression_analysis(target_series)

        result["task"] = task.value
        result["target_col"] = target

        # Feature importance proxy (correlations / MI)
        feature_cols = [c for c in df.columns if c != target]
        feature_scores = _score_features(df, feature_cols, target_series, task, config)
        result["feature_scores"] = feature_scores

        self._results = result

    def _build_figures(self) -> dict[str, go.Figure]:
        df = self._df
        target = self._target
        figs: dict[str, go.Figure] = {}
        task = self._results.get("task")

        if task == TaskType.CLASSIFICATION.value:
            figs["class_distribution"] = _class_dist_fig(
                df[target], self._results
            )
        else:
            figs["target_distribution"] = _target_dist_fig(df[target])

        # Feature-target plots for top-5 features
        top_features = [
            f["column"] for f in self._results.get("feature_scores", [])[:5]
        ]
        config = self._config
        for feat_col in top_features:
            if feat_col not in df.columns:
                continue
            feat_type = infer_column_type(df[feat_col], config)
            fig = _feature_target_fig(
                df[feat_col], df[target], feat_type, task
            )
            if fig is not None:
                figs[f"feat_{feat_col}"] = fig

        return figs

    def summary(self) -> str:
        task = self._results.get("task")
        target = self._results.get("target_col", "target")
        if task == TaskType.CLASSIFICATION.value:
            n_classes = self._results.get("n_classes", "?")
            severity = self._results.get("imbalance_severity", "unknown")
            return (
                f"Target '{target}' — {n_classes}-class classification. "
                f"Class imbalance: {severity}."
            )
        else:
            mean = self._results.get("mean", "?")
            skew = self._results.get("skewness", "?")
            return (
                f"Target '{target}' — regression. "
                f"Mean={mean:.3g}, skewness={skew:.3g}."
                if isinstance(mean, float) and isinstance(skew, float)
                else f"Target '{target}' — regression."
            )

    # --- Public accessors ---

    def balance(self) -> dict[str, Any]:
        """Class balance info (classification only)."""
        self._ensure_computed()
        return {k: v for k, v in self._results.items()
                if k in ("class_counts", "class_pct", "imbalance_ratio",
                         "imbalance_severity", "recommendation", "n_classes")}

    def distribution(self) -> dict[str, Any]:
        """Target distribution stats (regression only)."""
        self._ensure_computed()
        return {k: v for k, v in self._results.items()
                if k in ("mean", "median", "std", "skewness", "kurtosis",
                         "is_normal", "transform_recommendation")}

    def feature_target_plots(self, top_n: int = 10) -> dict[str, go.Figure]:
        """Return feature-vs-target figures for top-N features."""
        self._ensure_computed()
        self._ensure_computed()  # figures already built
        return {k: v for k, v in self._figures.items()
                if k.startswith("feat_")}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classification_analysis(target: pd.Series, alpha: float) -> dict[str, Any]:
    vc = target.value_counts()
    n_classes = int(target.nunique())
    total = len(target.dropna())
    class_pct = (vc / total * 100).round(2).to_dict()
    minority_ratio = float(vc.min() / vc.max()) if vc.max() > 0 else 1.0

    if minority_ratio > 0.8:
        severity = "balanced"
    elif minority_ratio > 0.4:
        severity = "mild"
    elif minority_ratio > 0.2:
        severity = "moderate"
    else:
        severity = "severe"

    rec = None
    if severity in ("moderate", "severe"):
        rec = (
            f"Consider SMOTE oversampling, class_weight='balanced', "
            f"or threshold tuning (minority_ratio={minority_ratio:.2f})."
        )

    return {
        "n_classes": n_classes,
        "class_counts": {str(k): int(v) for k, v in vc.items()},
        "class_pct": {str(k): float(v) for k, v in class_pct.items()},
        "imbalance_ratio": round(minority_ratio, 4),
        "imbalance_severity": severity,
        "recommendation": rec,
    }


def _regression_analysis(target: pd.Series) -> dict[str, Any]:
    data = target.dropna()
    skewness = float(data.skew())
    kurtosis = float(data.kurtosis())

    # Normality (Shapiro on sample)
    sample = data if len(data) <= 5000 else data.sample(5000, random_state=42)
    try:
        _, sw_p = shapiro(sample)
        is_normal = float(sw_p) >= 0.05
    except Exception:
        is_normal = None

    transform_rec = None
    if abs(skewness) > 1:
        transform_rec = (
            "Log transform recommended (log1p)" if skewness > 1
            else "Reflection + log transform recommended (negative skew)"
        )
    elif abs(skewness) > 0.5:
        transform_rec = "Sqrt transform may help reduce moderate skewness."

    return {
        "mean": round(float(data.mean()), 4),
        "median": round(float(data.median()), 4),
        "std": round(float(data.std()), 4),
        "min": round(float(data.min()), 4),
        "max": round(float(data.max()), 4),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
        "is_normal": is_normal,
        "transform_recommendation": transform_rec,
    }


def _score_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: pd.Series,
    task: TaskType,
    config,
) -> list[dict[str, Any]]:
    # For classification, encode target to integers for correlation
    if task == TaskType.CLASSIFICATION:
        target_enc = pd.Categorical(target).codes.astype(float)
        target_enc = pd.Series(target_enc, index=target.index)
        target_enc[target.isna()] = np.nan
    else:
        target_enc = pd.to_numeric(target, errors="coerce")

    scores = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        mask = df[col].notna() & target_enc.notna()
        if mask.sum() < 5:
            continue
        try:
            corr = abs(float(df.loc[mask, col].corr(target_enc[mask])))
            scores.append({"column": col, "abs_correlation": round(corr, 4)})
        except Exception:
            pass
    scores.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return scores


def _class_dist_fig(target: pd.Series, results: dict) -> go.Figure:
    vc = target.value_counts()
    severity = results.get("imbalance_severity", "")
    color_map = {"balanced": "#54a24b", "mild": "#4c78a8",
                 "moderate": "#f58518", "severe": "#e45756"}
    bar_color = color_map.get(severity, "#4c78a8")
    fig = make_figure(title=f"Class Distribution (imbalance: {severity})")
    fig.add_trace(go.Bar(
        x=[str(v) for v in vc.index],
        y=vc.values,
        marker_color=bar_color,
        text=vc.values, textposition="outside",
    ))
    fig.update_layout(xaxis_title="Class", yaxis_title="Count",
                      showlegend=False, height=350)
    return fig


def _target_dist_fig(target: pd.Series) -> go.Figure:
    data = target.dropna()
    fig = make_figure(title=f"{target.name} — Target Distribution")
    fig.add_trace(go.Histogram(
        x=data, nbinsx=40,
        marker_color=QUALITATIVE[0], opacity=0.75,
        name="Count",
    ))
    try:
        from scipy.stats import gaussian_kde
        kde_x = np.linspace(data.min(), data.max(), 200)
        kde_y = gaussian_kde(data)(kde_x)
        counts, _ = np.histogram(data, bins=40)
        scale = counts.max() / (kde_y.max() + 1e-10)
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y * scale,
            mode="lines", name="KDE",
            line={"color": QUALITATIVE[1], "width": 2},
        ))
    except Exception:
        pass
    fig.update_layout(xaxis_title=target.name, yaxis_title="Count", height=350)
    return fig


def _feature_target_fig(
    feature: pd.Series, target: pd.Series,
    feat_type: ColumnType, task: str,
) -> go.Figure | None:
    mask = feature.notna() & target.notna()
    if mask.sum() < 5:
        return None

    if feat_type == ColumnType.NUMERIC:
        if task == TaskType.REGRESSION.value:
            fig = make_figure(title=f"{feature.name} vs {target.name}")
            fig.add_trace(go.Scatter(
                x=feature[mask], y=target[mask],
                mode="markers",
                marker={"color": QUALITATIVE[0], "size": 4, "opacity": 0.5},
                showlegend=False,
            ))
            fig.update_layout(xaxis_title=feature.name, yaxis_title=target.name,
                              height=300)
            return fig
        else:
            fig = make_figure(title=f"{feature.name} by {target.name}")
            for i, cls in enumerate(target[mask].unique()):
                vals = feature[mask & (target == cls)]
                fig.add_trace(go.Violin(
                    y=vals, name=str(cls),
                    marker_color=QUALITATIVE[i % len(QUALITATIVE)],
                    box_visible=True, meanline_visible=True,
                ))
            fig.update_layout(xaxis_title=target.name,
                              yaxis_title=feature.name, height=300)
            return fig

    if feat_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
        if task == TaskType.CLASSIFICATION.value:
            ct = pd.crosstab(feature[mask], target[mask], normalize="index") * 100
            fig = make_figure(title=f"{feature.name} vs {target.name} (%)")
            for i, col in enumerate(ct.columns):
                fig.add_trace(go.Bar(
                    name=str(col), x=[str(v) for v in ct.index],
                    y=ct[col].values,
                    marker_color=QUALITATIVE[i % len(QUALITATIVE)],
                ))
            fig.update_layout(barmode="stack", height=300,
                              xaxis_title=feature.name, yaxis_title="%")
            return fig
        else:
            top_cats = feature[mask].value_counts().head(8).index
            mask2 = mask & feature.isin(top_cats)
            fig = make_figure(title=f"{target.name} by {feature.name}")
            for i, cat in enumerate(top_cats):
                vals = target[mask2 & (feature == cat)]
                fig.add_trace(go.Box(
                    y=vals, name=str(cat),
                    marker_color=QUALITATIVE[i % len(QUALITATIVE)],
                    boxpoints="outliers",
                ))
            fig.update_layout(xaxis_title=feature.name,
                              yaxis_title=target.name, height=300)
            return fig
    return None
