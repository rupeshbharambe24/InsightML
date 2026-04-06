"""Data Readiness Score — 0-100 composite score with breakdown."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from insightml.viz.theme import QUALITATIVE, make_figure

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

_MAX_MISSING_PENALTY = 25
_MAX_IMBALANCE_PENALTY = 20
_MAX_MULTICOLLINEARITY_PENALTY = 15
_MAX_OUTLIER_PENALTY = 10
_CONSTANT_FEATURE_PENALTY = 2   # per column

_MAX_SAMPLE_BONUS = 10
_MAX_DIVERSITY_BONUS = 5

_GRADE_THRESHOLDS = [("A", 90), ("B", 80), ("C", 70), ("D", 60), ("F", 0)]


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReadinessResult:
    """Result of a data readiness assessment.

    Attributes:
        score: Overall score [0, 100].
        grade: Letter grade A/B/C/D/F.
        breakdown: Dict of category -> {penalty/bonus, description}.
        recommendations: Actionable text recommendations.
        n_samples: Number of rows.
        n_features: Number of feature columns.
    """

    score: float
    grade: str
    breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    n_samples: int = 0
    n_features: int = 0

    def summary(self) -> str:
        rec_str = "\n".join(f"  • {r}" for r in self.recommendations) if self.recommendations else "  None"
        return (
            f"Data Readiness: {self.score:.0f}/100 (Grade {self.grade})\n"
            f"Samples: {self.n_samples}, Features: {self.n_features}\n"
            f"Recommendations:\n{rec_str}"
        )

    def gauge_figure(self) -> go.Figure:
        """Gauge chart showing the readiness score."""
        color = (
            "#54a24b" if self.score >= 80
            else "#f58518" if self.score >= 60
            else "#e45756"
        )
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=self.score,
            title={"text": f"Data Readiness Score (Grade {self.grade})", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 60], "color": "#fde0d9"},
                    {"range": [60, 80], "color": "#fef3d0"},
                    {"range": [80, 100], "color": "#d9f0d3"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": self.score,
                },
            },
        ))
        fig.update_layout(height=300, margin={"t": 50, "b": 10, "l": 20, "r": 20})
        return fig

    def waterfall_figure(self) -> go.Figure:
        """Waterfall chart showing score breakdown."""
        measures = ["absolute"]
        x_labels = ["Base (100)"]
        y_values = [100.0]

        for cat, info in self.breakdown.items():
            delta = info.get("delta", 0)
            if delta != 0:
                measures.append("relative")
                x_labels.append(cat.replace("_", " ").title())
                y_values.append(delta)

        measures.append("total")
        x_labels.append(f"Score ({self.score:.0f})")
        y_values.append(0)

        colors = []
        for m, v in zip(measures, y_values):
            if m == "absolute":
                colors.append(QUALITATIVE[0])
            elif m == "total":
                colors.append("#4c78a8")
            elif v < 0:
                colors.append("#e45756")
            else:
                colors.append("#54a24b")

        fig = make_figure(title="Readiness Score Breakdown")
        fig.add_trace(go.Waterfall(
            orientation="v",
            measure=measures,
            x=x_labels,
            y=y_values,
            connector={"line": {"color": "rgb(63,63,63)"}},
            decreasing={"marker": {"color": "#e45756"}},
            increasing={"marker": {"color": "#54a24b"}},
            totals={"marker": {"color": "#4c78a8"}},
            text=[f"{v:+.0f}" if m == "relative" else f"{abs(v):.0f}" for m, v in zip(measures, y_values)],
            textposition="outside",
        ))
        fig.update_layout(height=400, showlegend=False)
        return fig

    def _repr_html_(self) -> str:
        rows = "".join(
            f"<tr><td>{cat.replace('_',' ').title()}</td>"
            f"<td>{info.get('delta', 0):+.1f}</td>"
            f"<td>{info.get('description', '')}</td></tr>"
            for cat, info in self.breakdown.items()
        )
        return (
            f"<h3>Data Readiness: {self.score:.0f}/100 (Grade <b>{self.grade}</b>)</h3>"
            f"<table border='1' style='border-collapse:collapse;font-size:13px'>"
            f"<tr><th>Category</th><th>Delta</th><th>Details</th></tr>"
            f"{rows}</table>"
            + (
                "<ul>" + "".join(f"<li>{r}</li>" for r in self.recommendations) + "</ul>"
                if self.recommendations else ""
            )
        )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def compute_readiness(
    df: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
    vif_df: pd.DataFrame | None = None,
) -> ReadinessResult:
    """Compute a 0-100 data readiness score with category breakdown.

    Args:
        df: Full DataFrame (features + optional target).
        target: Target column name (needed for imbalance penalty).
        task: ``"classification"`` or ``"regression"``. Inferred if None.
        vif_df: Optional output of :func:`~insightml.intelligence.multicollinearity.compute_vif`.

    Returns:
        :class:`ReadinessResult`
    """
    score = 100.0
    breakdown: dict[str, dict[str, Any]] = {}
    recommendations: list[str] = []

    feature_cols = [c for c in df.columns if c != target]
    n_samples = len(df)
    n_features = len(feature_cols)

    # --- Missing value penalty ---
    total_cells = n_samples * max(n_features, 1)
    total_missing = int(df[feature_cols].isna().sum().sum())
    missing_pct = total_missing / total_cells if total_cells > 0 else 0.0
    n_high_missing = int((df[feature_cols].isna().mean() > 0.50).sum())

    missing_penalty = min(_MAX_MISSING_PENALTY, missing_pct * 50 + n_high_missing * 2)
    score -= missing_penalty
    breakdown["missing_values"] = {
        "delta": -missing_penalty,
        "description": f"{missing_pct:.1%} missing overall; {n_high_missing} col(s) >50% missing",
    }
    if missing_pct > 0.1:
        recommendations.append(
            f"High missing rate ({missing_pct:.1%}): consider imputation or column removal."
        )

    # --- Class imbalance penalty (classification only) ---
    if target and target in df.columns:
        y = df[target]
        if task is None:
            task = _infer_task(y)
        if task == "classification":
            vc = y.value_counts()
            minority_ratio = float(vc.min() / vc.max()) if vc.max() > 0 else 1.0
            imbalance_penalty = max(0, (1 - minority_ratio) * _MAX_IMBALANCE_PENALTY)
            score -= imbalance_penalty
            breakdown["class_imbalance"] = {
                "delta": -imbalance_penalty,
                "description": f"minority ratio={minority_ratio:.2f}",
            }
            if minority_ratio < 0.3:
                recommendations.append(
                    f"Severe class imbalance (ratio={minority_ratio:.2f}): "
                    "consider SMOTE, class_weight='balanced', or threshold tuning."
                )

    # --- Multicollinearity penalty ---
    if vif_df is not None and not vif_df.empty and "vif" in vif_df.columns:
        n_high_vif = int((vif_df["vif"] >= 10).sum())
        multicol_penalty = min(_MAX_MULTICOLLINEARITY_PENALTY, n_high_vif * 3)
        score -= multicol_penalty
        breakdown["multicollinearity"] = {
            "delta": -multicol_penalty,
            "description": f"{n_high_vif} feature(s) with VIF>=10",
        }
        if n_high_vif > 0:
            recommendations.append(
                f"{n_high_vif} highly collinear feature(s) (VIF>=10): "
                "consider removing or combining with PCA."
            )
    else:
        breakdown["multicollinearity"] = {"delta": 0, "description": "VIF not computed"}

    # --- Outlier penalty (heuristic: high skew) ---
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        skews = [abs(float(df[c].skew())) for c in numeric_cols if df[c].notna().sum() > 3]
        avg_outlier_proxy = sum(s > 3 for s in skews) / len(skews) if skews else 0.0
        outlier_penalty = min(_MAX_OUTLIER_PENALTY, avg_outlier_proxy * 20)
        score -= outlier_penalty
        breakdown["outliers"] = {
            "delta": -outlier_penalty,
            "description": f"{avg_outlier_proxy:.1%} of numeric columns have |skew|>3",
        }
    else:
        breakdown["outliers"] = {"delta": 0, "description": "no numeric columns"}

    # --- Constant feature penalty ---
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
    const_penalty = len(constant_cols) * _CONSTANT_FEATURE_PENALTY
    score -= const_penalty
    breakdown["constant_features"] = {
        "delta": -const_penalty,
        "description": f"{len(constant_cols)} constant/near-constant column(s)",
    }
    if constant_cols:
        recommendations.append(
            f"Remove {len(constant_cols)} constant column(s): {constant_cols[:5]}."
        )

    # --- Sample size bonus ---
    samples_per_feature = n_samples / max(n_features, 1)
    sample_bonus = min(_MAX_SAMPLE_BONUS, samples_per_feature / 10)
    score += sample_bonus
    breakdown["sample_size"] = {
        "delta": sample_bonus,
        "description": f"{n_samples} rows, {samples_per_feature:.1f} samples/feature",
    }

    # --- Feature diversity bonus ---
    if n_features > 0:
        type_set = set()
        for col in feature_cols:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                type_set.add("numeric")
            elif pd.api.types.is_bool_dtype(s):
                type_set.add("boolean")
            else:
                type_set.add("categorical")
        diversity_bonus = min(_MAX_DIVERSITY_BONUS, len(type_set) * 1.5)
        score += diversity_bonus
        breakdown["feature_diversity"] = {
            "delta": diversity_bonus,
            "description": f"{len(type_set)} distinct feature type(s): {sorted(type_set)}",
        }

    # Clamp [0, 100]
    score = float(np.clip(score, 0, 100))

    # Grade
    grade = "F"
    for g, threshold in _GRADE_THRESHOLDS:
        if score >= threshold:
            grade = g
            break

    return ReadinessResult(
        score=round(score, 1),
        grade=grade,
        breakdown=breakdown,
        recommendations=recommendations,
        n_samples=n_samples,
        n_features=n_features,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_task(y: pd.Series) -> str:
    if (
        pd.api.types.is_bool_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(y)
        or str(y.dtype) in ("string", "category")
        or y.nunique() <= 20
    ):
        return "classification"
    return "regression"
