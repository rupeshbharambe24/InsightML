"""FeatureInteractions — interaction strength and non-linearity detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from insightml._types import ColumnType
from insightml.core.validators import infer_column_type
from insightml.eda._base import BaseAnalysisModule
from insightml.viz.theme import QUALITATIVE, make_figure


class FeatureInteractions(BaseAnalysisModule):
    """Detect feature interactions and non-linear relationships.

    **Interaction strength (with target)**:
    OLS residual approach — measures how much A*B explains target beyond A+B alone.

    **Interaction strength (without target)**:
    ``1 - |corr(A*B, A+B)|`` — high value means non-additive interaction.

    **Non-linearity detection** per pair:
    Compare R² of linear vs quadratic fit. If improvement > 0.05 → non-linear.

    Access::

        eda.interactions.strengths()         # all pair strengths DataFrame
        eda.interactions.nonlinear_pairs()   # pairs with non-linear relationship
        eda.interactions.top_interactions(n=10)
        eda.interactions.interaction_plot("age", "fare")
    """

    def _compute(self) -> None:
        df = self._df
        config = self._config
        target = self._target

        num_cols = [
            c for c in df.columns
            if infer_column_type(df[c], config) == ColumnType.NUMERIC
            and c != target
        ]

        if len(num_cols) < 2:
            self._warn("Need at least 2 numeric features for interaction analysis.")
            self._results = {"num_cols": num_cols, "pairs": [], "skipped": True}
            return

        # Limit pairs to avoid O(n²) explosion on wide datasets
        max_cols = min(len(num_cols), 20)
        if len(num_cols) > max_cols:
            # Use top-variance columns
            variances = df[num_cols].var().sort_values(ascending=False)
            num_cols = list(variances.head(max_cols).index)
            self._warn(
                f"Limited to top {max_cols} numeric columns by variance for interaction analysis."
            )

        # Get target series if available
        target_series = df[target].dropna() if target and target in df.columns else None

        pairs_result = []
        for i, c1 in enumerate(num_cols):
            for j, c2 in enumerate(num_cols):
                if i >= j:
                    continue
                try:
                    result = _analyze_pair(df, c1, c2, target_series)
                    result["col_a"] = c1
                    result["col_b"] = c2
                    pairs_result.append(result)
                except Exception:
                    pass

        # Sort by interaction strength descending
        pairs_result.sort(
            key=lambda x: x.get("interaction_strength", 0) or 0, reverse=True
        )

        self._results = {
            "num_cols": num_cols,
            "pairs": pairs_result,
            "skipped": False,
        }

    def _build_figures(self) -> dict[str, go.Figure]:
        if self._results.get("skipped"):
            return {}

        figs: dict[str, go.Figure] = {}
        pairs = self._results["pairs"]

        if not pairs:
            return figs

        # Top interactions bar chart
        top_n = min(15, len(pairs))
        top = pairs[:top_n]
        labels = [f"{p['col_a']} × {p['col_b']}" for p in top]
        strengths = [p.get("interaction_strength", 0) or 0 for p in top]
        fig = make_figure(title="Top Feature Interaction Strengths")
        fig.add_trace(go.Bar(
            y=labels[::-1], x=strengths[::-1],
            orientation="h",
            marker_color=QUALITATIVE[0],
            text=[f"{s:.3f}" for s in strengths[::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Interaction Strength",
            height=max(300, top_n * 30),
        )
        figs["top_interactions"] = fig

        return figs

    def summary(self) -> str:
        self._ensure_computed()
        if self._results.get("skipped"):
            return "Interaction analysis skipped (insufficient numeric columns)."
        pairs = self._results["pairs"]
        if not pairs:
            return "No interaction pairs computed."
        strong = [p for p in pairs if (p.get("interaction_strength") or 0) > 0.3]
        nonlin = [p for p in pairs if p.get("is_nonlinear", False)]
        return (
            f"Analyzed {len(pairs)} feature pairs. "
            f"{len(strong)} with strong interactions (strength > 0.3). "
            f"{len(nonlin)} non-linear relationships detected."
        )

    # --- Public accessors ---

    def strengths(self) -> pd.DataFrame:
        self._ensure_computed()
        pairs = self._results.get("pairs", [])
        if not pairs:
            return pd.DataFrame()
        rows = [
            {
                "col_a": p["col_a"], "col_b": p["col_b"],
                "interaction_strength": p.get("interaction_strength"),
                "is_nonlinear": p.get("is_nonlinear"),
                "r2_linear": p.get("r2_linear"),
                "r2_poly": p.get("r2_poly"),
            }
            for p in pairs
        ]
        return pd.DataFrame(rows)

    def nonlinear_pairs(self) -> pd.DataFrame:
        self._ensure_computed()
        pairs = self._results.get("pairs", [])
        nonlin = [p for p in pairs if p.get("is_nonlinear", False)]
        return pd.DataFrame(nonlin) if nonlin else pd.DataFrame()

    def top_interactions(self, n: int = 10) -> pd.DataFrame:
        self._ensure_computed()
        return self.strengths().head(n)

    def interaction_plot(self, col_a: str, col_b: str) -> go.Figure:
        """3D scatter of col_a, col_b, and their product."""
        self._ensure_computed()
        df = self._df
        mask = df[col_a].notna() & df[col_b].notna()
        x = df.loc[mask, col_a]
        y = df.loc[mask, col_b]
        z = x * y
        fig = make_figure(title=f"{col_a} × {col_b} Interaction")
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=3, opacity=0.5,
                color=z, colorscale="Viridis", showscale=True,
            ),
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title=col_a,
                yaxis_title=col_b,
                zaxis_title=f"{col_a} × {col_b}",
            ),
            height=500,
        )
        return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analyze_pair(
    df: pd.DataFrame,
    c1: str, c2: str,
    target: pd.Series | None,
) -> dict[str, Any]:
    mask = df[c1].notna() & df[c2].notna()
    if target is not None:
        mask = mask & target.notna()
    if mask.sum() < 10:
        return {"interaction_strength": None, "is_nonlinear": None}

    x1 = df.loc[mask, c1].values.astype(float)
    x2 = df.loc[mask, c2].values.astype(float)

    # Interaction term
    interaction = x1 * x2
    additive = x1 + x2

    # Interaction strength (without target): non-additivity
    corr_with_additive = float(np.corrcoef(interaction, additive)[0, 1])
    strength_no_target = round(1 - abs(corr_with_additive), 4)

    # Interaction strength (with target): residual approach
    strength_with_target = None
    if target is not None:
        y = target.loc[mask].values.astype(float)
        strength_with_target = _residual_interaction_strength(x1, x2, y)

    interaction_strength = (
        strength_with_target if strength_with_target is not None
        else strength_no_target
    )

    # Non-linearity detection: R² linear vs polynomial
    r2_linear, r2_poly = _linearity_check(x1, x2)
    improvement = (r2_poly - r2_linear) if (r2_linear is not None and r2_poly is not None) else 0
    is_nonlinear = improvement > 0.05

    return {
        "interaction_strength": round(float(interaction_strength), 4)
        if interaction_strength is not None else None,
        "strength_no_target": round(strength_no_target, 4),
        "strength_with_target": round(float(strength_with_target), 4)
        if strength_with_target is not None else None,
        "r2_linear": round(r2_linear, 4) if r2_linear is not None else None,
        "r2_poly": round(r2_poly, 4) if r2_poly is not None else None,
        "is_nonlinear": bool(is_nonlinear),
        "nonlinearity_improvement": round(float(improvement), 4),
    }


def _residual_interaction_strength(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray
) -> float | None:
    """OLS residual approach to interaction strength."""
    try:
        X_main = np.column_stack([np.ones(len(x1)), x1, x2])
        # OLS via normal equations
        beta = np.linalg.lstsq(X_main, y, rcond=None)[0]
        residuals_y = y - X_main @ beta

        interaction = x1 * x2
        X_int = np.column_stack([np.ones(len(x1)), x1, x2])
        beta_int = np.linalg.lstsq(X_int, interaction, rcond=None)[0]
        residuals_int = interaction - X_int @ beta_int

        if residuals_int.std() < 1e-10:
            return 0.0
        corr = abs(float(np.corrcoef(residuals_y, residuals_int)[0, 1]))
        return corr
    except Exception:
        return None


def _linearity_check(
    x: np.ndarray, y: np.ndarray
) -> tuple[float | None, float | None]:
    """Compare R² of linear vs degree-2 polynomial fit of y ~ x."""
    try:
        # Handle NaN
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 5:
            return None, None

        # Linear
        coeffs_lin = np.polyfit(x, y, 1)
        y_pred_lin = np.polyval(coeffs_lin, x)
        ss_res_lin = np.sum((y - y_pred_lin) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_linear = 1 - ss_res_lin / (ss_tot + 1e-10)

        # Polynomial deg 2
        coeffs_poly = np.polyfit(x, y, 2)
        y_pred_poly = np.polyval(coeffs_poly, x)
        ss_res_poly = np.sum((y - y_pred_poly) ** 2)
        r2_poly = 1 - ss_res_poly / (ss_tot + 1e-10)

        return float(r2_linear), float(r2_poly)
    except Exception:
        return None, None
