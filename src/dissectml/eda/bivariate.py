"""BivariateAnalysis — cross-type pairwise analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, f_oneway

from dissectml._types import ColumnType
from dissectml.core.validators import infer_column_type
from dissectml.eda._base import BaseAnalysisModule
from dissectml.viz.theme import QUALITATIVE, make_figure


class BivariateAnalysis(BaseAnalysisModule):
    """Cross-type pairwise analysis between columns.

    | Type Pair  | Analysis                  | Chart                  |
    |------------|---------------------------|------------------------|
    | Num–Num    | Pearson r, Spearman, OLS  | Scatter + trendline    |
    | Num–Cat    | Group means, ANOVA F-stat | Grouped violin         |
    | Cat–Cat    | Chi-square, Cramer's V    | Contingency heatmap    |
    | Any–Target | Auto-selects by type      | Varies                 |

    Pair limiting: when > ``config.max_bivariate_pairs`` columns, only
    compute pairs involving the target or high-variance columns.

    Access::

        eda.bivariate.show()
        eda.bivariate.pair("age", "income")   # stats dict for one pair
    """

    def _compute(self) -> None:
        df = self._df
        config = self._config
        target = self._target

        col_types = {c: infer_column_type(df[c], config) for c in df.columns}
        list(df.columns)

        # --- Limit pairs ---
        cols_to_analyze = _select_columns(df, col_types, target, config.max_bivariate_pairs)
        pairs = [
            (c1, c2)
            for i, c1 in enumerate(cols_to_analyze)
            for j, c2 in enumerate(cols_to_analyze)
            if i < j
        ]

        results: dict[str, Any] = {}
        for c1, c2 in pairs:
            key = f"{c1}||{c2}"
            t1, t2 = col_types[c1], col_types[c2]
            try:
                results[key] = _analyze_pair(df, c1, t1, c2, t2)
            except Exception as e:
                results[key] = {"error": str(e)}

        self._results["pairs"] = results
        self._results["analyzed_cols"] = cols_to_analyze

    def _build_figures(self) -> dict[str, go.Figure]:
        df = self._df
        config = self._config
        col_types = {c: infer_column_type(df[c], config) for c in df.columns}
        cols = self._results.get("analyzed_cols", [])
        figs: dict[str, go.Figure] = {}

        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i >= j:
                    continue
                key = f"{c1}||{c2}"
                t1, t2 = col_types[c1], col_types[c2]
                try:
                    fig = _build_pair_figure(df, c1, t1, c2, t2)
                    if fig is not None:
                        figs[key] = fig
                except Exception:
                    pass

        return figs

    def summary(self) -> str:
        pairs = self._results.get("pairs", {})
        n_pairs = len(pairs)
        strong = [
            k for k, v in pairs.items()
            if isinstance(v, dict) and abs(v.get("correlation", 0) or 0) > 0.5
        ]
        return (
            f"Analyzed {n_pairs} column pairs. "
            f"{len(strong)} pairs show strong association (|r| > 0.5)."
        )

    def pair(self, col_a: str, col_b: str) -> dict[str, Any]:
        """Return analysis results for a specific pair."""
        self._ensure_computed()
        key = f"{col_a}||{col_b}"
        alt_key = f"{col_b}||{col_a}"
        return (self._results["pairs"].get(key)
                or self._results["pairs"].get(alt_key)
                or {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_columns(
    df: pd.DataFrame,
    col_types: dict[str, ColumnType],
    target: str | None,
    max_cols: int,
) -> list[str]:
    all_cols = list(df.columns)
    if len(all_cols) <= max_cols:
        return all_cols

    # Priority: target first, then high-variance numerics, then high-cardinality cats
    priority = []
    if target and target in all_cols:
        priority.append(target)

    num_cols = [c for c, t in col_types.items()
                if t == ColumnType.NUMERIC and c != target]
    if num_cols:
        variances = df[num_cols].var().sort_values(ascending=False)
        priority += list(variances.head(max_cols // 2).index)

    cat_cols = [c for c, t in col_types.items()
                if t == ColumnType.CATEGORICAL and c != target]
    remaining = max_cols - len(priority)
    priority += cat_cols[:remaining]

    # Deduplicate preserving order
    seen: set[str] = set()
    result = []
    for c in priority:
        if c not in seen and c in all_cols:
            seen.add(c)
            result.append(c)
    return result[:max_cols]


def _analyze_pair(
    df: pd.DataFrame,
    c1: str, t1: ColumnType,
    c2: str, t2: ColumnType,
) -> dict[str, Any]:
    IS_NUM = {ColumnType.NUMERIC}
    IS_CAT = {ColumnType.CATEGORICAL, ColumnType.BOOLEAN, ColumnType.HIGH_CARDINALITY}

    # Num–Num
    if t1 in IS_NUM and t2 in IS_NUM:
        mask = df[c1].notna() & df[c2].notna()
        x, y = df.loc[mask, c1], df.loc[mask, c2]
        if len(x) < 5:
            return {"type": "num-num", "n": len(x)}
        pearson = float(x.corr(y, method="pearson"))
        spearman = float(x.corr(y, method="spearman"))
        return {
            "type": "num-num", "n": int(mask.sum()),
            "pearson_r": round(pearson, 4),
            "spearman_rho": round(spearman, 4),
            "correlation": pearson,
        }

    # Num–Cat
    if (t1 in IS_NUM and t2 in IS_CAT) or (t1 in IS_CAT and t2 in IS_NUM):
        num_col, cat_col = (c1, c2) if t1 in IS_NUM else (c2, c1)
        mask = df[num_col].notna() & df[cat_col].notna()
        groups = [
            df.loc[mask & (df[cat_col] == g), num_col].values
            for g in df.loc[mask, cat_col].unique()
            if (df.loc[mask & (df[cat_col] == g), num_col]).count() >= 2
        ]
        if len(groups) < 2:
            return {"type": "num-cat", "n": int(mask.sum())}
        f_stat, p_val = f_oneway(*groups)
        group_means = df.loc[mask].groupby(cat_col)[num_col].mean().to_dict()
        return {
            "type": "num-cat", "n": int(mask.sum()),
            "anova_f": round(float(f_stat), 4),
            "anova_p": round(float(p_val), 4),
            "significant": float(p_val) < 0.05,
            "group_means": {str(k): round(float(v), 4) for k, v in group_means.items()},
            "correlation": None,
        }

    # Cat–Cat
    if t1 in IS_CAT and t2 in IS_CAT:
        mask = df[c1].notna() & df[c2].notna()
        if mask.sum() < 5:
            return {"type": "cat-cat", "n": int(mask.sum())}
        ct = pd.crosstab(df.loc[mask, c1], df.loc[mask, c2])
        chi2, p_val, dof, _ = chi2_contingency(ct, correction=False)
        n = ct.sum().sum()
        min_dim = min(ct.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n * max(min_dim, 1)))) if n > 0 else 0.0
        return {
            "type": "cat-cat", "n": int(mask.sum()),
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p_val), 4),
            "dof": int(dof),
            "cramers_v": round(cramers_v, 4),
            "significant": float(p_val) < 0.05,
            "correlation": cramers_v,
        }

    return {"type": "other"}


def _build_pair_figure(
    df: pd.DataFrame,
    c1: str, t1: ColumnType,
    c2: str, t2: ColumnType,
) -> go.Figure | None:
    IS_NUM = {ColumnType.NUMERIC}
    IS_CAT = {ColumnType.CATEGORICAL, ColumnType.BOOLEAN, ColumnType.HIGH_CARDINALITY}

    if t1 in IS_NUM and t2 in IS_NUM:
        mask = df[c1].notna() & df[c2].notna()
        x, y = df.loc[mask, c1], df.loc[mask, c2]
        fig = make_figure(title=f"{c1} vs {c2}")
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker={"color": QUALITATIVE[0], "size": 5, "opacity": 0.5},
            showlegend=False,
        ))
        if len(x) >= 2:
            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            fig.add_trace(go.Scatter(
                x=x_line, y=np.polyval(coeffs, x_line),
                mode="lines", name="OLS",
                line={"color": QUALITATIVE[1], "width": 2, "dash": "dash"},
            ))
        fig.update_layout(xaxis_title=c1, yaxis_title=c2, height=350)
        return fig

    if (t1 in IS_NUM and t2 in IS_CAT) or (t1 in IS_CAT and t2 in IS_NUM):
        num_col, cat_col = (c1, c2) if t1 in IS_NUM else (c2, c1)
        mask = df[num_col].notna() & df[cat_col].notna()
        top_cats = df.loc[mask, cat_col].value_counts().head(8).index
        mask2 = mask & df[cat_col].isin(top_cats)
        fig = make_figure(title=f"{num_col} by {cat_col}")
        for i, group in enumerate(top_cats):
            vals = df.loc[mask2 & (df[cat_col] == group), num_col]
            fig.add_trace(go.Violin(
                y=vals, name=str(group),
                marker_color=QUALITATIVE[i % len(QUALITATIVE)],
                box_visible=True, meanline_visible=True,
            ))
        fig.update_layout(xaxis_title=cat_col, yaxis_title=num_col, height=350)
        return fig

    if t1 in IS_CAT and t2 in IS_CAT:
        mask = df[c1].notna() & df[c2].notna()
        top1 = df.loc[mask, c1].value_counts().head(8).index
        top2 = df.loc[mask, c2].value_counts().head(8).index
        mask2 = mask & df[c1].isin(top1) & df[c2].isin(top2)
        ct = pd.crosstab(df.loc[mask2, c1], df.loc[mask2, c2])
        fig = make_figure(title=f"{c1} vs {c2} (counts)")
        fig.add_trace(go.Heatmap(
            z=ct.values.tolist(),
            x=[str(v) for v in ct.columns],
            y=[str(v) for v in ct.index],
            colorscale="Blues",
            text=ct.values.tolist(), texttemplate="%{text}",
        ))
        n = max(300, 30 * max(len(ct.index), len(ct.columns)))
        fig.update_layout(height=n, width=n)
        return fig

    return None
