"""CorrelationAnalysis — unified multi-type correlation matrix."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, pointbiserialr

from insightml._types import ColumnType
from insightml.core.validators import infer_column_type
from insightml.eda._base import BaseAnalysisModule
from insightml.viz.theme import make_figure


class CorrelationAnalysis(BaseAnalysisModule):
    """Unified correlation matrix using the statistically appropriate measure per cell.

    | Cell Types            | Measure          |
    |-----------------------|------------------|
    | Numeric–Numeric       | Pearson r        |
    | Numeric–Numeric(rank) | Spearman rho     |
    | Cat–Cat               | Cramer's V       |
    | Numeric–Binary        | Point-biserial r |
    | Numeric–Categorical   | Correlation ratio (eta) |

    Access::

        eda.correlations.heatmap()               # unified matrix
        eda.correlations.heatmap("pearson")      # pearson only
        eda.correlations.top_correlations(n=10)  # strongest pairs
        eda.correlations.unified()               # raw DataFrame
    """

    def _compute(self) -> None:
        df = self._df
        config = self._config

        col_types = {c: infer_column_type(df[c], config) for c in df.columns}

        num_cols = [c for c, t in col_types.items() if t == ColumnType.NUMERIC]
        cat_cols = [c for c, t in col_types.items()
                    if t in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN,
                              ColumnType.HIGH_CARDINALITY)]
        bin_cols = [c for c in cat_cols if df[c].nunique() == 2]

        # --- Pearson ---
        if len(num_cols) >= 2:
            pearson = df[num_cols].corr(method="pearson")
        else:
            pearson = pd.DataFrame()

        # --- Spearman ---
        if len(num_cols) >= 2:
            spearman = df[num_cols].corr(method="spearman")
        else:
            spearman = pd.DataFrame()

        # --- Cramer's V matrix ---
        cramers = _cramers_v_matrix(df, cat_cols)

        # --- Unified matrix ---
        all_cols = num_cols + cat_cols
        unified = pd.DataFrame(np.nan, index=all_cols, columns=all_cols)

        # Fill Pearson for num-num
        for c1 in num_cols:
            for c2 in num_cols:
                if not pearson.empty and c1 in pearson.index and c2 in pearson.columns:
                    unified.loc[c1, c2] = pearson.loc[c1, c2]

        # Fill Cramer's V for cat-cat
        for c1 in cat_cols:
            for c2 in cat_cols:
                if c1 in cramers.index and c2 in cramers.columns:
                    unified.loc[c1, c2] = cramers.loc[c1, c2]

        # Fill point-biserial for num-binary
        for nc in num_cols:
            for bc in bin_cols:
                val = _point_biserial(df[nc], df[bc])
                unified.loc[nc, bc] = val
                unified.loc[bc, nc] = val

        # Fill eta (correlation ratio) for num-categorical
        non_bin_cats = [c for c in cat_cols if c not in bin_cols]
        for nc in num_cols:
            for cc in non_bin_cats:
                val = _correlation_ratio(df[cc], df[nc])
                unified.loc[nc, cc] = val
                unified.loc[cc, nc] = val

        # Diagonal = 1
        for c in all_cols:
            if c in unified.index:
                unified.loc[c, c] = 1.0

        # Top correlations (off-diagonal, dedup)
        top_pairs = _top_correlations(unified)

        self._results = {
            "pearson": pearson,
            "spearman": spearman,
            "cramers_v": cramers,
            "unified": unified,
            "top_correlations": top_pairs,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        }

    def _build_figures(self) -> dict[str, go.Figure]:
        figs: dict[str, go.Figure] = {}
        unified = self._results["unified"]
        if not unified.empty:
            figs["unified"] = _heatmap_fig(unified, "Unified Correlation Matrix")
        pearson = self._results["pearson"]
        if not pearson.empty:
            figs["pearson"] = _heatmap_fig(pearson, "Pearson Correlation")
        return figs

    def summary(self) -> str:
        top = self._results.get("top_correlations", [])
        unified: pd.DataFrame = self._results.get("unified", pd.DataFrame())
        n_cols = len(unified.columns) if not unified.empty else 0
        if not top:
            return f"Correlation analysis on {n_cols} columns. No strong correlations found."
        strongest = top[0]
        return (
            f"Unified correlation matrix for {n_cols} columns. "
            f"Strongest: {strongest['col_a']} ↔ {strongest['col_b']} "
            f"= {strongest['value']:.3f} ({strongest['method']})."
        )

    # --- Public methods ---

    def pearson(self) -> pd.DataFrame:
        """Return Pearson correlation matrix (numeric columns only)."""
        self._ensure_computed()
        return self._results["pearson"]

    def spearman(self) -> pd.DataFrame:
        """Return Spearman rank correlation matrix (numeric columns only)."""
        self._ensure_computed()
        return self._results["spearman"]

    def cramers_v_matrix(self) -> pd.DataFrame:
        """Return Cramer's V matrix (categorical columns only)."""
        self._ensure_computed()
        return self._results["cramers_v"]

    def unified(self) -> pd.DataFrame:
        """Return the unified correlation matrix (all columns)."""
        self._ensure_computed()
        return self._results["unified"]

    def heatmap(self, method: str = "unified") -> go.Figure:
        """Return a Plotly heatmap for the chosen correlation method."""
        self._ensure_computed()
        key = method if method in self._figures else "unified"
        return self._figures.get(key, make_figure(title="No data"))

    def top_correlations(self, n: int = 20) -> pd.DataFrame:
        """Return top-N correlated pairs as a DataFrame."""
        self._ensure_computed()
        pairs = self._results["top_correlations"][:n]
        return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cramers_v_matrix(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    if len(cat_cols) < 2:
        if len(cat_cols) == 1:
            return pd.DataFrame([[1.0]], index=cat_cols, columns=cat_cols)
        return pd.DataFrame()
    n = len(cat_cols)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = _cramers_v(df[cat_cols[i]], df[cat_cols[j]])
            mat[i, j] = v
            mat[j, i] = v
    return pd.DataFrame(mat, index=cat_cols, columns=cat_cols)


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    try:
        ct = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(ct, correction=False)
        n = ct.sum().sum()
        min_dim = min(ct.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))
    except Exception:
        return np.nan


def _point_biserial(numeric: pd.Series, binary: pd.Series) -> float:
    try:
        mask = numeric.notna() & binary.notna()
        if mask.sum() < 5:
            return np.nan
        num_clean = numeric[mask]
        bin_clean = binary[mask]
        # Encode binary to 0/1
        unique_vals = bin_clean.dropna().unique()
        if len(unique_vals) != 2:
            return np.nan
        encoded = (bin_clean == unique_vals[0]).astype(int)
        r, _ = pointbiserialr(encoded, num_clean)
        return float(r)
    except Exception:
        return np.nan


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio (eta) — strength of relationship between categorical and numeric."""
    try:
        mask = categories.notna() & values.notna()
        cats = categories[mask]
        vals = values[mask]
        if len(vals) < 5:
            return np.nan
        grand_mean = vals.mean()
        groups = vals.groupby(cats)
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2 for _, g in groups
        )
        ss_total = ((vals - grand_mean) ** 2).sum()
        if ss_total == 0:
            return 0.0
        return float(np.sqrt(ss_between / ss_total))
    except Exception:
        return np.nan


def _top_correlations(unified: pd.DataFrame) -> list[dict[str, Any]]:
    cols = list(unified.columns)
    pairs = []
    seen: set[frozenset] = set()
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                continue
            key = frozenset([c1, c2])
            if key in seen:
                continue
            seen.add(key)
            val = unified.loc[c1, c2]
            if pd.isna(val):
                continue
            pairs.append({
                "col_a": c1, "col_b": c2,
                "value": round(float(val), 4),
                "abs_value": round(abs(float(val)), 4),
                "method": "unified",
            })
    pairs.sort(key=lambda x: x["abs_value"], reverse=True)
    return pairs


def _heatmap_fig(matrix: pd.DataFrame, title: str) -> go.Figure:
    n = len(matrix.columns)
    size = max(500, 40 * n)
    fig = make_figure(title=title)
    fig.add_trace(go.Heatmap(
        z=matrix.values.tolist(),
        x=list(matrix.columns),
        y=list(matrix.index),
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" if not np.isnan(v) else ""
               for v in row]
              for row in matrix.values],
        texttemplate="%{text}",
        hoverongaps=False,
    ))
    fig.update_layout(width=size, height=size)
    return fig
