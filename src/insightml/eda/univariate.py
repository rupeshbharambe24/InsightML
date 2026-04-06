"""UnivariateAnalysis — per-column distributions and descriptive statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import entropy as scipy_entropy

from insightml._types import ColumnType
from insightml.core.validators import infer_column_type
from insightml.eda._base import BaseAnalysisModule
from insightml.viz.theme import QUALITATIVE, make_figure


class UnivariateAnalysis(BaseAnalysisModule):
    """Per-column univariate analysis.

    - **Numeric**: histogram + KDE overlay, box plot, full descriptive stats,
      quick normality flag (skewness + kurtosis heuristic).
    - **Categorical/Boolean**: frequency bar chart (top 20), Shannon entropy.
    - **DateTime**: time range, gap detection, frequency bar of gaps.

    Access::

        eda.univariate.show()              # all charts
        eda.univariate.plot("age")         # chart for one column
        eda.univariate.stats("age")        # descriptive stats dict
    """

    def _compute(self) -> None:
        df = self._df
        config = self._config
        stats: dict[str, Any] = {}

        for col_name in df.columns:
            col = df[col_name]
            ct = infer_column_type(col, config)
            col_stats: dict[str, Any] = {"type": ct.value}

            if ct == ColumnType.NUMERIC:
                non_null = col.dropna()
                col_stats.update({
                    "count": int(non_null.count()),
                    "missing": int(col.isna().sum()),
                    "mean": float(non_null.mean()),
                    "median": float(non_null.median()),
                    "std": float(non_null.std()),
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "q1": float(non_null.quantile(0.25)),
                    "q3": float(non_null.quantile(0.75)),
                    "iqr": float(non_null.quantile(0.75) - non_null.quantile(0.25)),
                    "skewness": float(non_null.skew()),
                    "kurtosis": float(non_null.kurtosis()),
                    "likely_normal": abs(float(non_null.skew())) < 0.5
                                     and abs(float(non_null.kurtosis())) < 1.0,
                })

            elif ct in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN,
                        ColumnType.HIGH_CARDINALITY):
                vc = col.value_counts(dropna=True)
                probs = vc / vc.sum()
                col_stats.update({
                    "n_unique": int(col.nunique()),
                    "missing": int(col.isna().sum()),
                    "top_value": str(vc.index[0]) if len(vc) > 0 else None,
                    "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
                    "entropy": float(scipy_entropy(probs)),
                    "value_counts": {str(k): int(v)
                                     for k, v in vc.head(20).items()},
                })

            elif ct == ColumnType.DATETIME:
                non_null = col.dropna()
                if not pd.api.types.is_datetime64_any_dtype(non_null):
                    try:
                        non_null = pd.to_datetime(non_null, errors="coerce",
                                                  format="mixed").dropna()
                    except Exception:
                        pass
                if pd.api.types.is_datetime64_any_dtype(non_null) and len(non_null) >= 2:
                    sorted_col = non_null.sort_values()
                    diffs = sorted_col.diff().dropna()
                    median_diff = diffs.median()
                    gap_threshold = median_diff * 3
                    n_gaps = int((diffs > gap_threshold).sum())
                    col_stats.update({
                        "min": str(sorted_col.min()),
                        "max": str(sorted_col.max()),
                        "range_days": float(
                            (sorted_col.max() - sorted_col.min()).days
                        ),
                        "n_gaps": n_gaps,
                        "median_interval": str(median_diff),
                    })

            stats[col_name] = col_stats

        self._results["stats"] = stats

    def _build_figures(self) -> dict[str, go.Figure]:
        df = self._df
        config = self._config
        figs: dict[str, go.Figure] = {}

        for col_name in df.columns:
            col = df[col_name]
            ct = infer_column_type(col, config)

            if ct == ColumnType.NUMERIC:
                figs[f"{col_name}_hist"] = _numeric_fig(col)
                figs[f"{col_name}_box"] = _box_fig(col)

            elif ct in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN,
                        ColumnType.HIGH_CARDINALITY):
                figs[f"{col_name}_freq"] = _freq_fig(col)

            elif ct == ColumnType.DATETIME:
                figs[f"{col_name}_timeline"] = _datetime_fig(col)

        return figs

    def summary(self) -> str:
        stats = self._results.get("stats", {})
        n_numeric = sum(1 for s in stats.values()
                        if s.get("type") == ColumnType.NUMERIC.value)
        n_cat = sum(1 for s in stats.values()
                    if s.get("type") in (ColumnType.CATEGORICAL.value,
                                         ColumnType.BOOLEAN.value))
        skewed = [c for c, s in stats.items()
                  if s.get("type") == ColumnType.NUMERIC.value
                  and abs(s.get("skewness", 0)) > 1]
        result = (
            f"Analyzed {len(stats)} columns: {n_numeric} numeric, {n_cat} categorical."
        )
        if skewed:
            result += f" {len(skewed)} numeric column(s) are highly skewed: {skewed}."
        return result

    # --- Convenience accessors ---

    def stats(self, col: str) -> dict[str, Any]:
        """Return computed statistics for a single column."""
        self._ensure_computed()
        if col not in self._results["stats"]:
            raise KeyError(f"Column '{col}' not found.")
        return self._results["stats"][col]

    def plot(self, col: str | None = None):
        """Return figure(s) for a column, or all figures if col is None."""
        self._ensure_computed()
        if col is None:
            return dict(self._figures)
        matching = {k: v for k, v in self._figures.items()
                    if k.startswith(f"{col}_")}
        if not matching:
            raise KeyError(f"No figures for column '{col}'.")
        if len(matching) == 1:
            return next(iter(matching.values()))
        return matching


# ---------------------------------------------------------------------------
# Private figure builders
# ---------------------------------------------------------------------------

def _numeric_fig(col: pd.Series) -> go.Figure:
    data = col.dropna()
    fig = make_figure(title=f"{col.name} — Distribution")
    bin_count = min(50, max(10, len(data) // 20))
    fig.add_trace(go.Histogram(
        x=data, nbinsx=bin_count,
        name="Count", marker_color=QUALITATIVE[0], opacity=0.75,
    ))
    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        if len(data) >= 5:
            kde_x = np.linspace(data.min(), data.max(), 200)
            kde_y = gaussian_kde(data)(kde_x)
            counts, edges = np.histogram(data, bins=bin_count)
            scale = counts.max() / (kde_y.max() + 1e-10)
            fig.add_trace(go.Scatter(
                x=kde_x, y=kde_y * scale,
                mode="lines", name="KDE",
                line={"color": QUALITATIVE[1], "width": 2},
            ))
    except Exception:
        pass
    fig.update_layout(xaxis_title=col.name, yaxis_title="Count", height=350)
    return fig


def _box_fig(col: pd.Series) -> go.Figure:
    fig = make_figure(title=f"{col.name} — Box Plot")
    fig.add_trace(go.Box(
        y=col.dropna(), name=col.name,
        marker_color=QUALITATIVE[0],
        boxpoints="outliers", jitter=0.3, pointpos=-1.8,
    ))
    fig.update_layout(height=350)
    return fig


def _freq_fig(col: pd.Series, top_n: int = 20) -> go.Figure:
    counts = col.value_counts().head(top_n)
    fig = make_figure(title=f"{col.name} — Frequencies")
    fig.add_trace(go.Bar(
        y=counts.index.astype(str), x=counts.values,
        orientation="h", marker_color=QUALITATIVE[0],
        text=counts.values, textposition="outside",
    ))
    fig.update_layout(
        yaxis={"autorange": "reversed"},
        xaxis_title="Count", height=max(300, len(counts) * 30),
    )
    return fig


def _datetime_fig(col: pd.Series) -> go.Figure:
    non_null = col.dropna()
    try:
        if not pd.api.types.is_datetime64_any_dtype(non_null):
            non_null = pd.to_datetime(non_null, errors="coerce",
                                      format="mixed").dropna()
        sorted_col = non_null.sort_values()
        diffs_days = sorted_col.diff().dropna().dt.days
        fig = make_figure(title=f"{col.name} — Gap Distribution (days)")
        fig.add_trace(go.Histogram(
            x=diffs_days, nbinsx=30,
            marker_color=QUALITATIVE[2], opacity=0.75,
        ))
        fig.update_layout(xaxis_title="Days between events",
                          yaxis_title="Count", height=300)
        return fig
    except Exception:
        return make_figure(title=f"{col.name} — DateTime (no chart)")
