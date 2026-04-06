"""OutlierDetection — IQR, Z-score, Isolation Forest with consensus."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from insightml._types import ColumnType
from insightml.core.validators import infer_column_type
from insightml.eda._base import BaseAnalysisModule
from insightml.viz.theme import QUALITATIVE, make_figure


class OutlierDetection(BaseAnalysisModule):
    """Multi-method outlier detection for all numeric columns.

    Three methods per column:
    - **IQR**: fence = Q1 – k·IQR to Q3 + k·IQR (k=1.5 by default)
    - **Z-score**: |z| > threshold (default 3.0)
    - **Isolation Forest**: multivariate, contamination=0.05

    Consensus: a sample is flagged if ≥ 2 methods agree.

    Access::

        eda.outliers.by_iqr()               # per-column IQR results
        eda.outliers.by_zscore()            # per-column z-score results
        eda.outliers.consensus()            # consensus outlier indices
        eda.outliers.comparison()           # side-by-side counts DataFrame
        eda.outliers.plot("age")            # box plot for one column
    """

    def _compute(self) -> None:
        df = self._df
        config = self._config

        num_cols = [
            c for c in df.columns
            if infer_column_type(df[c], config) == ColumnType.NUMERIC
        ]

        iqr_results: dict[str, Any] = {}
        zscore_results: dict[str, Any] = {}
        iso_forest_result: dict[str, Any] = {}
        consensus_result: dict[str, Any] = {}

        # Per-column IQR and Z-score
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue

            # IQR
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            k = config.iqr_multiplier
            lower, upper = q1 - k * iqr, q3 + k * iqr
            iqr_outliers = series[(series < lower) | (series > upper)]
            iqr_results[col] = {
                "lower_fence": float(lower),
                "upper_fence": float(upper),
                "n_outliers": int(len(iqr_outliers)),
                "pct_outliers": round(100 * len(iqr_outliers) / len(series), 2),
                "outlier_indices": list(iqr_outliers.index),
            }

            # Z-score
            z = (series - series.mean()) / (series.std() + 1e-10)
            threshold = config.zscore_threshold
            z_outliers = series[z.abs() > threshold]
            zscore_results[col] = {
                "threshold": float(threshold),
                "n_outliers": int(len(z_outliers)),
                "pct_outliers": round(100 * len(z_outliers) / len(series), 2),
                "outlier_indices": list(z_outliers.index),
            }

        # Isolation Forest (multivariate, on all numeric cols together)
        if len(num_cols) >= 1:
            iso_forest_result = _run_isolation_forest(df, num_cols)

        # Consensus
        for col in num_cols:
            if col not in iqr_results:
                continue
            iqr_idx = set(iqr_results[col]["outlier_indices"])
            z_idx = set(zscore_results[col]["outlier_indices"])
            iso_idx = set(iso_forest_result.get("outlier_indices", []))

            # A point is a consensus outlier if flagged by ≥2 methods
            consensus_idx = (iqr_idx & z_idx) | (iqr_idx & iso_idx) | (z_idx & iso_idx)
            total_non_null = df[col].count()
            consensus_result[col] = {
                "n_outliers": len(consensus_idx),
                "pct_outliers": round(100 * len(consensus_idx) / max(total_non_null, 1), 2),
                "outlier_indices": list(consensus_idx),
            }

        self._results = {
            "num_cols": num_cols,
            "iqr": iqr_results,
            "zscore": zscore_results,
            "isolation_forest": iso_forest_result,
            "consensus": consensus_result,
        }

        # Warn about columns with many outliers
        high_outlier_cols = [
            c for c, r in consensus_result.items()
            if r["pct_outliers"] > 5
        ]
        if high_outlier_cols:
            self._warn(
                f"{len(high_outlier_cols)} column(s) have >5% outliers "
                f"by consensus: {high_outlier_cols}"
            )

    def _build_figures(self) -> dict[str, go.Figure]:
        df = self._df
        num_cols = self._results["num_cols"]
        figs: dict[str, go.Figure] = {}

        # Box plots per column
        for col in num_cols:
            fig = make_figure(title=f"{col} — Outlier Detection")
            fig.add_trace(go.Box(
                y=df[col].dropna(), name=col,
                marker_color=QUALITATIVE[0],
                boxpoints="outliers",
                jitter=0.3, pointpos=-1.8,
            ))
            fig.update_layout(height=350, showlegend=False)
            figs[f"{col}_box"] = fig

        # Comparison bar chart
        if self._results["consensus"]:
            figs["comparison"] = _comparison_fig(
                self._results["iqr"],
                self._results["zscore"],
                self._results["consensus"],
            )

        return figs

    def summary(self) -> str:
        consensus = self._results.get("consensus", {})
        if not consensus:
            return "No numeric columns found for outlier analysis."
        total_flagged = sum(r["n_outliers"] for r in consensus.values())
        cols_affected = sum(1 for r in consensus.values() if r["n_outliers"] > 0)
        return (
            f"Outlier detection on {len(consensus)} numeric column(s). "
            f"{total_flagged} consensus outlier(s) across {cols_affected} column(s)."
        )

    # --- Public accessors ---

    def by_iqr(self) -> dict[str, Any]:
        self._ensure_computed()
        return self._results["iqr"]

    def by_zscore(self) -> dict[str, Any]:
        self._ensure_computed()
        return self._results["zscore"]

    def by_isolation_forest(self) -> dict[str, Any]:
        self._ensure_computed()
        return self._results["isolation_forest"]

    def consensus(self, min_methods: int = 2) -> dict[str, Any]:
        self._ensure_computed()
        return self._results["consensus"]

    def comparison(self) -> pd.DataFrame:
        self._ensure_computed()
        rows = []
        for col in self._results["num_cols"]:
            rows.append({
                "column": col,
                "iqr_outliers": self._results["iqr"].get(col, {}).get("n_outliers", 0),
                "zscore_outliers": self._results["zscore"].get(col, {}).get("n_outliers", 0),
                "iso_forest_outliers": self._results["isolation_forest"].get(
                    "per_col_counts", {}
                ).get(col, 0),
                "consensus_outliers": self._results["consensus"].get(col, {}).get("n_outliers", 0),
            })
        return pd.DataFrame(rows)

    def plot(self, col: str | None = None):
        self._ensure_computed()
        if col is not None:
            return self._figures.get(f"{col}_box")
        return dict(self._figures)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_isolation_forest(
    df: pd.DataFrame, num_cols: list[str]
) -> dict[str, Any]:
    try:
        from sklearn.ensemble import IsolationForest

        X = df[num_cols].dropna()
        if len(X) < 10:
            return {}

        clf = IsolationForest(
            contamination=0.05, n_estimators=100, random_state=42, n_jobs=-1
        )
        preds = clf.fit_predict(X)
        outlier_idx = list(X.index[preds == -1])

        return {
            "n_outliers": int((preds == -1).sum()),
            "pct_outliers": round(100 * (preds == -1).mean(), 2),
            "outlier_indices": outlier_idx,
        }
    except ImportError:
        return {"note": "scikit-learn not available"}
    except Exception as e:
        return {"error": str(e)}


def _comparison_fig(
    iqr: dict, zscore: dict, consensus: dict
) -> go.Figure:
    cols = list(consensus.keys())
    fig = make_figure(title="Outlier Counts by Method")
    methods = [
        ("IQR", [iqr.get(c, {}).get("n_outliers", 0) for c in cols]),
        ("Z-score", [zscore.get(c, {}).get("n_outliers", 0) for c in cols]),
        ("Consensus", [consensus.get(c, {}).get("n_outliers", 0) for c in cols]),
    ]
    for i, (name, counts) in enumerate(methods):
        fig.add_trace(go.Bar(
            name=name, x=cols, y=counts,
            marker_color=QUALITATIVE[i],
        ))
    fig.update_layout(barmode="group", xaxis_title="Column",
                      yaxis_title="Outlier Count", height=400)
    return fig
