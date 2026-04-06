"""MissingDataIntelligence — Little's MCAR test, MAR/MNAR classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2, ttest_ind

from insightml._types import MissingnessType
from insightml.eda._base import BaseAnalysisModule
from insightml.viz.theme import make_figure


class MissingDataIntelligence(BaseAnalysisModule):
    """Missing data analysis with statistical mechanism classification.

    Features:
    - Per-column missing counts and percentages
    - Missing pattern matrix (unique row-level null patterns)
    - Little's MCAR test (chi-square test on missing patterns)
    - Per-column MAR detection (t-test against other columns)
    - MNAR as residual category
    - Imputation recommendations per column

    Access::

        eda.missing.counts()          # DataFrame of missing stats
        eda.missing.patterns()        # unique missing patterns
        eda.missing.littles_test()    # {'p_value': ..., 'mechanism': 'MCAR'}
        eda.missing.classify()        # per-column MissingnessType
        eda.missing.recommendations() # imputation suggestions
    """

    def _compute(self) -> None:
        df = self._df
        config = self._config

        # --- Basic counts ---
        missing_counts = df.isnull().sum()
        missing_pct = 100 * df.isnull().mean()
        cols_with_missing = missing_counts[missing_counts > 0].index.tolist()

        counts_df = pd.DataFrame({
            "column": df.columns,
            "missing_count": missing_counts.values,
            "missing_pct": missing_pct.round(2).values,
            "present_count": df.count().values,
        })

        # --- Missing pattern matrix ---
        null_mask = df.isnull().astype(int)
        patterns = null_mask.drop_duplicates()
        pattern_counts = null_mask.apply(tuple, axis=1).value_counts()

        # --- Little's MCAR test ---
        littles_result = _littles_mcar_test(df)

        # --- Per-column mechanism classification ---
        classification: dict[str, MissingnessType] = {}
        for col in cols_with_missing:
            mech = _classify_column(df, col, config.significance_level)
            classification[col] = mech

        # --- Imputation recommendations ---
        recommendations = {
            col: _recommend_imputation(
                col, classification.get(col, MissingnessType.UNKNOWN),
                df[col].dtype
            )
            for col in cols_with_missing
        }

        self._results = {
            "counts_df": counts_df,
            "cols_with_missing": cols_with_missing,
            "total_missing": int(missing_counts.sum()),
            "total_missing_pct": round(100 * df.isnull().values.mean(), 2),
            "patterns": patterns,
            "pattern_counts": pattern_counts,
            "littles_result": littles_result,
            "classification": classification,
            "recommendations": recommendations,
            "null_mask": null_mask,
        }

        if len(cols_with_missing) == 0:
            self._warn("No missing values detected.")
        elif self._results["total_missing_pct"] > 20:
            self._warn(
                f"High overall missingness: {self._results['total_missing_pct']:.1f}% "
                "of all values are missing."
            )

    def _build_figures(self) -> dict[str, go.Figure]:
        figs: dict[str, go.Figure] = {}
        null_mask: pd.DataFrame = self._results["null_mask"]
        cols_with_missing = self._results["cols_with_missing"]

        if not cols_with_missing:
            return figs

        # Sort rows by total missingness
        null_subset = null_mask[cols_with_missing]
        row_order = null_subset.sum(axis=1).sort_values(ascending=False).index
        display_mask = null_subset.loc[row_order].head(200)  # cap at 200 rows for display

        fig_pattern = make_figure(title="Missing Value Patterns")
        fig_pattern.add_trace(go.Heatmap(
            z=display_mask.values.tolist(),
            x=cols_with_missing,
            y=[str(i) for i in range(len(display_mask))],
            colorscale=[[0, "#54a24b"], [1, "#e45756"]],
            showscale=True,
            colorbar={
                "tickvals": [0, 1],
                "ticktext": ["Present", "Missing"],
            },
            hoverongaps=False,
        ))
        fig_pattern.update_layout(
            xaxis_title="Column",
            yaxis_title="Row (sorted by missingness)",
            height=max(300, min(600, len(display_mask) * 3)),
        )
        figs["patterns"] = fig_pattern

        # Bar chart of missing %
        counts_df = self._results["counts_df"]
        has_missing = counts_df[counts_df["missing_count"] > 0].sort_values(
            "missing_pct", ascending=False
        )
        if len(has_missing) > 0:
            fig_bar = make_figure(title="Missing Values by Column")
            fig_bar.add_trace(go.Bar(
                x=has_missing["column"],
                y=has_missing["missing_pct"],
                marker_color=[
                    "#e45756" if p > 50 else "#f58518" if p > 20 else "#4c78a8"
                    for p in has_missing["missing_pct"]
                ],
                text=[f"{p:.1f}%" for p in has_missing["missing_pct"]],
                textposition="outside",
            ))
            fig_bar.update_layout(
                xaxis_title="Column", yaxis_title="Missing (%)",
                showlegend=False, height=350,
            )
            figs["missing_bar"] = fig_bar

        return figs

    def summary(self) -> str:
        r = self._results
        if r["total_missing"] == 0:
            return "No missing values in the dataset."
        lr = r["littles_result"]
        mech = lr.get("mechanism", "unknown")
        return (
            f"{r['total_missing_pct']:.1f}% of values missing across "
            f"{len(r['cols_with_missing'])} column(s). "
            f"Little's MCAR test: mechanism likely {mech} "
            f"(p={lr.get('p_value', float('nan')):.3f})."
        )

    # --- Public accessors ---

    def counts(self) -> pd.DataFrame:
        self._ensure_computed()
        return self._results["counts_df"]

    def patterns(self) -> pd.DataFrame:
        self._ensure_computed()
        return self._results["patterns"]

    def littles_test(self) -> dict[str, Any]:
        self._ensure_computed()
        return self._results["littles_result"]

    def classify(self) -> dict[str, MissingnessType]:
        self._ensure_computed()
        return self._results["classification"]

    def recommendations(self) -> dict[str, list[str]]:
        self._ensure_computed()
        return self._results["recommendations"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _littles_mcar_test(df: pd.DataFrame) -> dict[str, Any]:
    """Compute Little's MCAR test statistic."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_with_missing = [c for c in num_cols if df[c].isna().any()]

    if len(cols_with_missing) < 2:
        return {
            "statistic": None, "p_value": None,
            "mechanism": MissingnessType.UNKNOWN.value,
            "note": "Not enough numeric columns with missing values for MCAR test.",
        }

    sub = df[cols_with_missing].copy()
    grand_mean = sub.mean()
    cov = sub.cov()

    # Identify unique missing patterns
    null_pattern = sub.isnull()
    pattern_groups = sub.groupby(null_pattern.apply(tuple, axis=1))

    d2 = 0.0
    df_total = 0

    for pattern_key, group in pattern_groups:
        observed_cols = [c for c, is_null in zip(cols_with_missing, pattern_key)
                         if not is_null]
        if len(observed_cols) < 1:
            continue
        n_j = len(group)
        mean_j = group[observed_cols].mean()
        grand_mean_j = grand_mean[observed_cols]
        cov_j = cov.loc[observed_cols, observed_cols]
        diff = (mean_j - grand_mean_j).values.reshape(-1)
        try:
            cov_inv = np.linalg.pinv(cov_j.values)
            d2 += n_j * float(diff @ cov_inv @ diff)
            df_total += len(observed_cols)
        except Exception:
            continue

    df_chi2 = max(df_total - len(cols_with_missing), 1)
    if d2 <= 0 or df_chi2 <= 0:
        return {
            "statistic": None, "p_value": None,
            "mechanism": MissingnessType.UNKNOWN.value,
            "note": "Could not compute test statistic.",
        }

    p_value = float(1 - chi2.cdf(d2, df_chi2))
    mechanism = MissingnessType.MCAR.value if p_value > 0.05 else MissingnessType.MAR.value

    return {
        "statistic": round(d2, 4),
        "df": df_chi2,
        "p_value": round(p_value, 4),
        "mechanism": mechanism,
        "interpretation": (
            "Fail to reject MCAR (p > 0.05)" if p_value > 0.05
            else "Reject MCAR — likely MAR or MNAR (p ≤ 0.05)"
        ),
    }


def _classify_column(
    df: pd.DataFrame, col: str, alpha: float
) -> MissingnessType:
    """Classify missingness mechanism for one column."""
    missing_mask = df[col].isna()
    if missing_mask.sum() == 0:
        return MissingnessType.UNKNOWN

    for other_col in df.columns:
        if other_col == col:
            continue
        other = df[other_col].dropna()
        if len(other) < 5:
            continue
        if pd.api.types.is_numeric_dtype(df[other_col]):
            group_present = df.loc[~missing_mask, other_col].dropna()
            group_missing = df.loc[missing_mask, other_col].dropna()
            if len(group_present) < 3 or len(group_missing) < 3:
                continue
            try:
                _, p = ttest_ind(group_present, group_missing, equal_var=False)
                if p < alpha:
                    return MissingnessType.MAR
            except Exception:
                pass

    return MissingnessType.MCAR


def _recommend_imputation(
    col: str, mechanism: MissingnessType, dtype
) -> list[str]:
    is_numeric = pd.api.types.is_numeric_dtype(dtype)
    recs: dict[tuple, list[str]] = {
        (MissingnessType.MCAR, True):  ["median", "mean", "KNNImputer"],
        (MissingnessType.MCAR, False): ["mode", "KNNImputer"],
        (MissingnessType.MAR, True):   ["KNNImputer", "IterativeImputer (MICE)"],
        (MissingnessType.MAR, False):  ["KNNImputer", "IterativeImputer"],
        (MissingnessType.MNAR, True):  ["add missing-indicator column", "domain imputation"],
        (MissingnessType.MNAR, False): ["add missing-indicator column", "domain imputation"],
        (MissingnessType.UNKNOWN, True):  ["median", "KNNImputer"],
        (MissingnessType.UNKNOWN, False): ["mode", "KNNImputer"],
    }
    return recs.get((mechanism, is_numeric), ["investigate manually"])
