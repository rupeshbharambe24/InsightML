"""StatisticalTests — normality, independence, variance, group comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from insightml._types import ColumnType
from insightml.core.validators import infer_column_type
from insightml.eda._base import BaseAnalysisModule
from insightml.viz.theme import QUALITATIVE, make_figure


class StatisticalTests(BaseAnalysisModule):
    """Statistical hypothesis tests for EDA.

    **Normality** (per numeric column, consensus of 3 tests):
    - Shapiro-Wilk (sample ≤5000)
    - D'Agostino-Pearson (n ≥20)
    - Anderson-Darling

    **Independence**: Chi-square on categorical pairs.

    **Variance**: Levene's (robust) + Bartlett's (assumes normality).

    **Group comparison** (auto-selects):
    - All groups normal + Levene passes → ANOVA
    - Otherwise → Kruskal-Wallis

    Access::

        eda.tests.normality()              # all numeric columns
        eda.tests.normality("age")         # single column
        eda.tests.independence("sex", "survived")
        eda.tests.variance("age", "sex")
        eda.tests.group_comparison("fare", "pclass")
    """

    def _compute(self) -> None:
        df = self._df
        config = self._config

        num_cols = [
            c for c in df.columns
            if infer_column_type(df[c], config) == ColumnType.NUMERIC
        ]
        cat_cols = [
            c for c in df.columns
            if infer_column_type(df[c], config) in (
                ColumnType.CATEGORICAL, ColumnType.BOOLEAN
            )
        ]

        # Normality for every numeric column
        normality_results: dict[str, Any] = {}
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 8:
                continue
            normality_results[col] = _test_normality(series, config.significance_level)

        # Auto group comparison: each numeric vs each categorical (if target given)
        group_results: dict[str, Any] = {}
        if self._target and self._target in cat_cols:
            for col in num_cols:
                if col == self._target:
                    continue
                key = f"{col}|{self._target}"
                group_results[key] = self.group_comparison(col, self._target, _run=True)

        self._results = {
            "normality": normality_results,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "group_results": group_results,
        }

        non_normal = [c for c, r in normality_results.items()
                      if not r.get("is_normal", True)]
        if non_normal:
            self._warn(
                f"{len(non_normal)} column(s) failed normality tests: {non_normal[:5]}"
                + (" ..." if len(non_normal) > 5 else "")
            )

    def _build_figures(self) -> dict[str, go.Figure]:
        df = self._df
        figs: dict[str, go.Figure] = {}
        for col in self._results.get("num_cols", []):
            series = df[col].dropna()
            if len(series) >= 8:
                figs[f"{col}_qq"] = _qq_plot(series, col)
        return figs

    def summary(self) -> str:
        nr = self._results.get("normality", {})
        n_tested = len(nr)
        n_normal = sum(1 for r in nr.values() if r.get("is_normal", False))
        return (
            f"Normality tests on {n_tested} numeric column(s): "
            f"{n_normal} pass, {n_tested - n_normal} fail."
        )

    # --- Public methods ---

    def normality(self, col: str | None = None) -> dict[str, Any] | pd.DataFrame:
        """Return normality test results for one or all numeric columns."""
        self._ensure_computed()
        if col is not None:
            return self._results["normality"].get(col, {})
        return pd.DataFrame(self._results["normality"]).T

    def independence(self, col_a: str, col_b: str) -> dict[str, Any]:
        """Chi-square test of independence between two categorical columns."""
        df = self._df
        mask = df[col_a].notna() & df[col_b].notna()
        if mask.sum() < 5:
            return {"error": "Not enough data"}
        ct = pd.crosstab(df.loc[mask, col_a], df.loc[mask, col_b])
        chi2_stat, p_value, dof, _ = stats.chi2_contingency(ct, correction=False)
        n = int(ct.sum().sum())
        cramers_v = float(np.sqrt(chi2_stat / (n * max(min(ct.shape) - 1, 1))))
        return {
            "test": "chi2_contingency",
            "chi2": round(float(chi2_stat), 4),
            "p_value": round(float(p_value), 4),
            "dof": int(dof),
            "cramers_v": round(cramers_v, 4),
            "significant": float(p_value) < self._config.significance_level,
        }

    def variance(self, num_col: str, group_col: str) -> dict[str, Any]:
        """Levene's and Bartlett's tests for equality of variance across groups."""
        df = self._df
        mask = df[num_col].notna() & df[group_col].notna()
        groups = [
            df.loc[mask & (df[group_col] == g), num_col].values
            for g in df.loc[mask, group_col].unique()
            if len(df.loc[mask & (df[group_col] == g), num_col]) >= 2
        ]
        if len(groups) < 2:
            return {"error": "Need at least 2 groups"}
        levene_stat, levene_p = stats.levene(*groups)
        bartlett_stat, bartlett_p = stats.bartlett(*groups)
        alpha = self._config.significance_level
        return {
            "levene": {
                "statistic": round(float(levene_stat), 4),
                "p_value": round(float(levene_p), 4),
                "equal_variances": float(levene_p) >= alpha,
            },
            "bartlett": {
                "statistic": round(float(bartlett_stat), 4),
                "p_value": round(float(bartlett_p), 4),
                "equal_variances": float(bartlett_p) >= alpha,
            },
        }

    def group_comparison(
        self, num_col: str, group_col: str, *, _run: bool = False
    ) -> dict[str, Any]:
        """Auto-select ANOVA or Kruskal-Wallis for group comparison."""
        if not _run:
            self._ensure_computed()
        df = self._df
        alpha = self._config.significance_level
        mask = df[num_col].notna() & df[group_col].notna()
        groups = {
            str(g): df.loc[mask & (df[group_col] == g), num_col].values
            for g in df.loc[mask, group_col].unique()
        }
        groups = {k: v for k, v in groups.items() if len(v) >= 3}
        if len(groups) < 2:
            return {"error": "Need at least 2 groups with ≥3 samples each"}

        # Normality check
        all_normal = all(
            _test_normality(pd.Series(v), alpha).get("is_normal", False)
            for v in groups.values()
        )
        # Levene's test
        levene_stat, levene_p = stats.levene(*groups.values())
        equal_var = float(levene_p) >= alpha

        if all_normal and equal_var:
            f_stat, p_value = stats.f_oneway(*groups.values())
            return {
                "test_used": "ANOVA",
                "reason": "All groups normal + equal variances (Levene p={:.3f})".format(levene_p),
                "statistic": round(float(f_stat), 4),
                "p_value": round(float(p_value), 4),
                "significant": float(p_value) < alpha,
                "groups": list(groups.keys()),
            }
        else:
            h_stat, p_value = stats.kruskal(*groups.values())
            reason = (
                "Non-normal distribution" if not all_normal
                else "Unequal variances (Levene p={:.3f})".format(levene_p)
            )
            return {
                "test_used": "Kruskal-Wallis",
                "reason": reason,
                "statistic": round(float(h_stat), 4),
                "p_value": round(float(p_value), 4),
                "significant": float(p_value) < alpha,
                "groups": list(groups.keys()),
            }

    def all_tests(self) -> dict[str, Any]:
        """Return all computed test results."""
        self._ensure_computed()
        return {
            "normality": self._results["normality"],
            "group_comparisons": self._results["group_results"],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _test_normality(series: pd.Series, alpha: float) -> dict[str, Any]:
    data = series.dropna().values
    n = len(data)
    results: dict[str, Any] = {"n": n}
    votes_normal = 0
    votes_total = 0

    # Shapiro-Wilk (sample if large)
    if n >= 8:
        sample = data if n <= 5000 else np.random.default_rng(42).choice(
            data, 5000, replace=False
        )
        try:
            sw_stat, sw_p = stats.shapiro(sample)
            results["shapiro_wilk"] = {
                "statistic": round(float(sw_stat), 4),
                "p_value": round(float(sw_p), 4),
                "normal": float(sw_p) >= alpha,
            }
            votes_total += 1
            if float(sw_p) >= alpha:
                votes_normal += 1
        except Exception:
            pass

    # D'Agostino-Pearson
    if n >= 20:
        try:
            dp_stat, dp_p = stats.normaltest(data)
            results["dagostino_pearson"] = {
                "statistic": round(float(dp_stat), 4),
                "p_value": round(float(dp_p), 4),
                "normal": float(dp_p) >= alpha,
            }
            votes_total += 1
            if float(dp_p) >= alpha:
                votes_normal += 1
        except Exception:
            pass

    # Anderson-Darling (method='interpolate' keeps backward-compatible behaviour)
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", FutureWarning)
            ad_result = stats.anderson(data, dist="norm")
        # critical_values index 2 = 5% significance level
        ad_critical = float(ad_result.critical_values[2])
        ad_normal = float(ad_result.statistic) < ad_critical
        results["anderson_darling"] = {
            "statistic": round(float(ad_result.statistic), 4),
            "critical_5pct": round(float(ad_critical), 4),
            "normal": ad_normal,
        }
        votes_total += 1
        if ad_normal:
            votes_normal += 1
    except Exception:
        pass

    # Consensus: majority vote
    results["is_normal"] = (votes_normal >= votes_total / 2) if votes_total > 0 else None
    results["votes_normal"] = votes_normal
    results["votes_total"] = votes_total
    return results


def _qq_plot(series: pd.Series, col_name: str) -> go.Figure:
    data = series.dropna().values
    (osm, osr), (slope, intercept, _) = stats.probplot(data, dist="norm")
    fig = make_figure(title=f"{col_name} — Q-Q Plot (Normal)")
    fig.add_trace(go.Scatter(
        x=list(osm), y=list(osr),
        mode="markers",
        name="Sample quantiles",
        marker=dict(color=QUALITATIVE[0], size=4, opacity=0.7),
    ))
    x_line = [min(osm), max(osm)]
    fig.add_trace(go.Scatter(
        x=x_line,
        y=[slope * x + intercept for x in x_line],
        mode="lines", name="Normal line",
        line=dict(color=QUALITATIVE[1], width=2, dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=350,
    )
    return fig
