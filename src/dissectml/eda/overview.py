"""DataOverview — type detection, column profiles, dataset-level stats."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dissectml._types import ColumnProfile, ColumnType
from dissectml.core.validators import infer_column_type
from dissectml.eda._base import BaseAnalysisModule
from dissectml.viz.theme import QUALITATIVE, make_figure


class DataOverview(BaseAnalysisModule):
    """Dataset overview: type detection, per-column profiles, memory usage.

    Always computed on the **full** DataFrame (not the sample), so missing
    counts and shape statistics are exact.

    Access::

        eda.overview.show()
        eda.overview.to_dataframe()   # one row per column
        eda.overview.column_types     # dict[col_name -> ColumnType]
    """

    def _compute(self) -> None:
        df = self._df
        n_rows, n_cols = df.shape
        config = self._config

        profiles: list[ColumnProfile] = []
        col_types: dict[str, ColumnType] = {}

        for col_name in df.columns:
            col = df[col_name]
            ct = infer_column_type(col, config)
            col_types[col_name] = ct

            profile: ColumnProfile = {
                "name": col_name,
                "dtype": str(col.dtype),
                "inferred_type": ct.value,
                "count": int(col.count()),
                "unique": int(col.nunique(dropna=True)),
                "missing_count": int(col.isna().sum()),
                "missing_pct": round(100 * col.isna().mean(), 2),
                "memory_bytes": int(col.memory_usage(deep=True)),
            }

            if ct == ColumnType.NUMERIC:
                non_null = col.dropna()
                profile.update({
                    "mean": float(non_null.mean()),
                    "median": float(non_null.median()),
                    "std": float(non_null.std()),
                    "variance": float(non_null.var()),
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "range": float(non_null.max() - non_null.min()),
                    "q1": float(non_null.quantile(0.25)),
                    "q3": float(non_null.quantile(0.75)),
                    "iqr": float(non_null.quantile(0.75) - non_null.quantile(0.25)),
                    "skewness": float(non_null.skew()),
                    "kurtosis": float(non_null.kurtosis()),
                })

            elif ct in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN,
                        ColumnType.HIGH_CARDINALITY):
                vc = col.value_counts()
                profile.update({
                    "top_value": str(vc.index[0]) if len(vc) > 0 else None,
                    "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
                    "cardinality_ratio": round(col.nunique() / max(len(col), 1), 4),
                    "value_counts": {str(k): int(v)
                                     for k, v in vc.head(20).items()},
                })

            elif ct == ColumnType.DATETIME:
                non_null = col.dropna()
                if pd.api.types.is_datetime64_any_dtype(non_null):
                    profile.update({
                        "dt_min": str(non_null.min()),
                        "dt_max": str(non_null.max()),
                        "range_days": float(
                            (non_null.max() - non_null.min()).days
                        ),
                        "inferred_frequency": _infer_freq(non_null),
                    })

            profiles.append(profile)

        # Dataset-level stats
        n_duplicates = int(df.duplicated().sum())
        total_memory_mb = round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3)
        type_counts = {
            ct.value: sum(1 for t in col_types.values() if t == ct)
            for ct in ColumnType
            if any(t == ct for t in col_types.values())
        }

        self._results = {
            "shape": (n_rows, n_cols),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_duplicates": n_duplicates,
            "total_memory_mb": total_memory_mb,
            "column_types": col_types,
            "type_counts": type_counts,
            "profiles": profiles,
        }

        # Warn about quality issues
        high_missing = [p["name"] for p in profiles if p["missing_pct"] > 50]
        if high_missing:
            self._warn(f"{len(high_missing)} column(s) >50% missing: {high_missing}")
        const_cols = [p["name"] for p in profiles
                      if p["inferred_type"] == ColumnType.CONSTANT.value]
        if const_cols:
            self._warn(f"Constant columns detected: {const_cols}")

    def _build_figures(self) -> dict[str, go.Figure]:
        type_counts: dict[str, int] = self._results["type_counts"]
        profiles: list[ColumnProfile] = self._results["profiles"]

        # --- Figure 1: Column type distribution ---
        fig_types = make_figure(title="Column Types")
        labels = list(type_counts.keys())
        values = list(type_counts.values())
        colors = QUALITATIVE[: len(labels)]
        fig_types.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=values, textposition="outside",
        ))
        fig_types.update_layout(
            xaxis_title="Column Type", yaxis_title="Count",
            showlegend=False, height=350,
        )

        # --- Figure 2: Missing values per column ---
        missing_data = [
            (p["name"], p["missing_pct"])
            for p in profiles
            if p["missing_pct"] > 0
        ]
        figs = {"column_types": fig_types}

        if missing_data:
            missing_data.sort(key=lambda x: x[1], reverse=True)
            cols_m, pcts_m = zip(*missing_data)
            fig_missing = make_figure(title="Missing Values (%)")
            fig_missing.add_trace(go.Bar(
                x=list(cols_m), y=list(pcts_m),
                marker_color=[
                    "#e45756" if p > 50 else "#f58518" if p > 20 else "#4c78a8"
                    for p in pcts_m
                ],
                text=[f"{p:.1f}%" for p in pcts_m],
                textposition="outside",
            ))
            fig_missing.update_layout(
                xaxis_title="Column", yaxis_title="Missing (%)",
                showlegend=False, height=350,
            )
            figs["missing_overview"] = fig_missing

        return figs

    def summary(self) -> str:
        self._ensure_computed()
        r = self._results
        n_rows, n_cols = r["shape"]
        tc = r["type_counts"]
        type_str = ", ".join(f"{v} {k}" for k, v in tc.items())
        missing_cols = sum(
            1 for p in r["profiles"] if p["missing_pct"] > 0
        )
        return (
            f"Dataset has {n_rows:,} rows and {n_cols} columns ({type_str}). "
            f"{r['n_duplicates']} duplicate rows. "
            f"{missing_cols} columns have missing values. "
            f"Total memory: {r['total_memory_mb']} MB."
        )

    # --- Convenience accessors ---

    @property
    def column_types(self) -> dict[str, ColumnType]:
        self._ensure_computed()
        return self._results["column_types"]

    @property
    def profiles(self) -> list[ColumnProfile]:
        self._ensure_computed()
        return self._results["profiles"]

    def to_dataframe(self) -> pd.DataFrame:
        self._ensure_computed()
        rows = []
        for p in self._results["profiles"]:
            rows.append({
                k: v for k, v in p.items()
                if k not in ("value_counts",)
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_freq(series: pd.Series) -> str | None:
    """Try to infer a string frequency label (e.g. 'D', 'M') from a datetime Series."""
    try:
        diffs = series.sort_values().diff().dropna()
        if len(diffs) == 0:
            return None
        median_diff = diffs.median()
        days = median_diff.days
        if days == 0:
            return "sub-daily"
        if days == 1:
            return "daily"
        if 6 <= days <= 8:
            return "weekly"
        if 28 <= days <= 32:
            return "monthly"
        if 88 <= days <= 93:
            return "quarterly"
        if 360 <= days <= 370:
            return "yearly"
        return f"~{days}d"
    except Exception:
        return None
