"""ComparisonTable — styled leaderboard with per-column best-value highlights."""

from __future__ import annotations

from typing import Any

import pandas as pd

from insightml.battle.result import BattleResult


class ComparisonTable:
    """Styled leaderboard wrapping a BattleResult.

    Usage::

        ct = ComparisonTable(battle_result)
        ct.dataframe()          # plain DataFrame
        ct.styled()             # pandas Styler with bold best values
        ct.to_latex()           # LaTeX tabular string
        ct.to_markdown()        # Markdown table string
    """

    def __init__(self, battle_result: BattleResult) -> None:
        self._result = battle_result

    def dataframe(self, include_std: bool = False) -> pd.DataFrame:
        """Return leaderboard as a plain DataFrame.

        Args:
            include_std: If True, append ±std columns for each metric.

        Returns:
            DataFrame sorted by primary metric descending.
        """
        lb = self._result.leaderboard()
        if lb.empty:
            return lb

        if include_std:
            std_rows = []
            for s in self._result.successful:
                row: dict[str, Any] = {"model": s.name}
                row.update({f"{k}_std": v for k, v in s.metrics_std.items()})
                std_rows.append(row)
            std_df = pd.DataFrame(std_rows)
            lb = lb.merge(std_df, on="model", how="left")

        return lb

    def styled(self) -> Any:
        """Return a pandas Styler with best-value highlighting.

        Best metric values are shown in bold; background gradient per column.
        Requires pandas >= 1.3.
        """
        df = self.dataframe()
        if df.empty:
            return df

        metric_cols = [
            c for c in df.columns
            if c not in ("model", "task", "error", "train_time_s", "predict_time_s")
        ]
        time_cols = [c for c in df.columns if c in ("train_time_s", "predict_time_s")]

        # Higher-is-better metrics (all except error-based)
        higher_better = [
            c for c in metric_cols
            if not any(c.startswith(p) for p in ("mean_absolute", "mean_squared", "root_mean"))
        ]
        lower_better = [c for c in metric_cols if c not in higher_better]

        styler = df.style.format(
            {c: "{:.4f}" for c in metric_cols + time_cols if c in df.columns},
            na_rep="—",
        )

        for col in higher_better:
            if col in df.columns:
                styler = styler.background_gradient(subset=[col], cmap="Greens", low=0.2)
        for col in lower_better:
            if col in df.columns:
                styler = styler.background_gradient(subset=[col], cmap="Reds_r", low=0.2)
        for col in time_cols:
            if col in df.columns:
                styler = styler.background_gradient(subset=[col], cmap="Blues_r", low=0.2)

        # Bold best row
        primary = self._result.primary_metric
        if primary and primary in df.columns:
            best_idx = df[primary].idxmax()

            def bold_best(row):
                return ["font-weight: bold" if row.name == best_idx else "" for _ in row]

            styler = styler.apply(bold_best, axis=1)

        return styler

    def to_latex(self) -> str:
        """Return a LaTeX tabular string."""
        df = self.dataframe()
        if df.empty:
            return ""
        return df.to_latex(index=False, float_format="{:.4f}".format)

    def to_markdown(self) -> str:
        """Return a Markdown table string."""
        df = self.dataframe()
        if df.empty:
            return ""
        return df.to_markdown(index=False, floatfmt=".4f")

    def _repr_html_(self) -> str:
        try:
            return self.styled().to_html()
        except Exception:
            return self.dataframe().to_html(index=False)

    def __repr__(self) -> str:
        df = self.dataframe()
        return f"ComparisonTable({len(df)} models, primary={self._result.primary_metric!r})"
