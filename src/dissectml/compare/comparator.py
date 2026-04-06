"""ModelComparator — lazy facade over all Stage 4 comparison modules."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dissectml.battle.result import BattleResult
from dissectml.compare.curves import (
    actual_vs_predicted,
    confusion_matrices,
    metric_bar_chart,
    pr_curves,
    residual_plots,
    roc_curves,
)
from dissectml.compare.error_analysis import ErrorAnalysisResult, analyze_errors
from dissectml.compare.metrics_table import ComparisonTable
from dissectml.compare.pareto import get_pareto_models, pareto_front
from dissectml.compare.significance import corrected_ttest_matrix, mcnemar_matrix


class ModelComparator:
    """Lazy facade over all comparative analysis results.

    All properties are computed on first access via ``@cached_property``.

    Usage::

        comp = ModelComparator(battle_result, X=X_df, y=y_series)
        comp.table                      # ComparisonTable
        comp.pareto_front               # Plotly Figure
        comp.error_analysis             # ErrorAnalysisResult
        comp.significance["p_matrix"]  # McNemar p-value DataFrame
        comp.roc_curves                 # Plotly Figure (classification)
        comp.residual_plots             # Plotly Figure (regression)
    """

    def __init__(
        self,
        battle_result: BattleResult,
        X: pd.DataFrame | None = None,
        y: pd.Series | np.ndarray | None = None,
    ) -> None:
        self._result = battle_result
        self._X = X
        self._y = y
        self._task = battle_result.task

    # ------------------------------------------------------------------
    # Core views
    # ------------------------------------------------------------------

    @cached_property
    def table(self) -> ComparisonTable:
        """Styled leaderboard table."""
        return ComparisonTable(self._result)

    @cached_property
    def pareto(self) -> go.Figure:
        """Pareto front figure (accuracy vs training time)."""
        return pareto_front(self._result)

    @cached_property
    def pareto_models(self) -> list[str]:
        """Names of Pareto-optimal models."""
        return get_pareto_models(self._result)

    @cached_property
    def metric_bar(self) -> go.Figure:
        """Bar chart of primary metric per model."""
        return metric_bar_chart(self._result)

    # ------------------------------------------------------------------
    # Significance tests
    # ------------------------------------------------------------------

    @cached_property
    def significance(self) -> dict[str, Any]:
        """Statistical significance matrices.

        Returns dict with keys:
        - ``mcnemar``: dict(p_matrix, figure) — classification only
        - ``ttest``: dict(p_matrix, figure) — classification + regression
        """
        if self._y is None:
            return {}

        result: dict[str, Any] = {}
        if self._task == "classification":
            result["mcnemar"] = mcnemar_matrix(self._result, self._y)
        result["ttest"] = corrected_ttest_matrix(self._result, self._y)
        return result

    # ------------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------------

    @cached_property
    def error_analysis(self) -> ErrorAnalysisResult:
        """Cross-model disagreement and hard sample analysis."""
        if self._y is None:
            from dissectml.compare.error_analysis import ErrorAnalysisResult
            return ErrorAnalysisResult(
                task=self._task, models=[], disagreement=pd.DataFrame(),
                complementarity=pd.DataFrame(), hard_indices=np.array([]),
                hard_sample_profile=pd.DataFrame(),
            )
        return analyze_errors(self._result, self._y, X=self._X)

    # ------------------------------------------------------------------
    # Visualisation curves
    # ------------------------------------------------------------------

    @cached_property
    def roc_curves(self) -> go.Figure | None:
        """ROC curves (classification only)."""
        if self._task != "classification" or self._y is None:
            return None
        return roc_curves(self._result, self._y)

    @cached_property
    def pr_curves(self) -> go.Figure | None:
        """Precision-Recall curves (classification only)."""
        if self._task != "classification" or self._y is None:
            return None
        return pr_curves(self._result, self._y)

    @cached_property
    def confusion_matrices(self) -> go.Figure | None:
        """Grid of confusion matrices (classification only)."""
        if self._task != "classification" or self._y is None:
            return None
        return confusion_matrices(self._result, self._y)

    @cached_property
    def residual_plots(self) -> go.Figure | None:
        """Grid of residual plots (regression only)."""
        if self._task != "regression" or self._y is None:
            return None
        return residual_plots(self._result, self._y)

    @cached_property
    def actual_vs_predicted(self) -> go.Figure | None:
        """Actual vs predicted scatter (regression only)."""
        if self._task != "regression" or self._y is None:
            return None
        return actual_vs_predicted(self._result, self._y)

    # ------------------------------------------------------------------
    # SHAP (optional)
    # ------------------------------------------------------------------

    def shap_comparison(self, top_n: int = 3) -> dict[str, Any]:
        """SHAP importance comparison for top-N models.

        Requires ``pip install dissectml[explain]``.
        """
        if self._X is None:
            raise ValueError("X must be provided to ModelComparator for SHAP analysis.")
        from dissectml.compare.shap_compare import shap_comparison
        return shap_comparison(self._result, self._X, top_n=top_n)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        best = self._result.best
        pareto = self.pareto_models
        return (
            f"ModelComparator ({self._task})\n"
            f"  Models: {len(self._result.successful)} successful, {len(self._result.failed)} failed\n"
            f"  Best: {best.name if best else 'N/A'} "
            f"({self._result.primary_metric}={best.primary_metric:.4f})\n"
            f"  Pareto optimal: {pareto}"
            if best else
            "ModelComparator: no successful models"
        )

    def _repr_html_(self) -> str:
        return (
            "<h2>ModelComparator</h2>"
            f"<pre>{self.summary()}</pre>"
            f"{self.table._repr_html_()}"
        )

    def __repr__(self) -> str:
        return (
            f"ModelComparator(task={self._task!r}, "
            f"models={len(self._result.successful)}, "
            f"pareto={self.pareto_models})"
        )
