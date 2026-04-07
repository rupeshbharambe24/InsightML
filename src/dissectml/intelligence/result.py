"""IntelligenceResult — lazy orchestrator for all Stage 2 intelligence modules."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import pandas as pd

from dissectml._config import DissectMLConfig, get_config
from dissectml._types import LeakageWarning
from dissectml.intelligence.feature_importance import compute_feature_importance
from dissectml.intelligence.leakage import detect_leakage
from dissectml.intelligence.multicollinearity import (
    compute_condition_number,
    compute_vif,
    removal_recommendations,
)
from dissectml.intelligence.readiness import ReadinessResult, compute_readiness
from dissectml.intelligence.recommendations import RecommendationResult, recommend_algorithms


class IntelligenceResult:
    """Stage 2 intelligence output — lazy access to all analysis modules.

    All properties are lazily computed on first access via ``@cached_property``.

    Access::

        intel = dml.analyze_intelligence(df, target="price")
        intel.leakage              # list of LeakageWarning
        intel.vif                  # pd.DataFrame
        intel.feature_importance   # pd.DataFrame
        intel.readiness            # ReadinessResult
        intel.recommendations      # RecommendationResult
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        task: str | None = None,
        datetime_col: str | None = None,
        config: DissectMLConfig | None = None,
        eda_result: Any = None,
    ) -> None:
        self._df = df
        self._target = target
        self._task = task
        self._datetime_col = datetime_col
        self._config = config or get_config()
        self._eda_result = eda_result

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @cached_property
    def leakage(self) -> list[LeakageWarning]:
        """Leakage warnings from the 4-pronged scan."""
        if self._target is None:
            return []
        return detect_leakage(
            self._df,
            target=self._target,
            datetime_col=self._datetime_col,
            significance_level=self._config.significance_level,
        )

    @cached_property
    def vif(self) -> pd.DataFrame:
        """VIF table for all numeric feature columns."""
        feature_cols = [c for c in self._df.columns if c != self._target]
        numeric_cols = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(self._df[c])
        ]
        return compute_vif(self._df[feature_cols], numeric_cols=numeric_cols)

    @cached_property
    def condition_number(self) -> dict[str, Any]:
        """Condition number of the feature matrix."""
        feature_cols = [c for c in self._df.columns if c != self._target]
        numeric_cols = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(self._df[c])
        ]
        return compute_condition_number(self._df[feature_cols], numeric_cols=numeric_cols)

    @cached_property
    def multicollinearity_recommendations(self) -> list[dict[str, Any]]:
        """Removal recommendations for high-VIF features."""
        return removal_recommendations(
            vif_df=self.vif,
            df=self._df,
            target=self._target,
        )

    @cached_property
    def feature_importance(self) -> pd.DataFrame:
        """Composite feature importance ranking."""
        if self._target is None:
            return pd.DataFrame(columns=["feature", "composite_rank"])
        return compute_feature_importance(
            self._df, target=self._target, task=self._task
        )

    @cached_property
    def readiness(self) -> ReadinessResult:
        """Data readiness score 0-100 with grade and breakdown."""
        return compute_readiness(
            df=self._df,
            target=self._target,
            task=self._task,
            vif_df=self.vif,
        )

    @cached_property
    def recommendations(self) -> RecommendationResult:
        """Algorithm family recommendations based on data characteristics."""
        df = self._df
        target = self._target
        task = self._task

        n_samples = len(df)
        n_features = len(df.columns) - (1 if target else 0)

        # Gather signals from EDA or heuristics
        has_nonlinear = False
        has_high_cardinality = False
        has_missing = bool(df.isnull().any().any())
        minority_ratio = 1.0

        # Try to get richer signals from EDA result
        if self._eda_result is not None:
            try:
                # Non-linearity signal
                interactions = self._eda_result.interactions
                top = interactions.top_interactions(n=5)
                if isinstance(top, pd.DataFrame) and "nonlinear" in top.columns:
                    has_nonlinear = bool(top["nonlinear"].any())
            except Exception:
                pass
            try:
                # Cardinality signal
                overview = self._eda_result.overview
                summary_df = overview.to_dataframe() if hasattr(overview, "to_dataframe") else None
                if summary_df is not None and "type" in summary_df.columns:
                    has_high_cardinality = "HIGH_CARDINALITY" in summary_df["type"].values
            except Exception:
                pass

        # Heuristic cardinality check
        if not has_high_cardinality:
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 50:
                    has_high_cardinality = True
                    break

        # Imbalance for classification
        if target and target in df.columns:
            y = df[target]
            if y.nunique() <= 20:
                vc = y.value_counts()
                if vc.max() > 0:
                    minority_ratio = float(vc.min() / vc.max())

        return recommend_algorithms(
            n_samples=n_samples,
            n_features=n_features,
            task=task or "classification",
            has_nonlinear=has_nonlinear,
            has_high_cardinality=has_high_cardinality,
            has_missing=has_missing,
            is_small_dataset=n_samples < 500,
            is_large_dataset=n_samples > 100_000,
            minority_ratio=minority_ratio,
            readiness_score=self.readiness.score,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "=== Intelligence Analysis ===",
            f"Readiness: {self.readiness.score:.0f}/100 (Grade {self.readiness.grade})",
            f"Leakage warnings: {len(self.leakage)}",
            f"High VIF features: {(self.vif['vif'] >= 10).sum() if not self.vif.empty else 0}",
            f"Top algorithm: {self.recommendations.top(1)[0] if self.recommendations.ranked else 'N/A'}",
        ]
        if self.leakage:
            lines.append("Leakage detected in: " + ", ".join(w["column"] for w in self.leakage[:5]))
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return (
            f"<h2>IntelligenceResult</h2>"
            f"<pre>{self.summary()}</pre>"
            f"{self.readiness._repr_html_()}"
            f"{self.recommendations._repr_html_()}"
        )

    def __repr__(self) -> str:
        return (
            f"IntelligenceResult("
            f"readiness={self.readiness.score:.0f}, "
            f"leakage={len(self.leakage)}, "
            f"top_algo={self.recommendations.top(1)[0] if self.recommendations.ranked else 'N/A'}"
            f")"
        )
