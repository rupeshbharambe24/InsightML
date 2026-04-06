"""Stage 2: Pre-model intelligence — leakage, multicollinearity, readiness, recommendations."""

from dissectml.intelligence.feature_importance import compute_feature_importance
from dissectml.intelligence.leakage import detect_leakage
from dissectml.intelligence.multicollinearity import (
    compute_condition_number,
    compute_vif,
    removal_recommendations,
)
from dissectml.intelligence.readiness import ReadinessResult, compute_readiness
from dissectml.intelligence.recommendations import RecommendationResult, recommend_algorithms
from dissectml.intelligence.result import IntelligenceResult

__all__ = [
    "IntelligenceResult",
    "detect_leakage",
    "compute_vif",
    "compute_condition_number",
    "removal_recommendations",
    "compute_feature_importance",
    "compute_readiness",
    "ReadinessResult",
    "recommend_algorithms",
    "RecommendationResult",
]


def analyze_intelligence(
    df,
    target: str | None = None,
    task: str | None = None,
    datetime_col: str | None = None,
    eda_result=None,
) -> IntelligenceResult:
    """Run Stage 2 intelligence analysis on a DataFrame.

    Returns an :class:`IntelligenceResult` with lazy-computed sub-modules.

    Args:
        df: Input DataFrame.
        target: Target column name.
        task: ``"classification"`` or ``"regression"``. Inferred if None.
        datetime_col: Optional datetime column for temporal leakage check.
        eda_result: Optional output of ``iml.explore()`` for richer signals.

    Returns:
        :class:`IntelligenceResult`

    Example::

        import dissectml as iml
        intel = iml.analyze_intelligence(df, target="survived")
        intel.readiness.summary()
        intel.leakage
        intel.recommendations.top(3)
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

    return IntelligenceResult(
        df=df,
        target=target,
        task=task,
        datetime_col=datetime_col,
        eda_result=eda_result,
    )
