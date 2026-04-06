"""
InsightML — The missing middle layer between EDA and AutoML.

Unified pipeline from deep data understanding to model comparison,
in as few as 3 function calls.

Quick start::

    import insightml as iml

    # Deep EDA (v0.1+)
    eda = iml.explore(df)
    eda.overview.show()
    eda.correlations.heatmap()

    # Model battle (v0.2+)
    models = iml.battle(df, target="price")

    # Full pipeline (v0.4+)
    report = iml.analyze(df, target="price")
    report.export("report.html")
"""

from __future__ import annotations

from insightml._compat import to_pandas
from insightml._config import InsightMLConfig, config_context, get_config, set_config
from insightml._version import __version__
from insightml.battle import battle
from insightml.compare import ModelComparator
from insightml.datasets import load_housing, load_titanic
from insightml.eda import explore
from insightml.exceptions import InsightMLError
from insightml.intelligence import analyze_intelligence
from insightml.report import AnalysisReport

__all__ = [
    "__version__",
    # Public API
    "explore",
    "battle",
    "analyze_intelligence",
    "analyze",
    "ModelComparator",
    "AnalysisReport",
    # Datasets
    "load_titanic",
    "load_housing",
    # Compat
    "to_pandas",
    # Config
    "InsightMLConfig",
    "get_config",
    "set_config",
    "config_context",
    # Exceptions
    "InsightMLError",
]


def analyze(
    df,
    target: str,
    *,
    task: str | None = None,
    run_battle: bool = True,
    battle_families: list[str] | None = None,
    battle_models: list[str] | None = None,
    battle_exclude: list[str] | None = None,
    cv: int | None = None,
    n_jobs: int | None = None,
    datetime_col: str | None = None,
) -> AnalysisReport:
    """Full pipeline: EDA → Intelligence → Battle → Compare → Report.

    Runs all five stages and returns an :class:`~insightml.report.AnalysisReport`
    that can be inspected interactively or exported to HTML.

    Args:
        df: Input DataFrame (features + target column).
        target: Name of the target column.
        task: ``"classification"`` or ``"regression"``. Inferred if None.
        run_battle: If False, skip Stages 3-4 (EDA + Intelligence only).
        battle_families: Filter models by family for the battle stage.
        battle_models: Explicit model names for the battle stage.
        battle_exclude: Model names to exclude from battle.
        cv: CV folds for the battle stage.
        n_jobs: Parallel workers for the battle stage.
        datetime_col: Optional datetime column for temporal leakage detection.

    Returns:
        :class:`~insightml.report.AnalysisReport`

    Example::

        import insightml as iml
        report = iml.analyze(df, target="survived")
        report.summary()
        report.export("report.html")
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    feature_cols = [c for c in df.columns if c != target]
    n_samples = len(df)
    n_features = len(feature_cols)

    # Infer task if not provided
    if task is None or task == "auto":
        from insightml.battle.runner import _infer_task
        task = _infer_task(df[target])

    # Stage 1: EDA
    eda_result = explore(df, target=target)

    # Stage 2: Intelligence
    intel_result = analyze_intelligence(
        df, target=target, task=task,
        datetime_col=datetime_col,
        eda_result=eda_result,
    )

    if not run_battle:
        return AnalysisReport(
            task=task, target=target,
            n_samples=n_samples, n_features=n_features,
            eda=eda_result, intelligence=intel_result,
        )

    # Stage 3: Battle
    battle_result = battle(
        df, target=target, task=task,
        families=battle_families,
        models=battle_models,
        exclude=battle_exclude,
        cv=cv,
        n_jobs=n_jobs,
        eda_result=eda_result,
    )

    # Stage 4: Compare
    X = df.drop(columns=[target])
    y = df[target]
    comparator = ModelComparator(battle_result, X=X, y=y)

    return AnalysisReport(
        task=task, target=target,
        n_samples=n_samples, n_features=n_features,
        eda=eda_result, intelligence=intel_result,
        models=battle_result, compare=comparator,
    )
