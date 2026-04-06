"""Stage 3: Multi-model training — iml.battle() entry point."""

from dissectml.battle.catalog import MODEL_CATALOG, ModelEntry, get_classifiers, get_regressors
from dissectml.battle.preprocessing import (
    PreprocessingPlan,
    build_full_pipeline,
    build_preprocessing_plan,
    build_transformer,
)
from dissectml.battle.registry import ModelRegistry, get_registry
from dissectml.battle.result import BattleResult, ModelScore
from dissectml.battle.runner import BattleRunner
from dissectml.battle.tuner import ModelTuner

__all__ = [
    "MODEL_CATALOG",
    "ModelEntry",
    "get_classifiers",
    "get_regressors",
    "PreprocessingPlan",
    "build_preprocessing_plan",
    "build_transformer",
    "build_full_pipeline",
    "ModelRegistry",
    "get_registry",
    "BattleResult",
    "ModelScore",
    "BattleRunner",
    "ModelTuner",
]


def battle(
    df,
    target: str,
    task: str | None = None,
    *,
    models=None,
    families=None,
    exclude=None,
    tune: bool = False,
    top_n: int = 3,
    n_iter: int = 20,
    cv: int | None = None,
    n_jobs: int | None = None,
    eda_result=None,
) -> BattleResult:
    """Train and compare multiple ML models via cross-validation.

    Args:
        df: Input DataFrame (features + target).
        target: Target column name.
        task: ``"classification"`` or ``"regression"``. Inferred if None.
        models: Explicit list of model names to include.
        families: Filter by family (e.g. ``["tree", "linear"]``).
        exclude: Model names to exclude.
        tune: If True, run RandomizedSearchCV on top-N models after battle.
        top_n: Number of models to tune when ``tune=True``.
        n_iter: Search iterations per model when ``tune=True``.
        cv: CV folds (default: config.battle_cv_folds = 5).
        n_jobs: Parallel workers (default: config.battle_n_jobs = -1).
        eda_result: Optional EDAResult from ``iml.explore()`` to inform preprocessing.

    Returns:
        :class:`BattleResult` with leaderboard and OOF predictions.

    Example::

        import dissectml as iml
        result = iml.battle(df, target="survived")
        result.leaderboard()
        result.best
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

    runner = BattleRunner(eda_result=eda_result)
    result = runner.run(
        df=df,
        target=target,
        task=task,
        models=models,
        families=families,
        exclude=exclude,
        cv=cv,
        n_jobs=n_jobs,
    )

    if tune:
        X = df.drop(columns=[target])
        y = df[target]
        tuner = ModelTuner(mode="tuned", top_n=top_n, n_iter=n_iter)
        result = tuner.tune(result, X, y)

    return result
