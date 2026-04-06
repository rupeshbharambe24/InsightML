"""BattleRunner — parallel cross-validation training for model comparison."""

from __future__ import annotations

import time
import traceback
import warnings
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from dissectml._config import InsightMLConfig, get_config
from dissectml.battle.catalog import ModelEntry
from dissectml.battle.preprocessing import (
    PreprocessingPlan,
    build_full_pipeline,
    build_preprocessing_plan,
)
from dissectml.battle.registry import ModelRegistry, get_registry
from dissectml.battle.result import BattleResult, ModelScore

# ---------------------------------------------------------------------------
# Default metrics
# ---------------------------------------------------------------------------

_CLASSIFICATION_METRICS = [
    "accuracy",
    "f1_weighted",
    "precision_weighted",
    "recall_weighted",
    "roc_auc_ovr_weighted",
]

_REGRESSION_METRICS = [
    "r2",
    "neg_mean_absolute_error",
    "neg_root_mean_squared_error",
    "neg_mean_absolute_percentage_error",
]

_PRIMARY_METRIC: dict[str, str] = {
    "classification": "test_accuracy",
    "regression": "test_r2",
}


# ---------------------------------------------------------------------------
# BattleRunner
# ---------------------------------------------------------------------------


class BattleRunner:
    """Train multiple models via cross-validation and collect results.

    Usage::

        runner = BattleRunner()
        result = runner.run(df, target="survived", task="classification")
        result.leaderboard()

    Args:
        config: InsightMLConfig to use. Defaults to the global config.
        registry: ModelRegistry to draw models from. Defaults to the module-level registry.
        eda_result: Optional EDAResult to inform preprocessing choices.
    """

    def __init__(
        self,
        config: InsightMLConfig | None = None,
        registry: ModelRegistry | None = None,
        eda_result: Any = None,
    ) -> None:
        self._config = config or get_config()
        self._registry = registry or get_registry()
        self._eda_result = eda_result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        target: str,
        task: str | None = None,
        *,
        models: list[str] | None = None,
        families: list[str] | None = None,
        exclude: list[str] | None = None,
        cv: int | None = None,
        n_jobs: int | None = None,
        timeout: int | None = None,
    ) -> BattleResult:
        """Run the model battle.

        Args:
            df: Input DataFrame (features + target).
            target: Name of the target column.
            task: "classification" or "regression". Inferred if None.
            models: Explicit list of model names to include.
            families: Filter by model family (e.g. ["tree", "linear"]).
            exclude: Model names to exclude.
            cv: Number of CV folds (overrides config).
            n_jobs: Parallel jobs (overrides config).
            timeout: Per-model timeout in seconds (overrides config).

        Returns:
            :class:`~dissectml.battle.result.BattleResult`
        """
        config = self._config

        # --- Infer task ---
        if task is None:
            task = _infer_task(df[target])

        # --- Prepare X, y ---
        X = df.drop(columns=[target])
        y = df[target]

        # --- Build preprocessing plan ---
        plan = build_preprocessing_plan(
            df=X,
            target=None,
            eda_result=self._eda_result,
        )

        # --- Select models ---
        entries = self._select_models(task, models, families, exclude)
        if not entries:
            raise ValueError(
                f"No models available for task={task!r} with the given filters."
            )

        # --- CV splitter ---
        n_folds = cv or config.cv_folds
        splitter = (
            StratifiedKFold(n_splits=n_folds, shuffle=True,
                            random_state=config.random_state)
            if task == "classification"
            else KFold(n_splits=n_folds, shuffle=True,
                       random_state=config.random_state)
        )

        # --- Primary metric ---
        primary = _PRIMARY_METRIC[task]

        # --- Parallel training ---
        n_parallel = n_jobs if n_jobs is not None else config.n_jobs
        per_model_timeout = timeout or config.timeout_per_model

        scores: list[ModelScore] = Parallel(n_jobs=n_parallel, prefer="threads")(
            delayed(_train_one)(
                entry=entry,
                X=X,
                y=y,
                plan=plan,
                splitter=splitter,
                task=task,
                timeout=per_model_timeout,
                random_state=config.random_state,
            )
            for entry in entries
        )

        # --- Sort by primary metric ---
        metric_key = primary.replace("test_", "")
        scores.sort(
            key=lambda s: s.metrics.get(metric_key, -np.inf),
            reverse=True,
        )

        return BattleResult(
            task=task,
            scores=scores,
            feature_names=list(X.columns),
            target_name=target,
            n_samples=len(df),
            cv_folds=n_folds,
            primary_metric=metric_key,
            config_snapshot={
                "cv_folds": n_folds,
                "n_jobs": n_parallel,
                "timeout": per_model_timeout,
                "random_state": config.random_state,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_models(
        self,
        task: str,
        names: list[str] | None,
        families: list[str] | None,
        exclude: list[str] | None,
    ) -> list[ModelEntry]:
        entries = self._registry.filter(
            task=task,
            families=families,
            names=names,
            exclude=exclude,
        )
        return entries


# ---------------------------------------------------------------------------
# Per-model training (runs in parallel worker)
# ---------------------------------------------------------------------------


def _train_one(
    entry: ModelEntry,
    X: pd.DataFrame,
    y: pd.Series,
    plan: PreprocessingPlan,
    splitter: Any,
    task: str,
    timeout: int,
    random_state: int,
) -> ModelScore:
    """Train one model via cross-validation, collecting OOF predictions."""
    start = time.perf_counter()
    try:
        estimator = entry.build()
        pipeline = build_full_pipeline(
            estimator=estimator,
            plan=plan,
            tree_based=entry.tree_based,
        )

        metrics_map = (
            _CLASSIFICATION_METRICS if task == "classification" else _REGRESSION_METRICS
        )

        # Use cross_validate for metric scores + timings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_out = cross_validate(
                pipeline,
                X, y,
                cv=splitter,
                scoring=metrics_map,
                return_estimator=True,
                return_train_score=False,
                n_jobs=1,
                error_score="raise",
            )

        # Aggregate metrics
        metrics: dict[str, float] = {}
        metrics_std: dict[str, float] = {}
        for key in metrics_map:
            raw = cv_out.get(f"test_{key}", np.array([]))
            # sklearn prefixes negated metrics with "neg_"; convert back
            vals = np.array(raw, dtype=float)
            if key.startswith("neg_"):
                vals = -vals
                display_key = key[4:]  # strip "neg_"
            else:
                display_key = key
            metrics[display_key] = round(float(np.nanmean(vals)), 6)
            metrics_std[display_key] = round(float(np.nanstd(vals)), 6)

        train_time = float(np.sum(cv_out.get("fit_time", [0])))
        predict_time = float(np.sum(cv_out.get("score_time", [0])))

        # OOF predictions
        oof_preds, oof_probs = _collect_oof(
            pipeline, X, y, splitter, task, cv_out["estimator"]
        )

        fitted_pipeline = cv_out["estimator"][-1]  # last fold's fitted pipeline

        return ModelScore(
            name=entry.name,
            task=task,
            metrics=metrics,
            metrics_std=metrics_std,
            train_time=train_time,
            predict_time=predict_time,
            oof_predictions=oof_preds,
            oof_probabilities=oof_probs,
            fitted_pipeline=fitted_pipeline,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return ModelScore(
            name=entry.name,
            task=task,
            train_time=elapsed,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
        )


def _collect_oof(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Any,
    task: str,
    fitted_estimators: list[Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Re-run the CV loop to collect out-of-fold predictions.

    Uses the already-fitted estimators from ``cross_validate`` to avoid
    re-fitting.  Returns (oof_predictions, oof_probabilities).
    """
    n = len(y)
    oof_preds = np.full(n, np.nan)
    oof_probs: np.ndarray | None = None

    try:
        X_arr = X.reset_index(drop=True)
        y_arr = y.reset_index(drop=True)

        for fold_idx, (_, test_idx) in enumerate(splitter.split(X_arr, y_arr)):
            est = fitted_estimators[fold_idx]
            X_test = X_arr.iloc[test_idx]

            preds = est.predict(X_test)
            oof_preds[test_idx] = preds

            if task == "classification":
                try:
                    probs = est.predict_proba(X_test)
                    if oof_probs is None:
                        oof_probs = np.full((n, probs.shape[1]), np.nan)
                    oof_probs[test_idx] = probs
                except AttributeError:
                    pass  # Model has no predict_proba (e.g. RidgeClassifier)

    except Exception:
        pass  # OOF collection is best-effort; failure doesn't break the result

    return oof_preds, oof_probs


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def _infer_task(target: pd.Series) -> str:
    """Infer 'classification' or 'regression' from a target series."""
    if (
        pd.api.types.is_bool_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(target)
        or str(target.dtype) in ("string", "category")
        or target.nunique() <= 20
    ):
        return "classification"
    return "regression"
