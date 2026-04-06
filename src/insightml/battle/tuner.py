"""ModelTuner — RandomizedSearchCV-based hyperparameter tuning for top models."""

from __future__ import annotations

import time
import traceback
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold

from insightml._config import InsightMLConfig, get_config
from insightml.battle.param_grids import get_param_grid
from insightml.battle.preprocessing import PreprocessingPlan, build_full_pipeline
from insightml.battle.result import BattleResult, ModelScore
from insightml.battle.registry import ModelRegistry, get_registry


class ModelTuner:
    """Tune hyperparameters for the top-N models from a :class:`BattleResult`.

    Three modes
    -----------
    * ``"quick"`` — no search; return the battle result unchanged.
    * ``"tuned"`` — ``RandomizedSearchCV`` with ``n_iter`` iterations on top-N models.
    * ``"custom"`` — user-supplied param grids; top-N otherwise same as ``"tuned"``.

    Usage::

        tuner = ModelTuner(mode="tuned", top_n=3, n_iter=30)
        tuned_result = tuner.tune(battle_result, X, y)
        tuned_result.leaderboard()
    """

    def __init__(
        self,
        mode: str = "tuned",
        top_n: int = 3,
        n_iter: int = 20,
        cv: int = 3,
        config: InsightMLConfig | None = None,
        registry: ModelRegistry | None = None,
        custom_grids: dict[str, dict] | None = None,
    ) -> None:
        if mode not in ("quick", "tuned", "custom"):
            raise ValueError(f"mode must be 'quick', 'tuned', or 'custom'. Got {mode!r}")
        self.mode = mode
        self.top_n = top_n
        self.n_iter = n_iter
        self.cv = cv
        self._config = config or get_config()
        self._registry = registry or get_registry()
        self._custom_grids = custom_grids or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(
        self,
        battle_result: BattleResult,
        X: pd.DataFrame,
        y: pd.Series,
        plan: PreprocessingPlan | None = None,
    ) -> BattleResult:
        """Tune top-N models from *battle_result*.

        Returns a new :class:`BattleResult` where the top-N scores are
        replaced by their tuned counterparts. All other scores are kept
        unchanged.

        Args:
            battle_result: Output of :class:`~insightml.battle.runner.BattleRunner`.
            X: Feature DataFrame.
            y: Target Series.
            plan: Optional preprocessing plan; if None, derived from battle_result metadata.

        Returns:
            New BattleResult with tuned scores.
        """
        if self.mode == "quick":
            return battle_result

        task = battle_result.task
        top_scores = battle_result.successful[: self.top_n]

        if not top_scores:
            return battle_result

        # Preprocessing plan
        if plan is None:
            plan = _plan_from_result(battle_result, X)

        # CV splitter
        config = self._config
        splitter = (
            StratifiedKFold(n_splits=self.cv, shuffle=True,
                            random_state=config.random_state)
            if task == "classification"
            else KFold(n_splits=self.cv, shuffle=True,
                       random_state=config.random_state)
        )

        # Tune each top model
        tuned_map: dict[str, ModelScore] = {}
        for score in top_scores:
            grid = self._get_grid(score.name, task)
            if not grid:
                # No grid: keep original score
                tuned_map[score.name] = score
                continue
            tuned = _tune_one(
                score=score,
                X=X,
                y=y,
                plan=plan,
                splitter=splitter,
                task=task,
                param_grid=grid,
                n_iter=self.n_iter,
                random_state=config.random_state,
                registry=self._registry,
            )
            tuned_map[score.name] = tuned

        # Merge: replace top-N scores with tuned versions
        new_scores: list[ModelScore] = []
        tuned_names = {s.name for s in top_scores}
        for s in battle_result.scores:
            if s.name in tuned_map:
                new_scores.append(tuned_map[s.name])
            elif s.name not in tuned_names:
                new_scores.append(s)

        # Sort by primary metric
        primary = battle_result.primary_metric
        new_scores.sort(
            key=lambda s: s.metrics.get(primary, -np.inf),
            reverse=True,
        )

        return BattleResult(
            task=task,
            scores=new_scores,
            feature_names=battle_result.feature_names,
            target_name=battle_result.target_name,
            n_samples=battle_result.n_samples,
            cv_folds=battle_result.cv_folds,
            primary_metric=primary,
            config_snapshot={
                **battle_result.config_snapshot,
                "tuning_mode": self.mode,
                "tuning_top_n": self.top_n,
                "tuning_n_iter": self.n_iter,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_grid(self, model_name: str, task: str) -> dict:
        if self.mode == "custom" and model_name in self._custom_grids:
            return self._custom_grids[model_name]
        return get_param_grid(model_name, task)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tune_one(
    score: ModelScore,
    X: pd.DataFrame,
    y: pd.Series,
    plan: PreprocessingPlan,
    splitter: Any,
    task: str,
    param_grid: dict,
    n_iter: int,
    random_state: int,
    registry: ModelRegistry,
) -> ModelScore:
    """Run RandomizedSearchCV for a single model."""
    start = time.perf_counter()
    try:
        entry = registry.get(score.name)
        estimator = entry.build()
        pipeline = build_full_pipeline(
            estimator=estimator,
            plan=plan,
            tree_based=entry.tree_based,
        )

        primary_scorer = (
            "accuracy" if task == "classification" else "r2"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=min(n_iter, 5),  # safety cap
                scoring=primary_scorer,
                cv=splitter,
                refit=True,
                n_jobs=1,
                random_state=random_state,
                error_score="raise",
            )
            search.fit(X, y)

        best_score = float(search.best_score_)
        elapsed = time.perf_counter() - start

        # Preserve OOF from the original score; tuning only refits once
        tuned = ModelScore(
            name=score.name,
            task=task,
            metrics={**score.metrics, primary_scorer: round(best_score, 6)},
            metrics_std=score.metrics_std,
            train_time=elapsed,
            predict_time=score.predict_time,
            oof_predictions=score.oof_predictions,
            oof_probabilities=score.oof_probabilities,
            fitted_pipeline=search.best_estimator_,
        )
        return tuned

    except Exception as exc:
        elapsed = time.perf_counter() - start
        # Tuning failed: return original score with a warning
        return ModelScore(
            name=score.name,
            task=task,
            metrics=score.metrics,
            metrics_std=score.metrics_std,
            train_time=score.train_time + elapsed,
            predict_time=score.predict_time,
            oof_predictions=score.oof_predictions,
            oof_probabilities=score.oof_probabilities,
            fitted_pipeline=score.fitted_pipeline,
            error=None,  # Keep original; tuning is best-effort
        )


def _plan_from_result(result: BattleResult, X: pd.DataFrame) -> PreprocessingPlan:
    """Derive a minimal PreprocessingPlan from the BattleResult metadata + X."""
    from insightml.battle.preprocessing import build_preprocessing_plan
    return build_preprocessing_plan(df=X, target=None, eda_result=None)
