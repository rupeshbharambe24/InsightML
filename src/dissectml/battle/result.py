"""BattleResult and ModelScore — structured output of the model battle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelScore:
    """Metrics and metadata for a single trained model.

    Attributes:
        name: Model name (e.g. "RandomForestClassifier").
        task: "classification" or "regression".
        metrics: Dict of metric_name -> mean CV score.
        metrics_std: Dict of metric_name -> std across CV folds.
        train_time: Total fit time across all CV folds (seconds).
        predict_time: Total score time across all CV folds (seconds).
        oof_predictions: Out-of-fold predictions (1-D array, length = n_samples).
        oof_probabilities: OOF class probabilities (2-D array, classification only).
        fitted_pipeline: The last fitted sklearn Pipeline (fold N).
        error: If training failed, the error message. None on success.
    """

    name: str
    task: str
    metrics: dict[str, float] = field(default_factory=dict)
    metrics_std: dict[str, float] = field(default_factory=dict)
    train_time: float = 0.0
    predict_time: float = 0.0
    oof_predictions: np.ndarray | None = None
    oof_probabilities: np.ndarray | None = None
    fitted_pipeline: Any = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        """True if training raised an error."""
        return self.error is not None

    @property
    def primary_metric(self) -> float | None:
        """Return the primary metric value (first entry in metrics dict)."""
        if not self.metrics:
            return None
        return next(iter(self.metrics.values()))

    def to_dict(self) -> dict[str, Any]:
        """Flat dict suitable for a DataFrame row."""
        row: dict[str, Any] = {"model": self.name, "task": self.task}
        row.update(self.metrics)
        row["train_time_s"] = round(self.train_time, 3)
        row["predict_time_s"] = round(self.predict_time, 3)
        if self.error:
            row["error"] = self.error
        return row

    def __repr__(self) -> str:
        if self.failed:
            return f"ModelScore({self.name!r}, FAILED: {self.error})"
        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in list(self.metrics.items())[:3]
        )
        return f"ModelScore({self.name!r}, {metrics_str})"


@dataclass
class BattleResult:
    """Aggregated results from a model battle run.

    Produced by :class:`~dissectml.battle.runner.BattleRunner`. Holds all
    :class:`ModelScore` objects plus the dataset metadata used during training.

    Attributes:
        task: "classification" or "regression".
        scores: List of ModelScore, one per model attempted.
        feature_names: Column names used as features.
        target_name: Target column name.
        n_samples: Number of training samples.
        cv_folds: Number of CV folds used.
        primary_metric: Metric name used for ranking.
        config_snapshot: Copy of InsightMLConfig values at battle time.
    """

    task: str
    scores: list[ModelScore] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    target_name: str = ""
    n_samples: int = 0
    cv_folds: int = 5
    primary_metric: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def successful(self) -> list[ModelScore]:
        """Scores for models that trained without error, sorted by primary metric desc."""
        ok = [s for s in self.scores if not s.failed]
        if self.primary_metric and ok and self.primary_metric in ok[0].metrics:
            ok.sort(key=lambda s: s.metrics.get(self.primary_metric, -np.inf), reverse=True)
        return ok

    @property
    def failed(self) -> list[ModelScore]:
        """Scores for models that raised an error during training."""
        return [s for s in self.scores if s.failed]

    @property
    def best(self) -> ModelScore | None:
        """Highest-scoring successful model."""
        ok = self.successful
        return ok[0] if ok else None

    def get(self, name: str) -> ModelScore:
        """Return ModelScore by model name.

        Raises:
            KeyError: If name not found.
        """
        for s in self.scores:
            if s.name == name:
                return s
        available = [s.name for s in self.scores]
        raise KeyError(f"Model '{name}' not in BattleResult. Available: {available}")

    # ------------------------------------------------------------------
    # Tabular views
    # ------------------------------------------------------------------

    def leaderboard(self, include_failed: bool = False) -> pd.DataFrame:
        """Return a DataFrame of model scores sorted by primary metric.

        Args:
            include_failed: If True, include rows for failed models.

        Returns:
            DataFrame with columns: model, <metrics>, train_time_s, predict_time_s.
        """
        source = self.scores if include_failed else self.successful
        if not source:
            return pd.DataFrame()
        rows = [s.to_dict() for s in source]
        df = pd.DataFrame(rows)
        if self.primary_metric and self.primary_metric in df.columns:
            df = df.sort_values(self.primary_metric, ascending=False).reset_index(drop=True)
        return df

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary (no numpy arrays / fitted models)."""
        return {
            "task": self.task,
            "target": self.target_name,
            "n_samples": self.n_samples,
            "cv_folds": self.cv_folds,
            "primary_metric": self.primary_metric,
            "n_models_attempted": len(self.scores),
            "n_models_ok": len(self.successful),
            "n_models_failed": len(self.failed),
            "best_model": self.best.name if self.best else None,
            "best_score": self.best.primary_metric if self.best else None,
            "leaderboard": [s.to_dict() for s in self.successful],
        }

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _repr_html_(self) -> str:
        lb = self.leaderboard()
        if lb.empty:
            return "<p><b>BattleResult</b>: no successful models.</p>"
        # Highlight best row
        title = (
            f"<h3>BattleResult — {self.task.title()} "
            f"({len(self.successful)}/{len(self.scores)} models trained)</h3>"
        )
        table = lb.to_html(index=False, float_format="{:.4f}".format, border=0)
        failed_note = ""
        if self.failed:
            names = ", ".join(s.name for s in self.failed)
            failed_note = f"<p style='color:#e45756'><b>Failed:</b> {names}</p>"
        return title + table + failed_note

    def __repr__(self) -> str:
        best = self.best
        best_str = (
            f"{best.name} ({self.primary_metric}={best.primary_metric:.4f})"
            if best else "none"
        )
        return (
            f"BattleResult(task={self.task!r}, "
            f"models={len(self.successful)}/{len(self.scores)}, "
            f"best={best_str})"
        )
