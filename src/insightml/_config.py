"""Global configuration for InsightML with context manager support."""

from __future__ import annotations

import contextlib
import copy
import threading
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InsightMLConfig:
    """Configuration dataclass for InsightML.

    Three override levels (highest to lowest priority):
        per-call kwargs > config_context() > set_config() > defaults
    """

    # --- EDA ---
    categorical_threshold: int = 50          # nunique <= this -> CATEGORICAL
    high_cardinality_threshold: int = 100    # nunique > this -> HIGH_CARDINALITY
    text_min_avg_length: int = 30            # avg str length > this -> TEXT
    significance_level: float = 0.05        # p-value threshold for all tests
    iqr_multiplier: float = 1.5             # outlier IQR fence multiplier
    zscore_threshold: float = 3.0           # outlier z-score cutoff
    isolation_forest_contamination: float = 0.05
    max_k_clusters: int = 10               # max K for auto K-Means
    max_bivariate_pairs: int = 30          # pair limit for bivariate analysis
    correlation_methods: list[str] = field(
        default_factory=lambda: ["pearson", "spearman", "cramers_v"]
    )

    # --- Battle ---
    cv_folds: int = 5
    timeout_per_model: int = 300            # seconds per model during training
    n_jobs: int = -1                        # joblib parallelism (-1 = all cores)
    random_state: int = 42

    # --- Scale ---
    large_dataset_threshold: int = 100_000  # rows; triggers auto-sampling
    sample_size: int = 50_000              # subsample size for expensive ops

    # --- Report ---
    report_theme: str = "default"
    plotly_template: str = "plotly_white"

    # --- General ---
    verbosity: int = 1                      # 0=silent, 1=progress, 2=debug

    def copy_with(self, **kwargs: Any) -> InsightMLConfig:
        """Return a copy of this config with the given fields overridden."""
        cfg = copy.copy(self)
        for key, value in kwargs.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown config key: {key!r}")
            setattr(cfg, key, value)
        return cfg


# ---------------------------------------------------------------------------
# Global config state (thread-local for context manager support)
# ---------------------------------------------------------------------------

_global_config = InsightMLConfig()
_thread_local = threading.local()


def get_config() -> InsightMLConfig:
    """Return the currently active configuration.

    Returns thread-local config if inside config_context(), else the global config.
    """
    return getattr(_thread_local, "config", _global_config)


def set_config(**kwargs: Any) -> None:
    """Update the global default configuration.

    Example::

        iml.set_config(cv_folds=10, verbosity=0)
    """
    global _global_config
    for key, value in kwargs.items():
        if not hasattr(_global_config, key):
            raise ValueError(f"Unknown config key: {key!r}")
        setattr(_global_config, key, value)


@contextlib.contextmanager
def config_context(**kwargs: Any) -> Generator[InsightMLConfig, None, None]:
    """Temporarily override configuration within a with-block.

    Example::

        with iml.config_context(cv_folds=3):
            result = iml.battle(df, target="y")
    """
    old_config = getattr(_thread_local, "config", None)
    base = old_config if old_config is not None else _global_config
    _thread_local.config = base.copy_with(**kwargs)
    try:
        yield _thread_local.config
    finally:
        if old_config is None:
            del _thread_local.config
        else:
            _thread_local.config = old_config
