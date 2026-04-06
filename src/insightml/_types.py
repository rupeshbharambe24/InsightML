"""Enums, TypedDicts, and type aliases used across InsightML."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    AUTO = "auto"


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    HIGH_CARDINALITY = "high_cardinality"
    CONSTANT = "constant"
    UNIQUE_ID = "unique_id"


class MissingnessType(str, Enum):
    MCAR = "MCAR"    # Missing Completely At Random
    MAR = "MAR"      # Missing At Random (depends on observed data)
    MNAR = "MNAR"    # Missing Not At Random (depends on missing value itself)
    UNKNOWN = "unknown"


class TuningMode(str, Enum):
    QUICK = "quick"    # Default hyperparameters only (no search)
    TUNED = "tuned"    # RandomizedSearchCV on top-N models
    CUSTOM = "custom"  # User-provided param grids


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

class ColumnProfile(TypedDict, total=False):
    """Per-column statistics computed by DataOverview."""
    name: str
    dtype: str
    inferred_type: str          # ColumnType value
    count: int
    unique: int
    missing_count: int
    missing_pct: float
    memory_bytes: int
    # Numeric fields
    mean: float
    median: float
    std: float
    variance: float
    min: float
    max: float
    range: float
    iqr: float
    q1: float
    q3: float
    skewness: float
    kurtosis: float
    # Categorical fields
    top_value: Any
    top_freq: int
    cardinality_ratio: float
    value_counts: dict[str, int]
    # DateTime fields
    dt_min: str
    dt_max: str
    range_days: float
    inferred_frequency: str | None


class DataSchema(TypedDict):
    """Schema inferred from the dataset."""
    column_types: dict[str, ColumnType]
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    text_cols: list[str]
    boolean_cols: list[str]
    high_cardinality_cols: list[str]
    constant_cols: list[str]
    unique_id_cols: list[str]
    target_col: str | None
    task: TaskType


class LeakageWarning(TypedDict):
    """A potential data leakage warning for a feature."""
    column: str
    score: float
    method: str          # "high_correlation" | "mutual_information" | "temporal" | "derived"
    severity: str        # "critical" | "warning" | "info"
    explanation: str


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DataFrame = pd.DataFrame
Series = pd.Series
Array = np.ndarray
