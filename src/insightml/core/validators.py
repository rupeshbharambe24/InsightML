"""Input validation, task inference, and schema detection."""

from __future__ import annotations

import warnings

import pandas as pd

from insightml._config import InsightMLConfig
from insightml._types import ColumnType, DataSchema, TaskType
from insightml.exceptions import EmptyDataFrameError, TargetNotFoundError


def infer_task(target: pd.Series) -> TaskType:
    """Auto-detect whether a target column is for classification or regression.

    Rules (in order):
    1. dtype is object/category/bool -> classification
    2. dtype is int AND nunique <= max(20, 5% of len) -> classification (emit warning if borderline)
    3. dtype is float AND nunique <= 20 AND all whole numbers -> classification
    4. Otherwise -> regression

    Args:
        target: The target column Series.

    Returns:
        TaskType.CLASSIFICATION or TaskType.REGRESSION.
    """
    n = len(target)
    nu = target.nunique()

    # Handle bool, object, string (pandas 3.x uses StringDtype), and category
    if pd.api.types.is_bool_dtype(target):
        return TaskType.CLASSIFICATION

    if (pd.api.types.is_object_dtype(target)
            or pd.api.types.is_string_dtype(target)
            or str(target.dtype) in ("category", "string", "object")):
        return TaskType.CLASSIFICATION

    if pd.api.types.is_integer_dtype(target):
        threshold = max(20, int(0.05 * n))
        if nu <= threshold:
            if 15 <= nu <= 25:
                warnings.warn(
                    f"Target column has {nu} unique integer values — auto-detected as "
                    "CLASSIFICATION. Pass task='regression' explicitly if this is wrong.",
                    UserWarning,
                    stacklevel=4,
                )
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION

    if pd.api.types.is_float_dtype(target):
        non_null = target.dropna()
        if nu <= 20 and (non_null == non_null.round()).all():
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION

    return TaskType.REGRESSION


def infer_column_type(
    col: pd.Series,
    config: InsightMLConfig,
) -> ColumnType:
    """Infer the semantic type of a single column.

    Args:
        col: Column Series.
        config: InsightML configuration (thresholds).

    Returns:
        ColumnType enum value.
    """
    n = len(col)
    nu = col.nunique(dropna=True)

    # --- Boolean ---
    if col.dtype == bool or pd.api.types.is_bool_dtype(col):
        return ColumnType.BOOLEAN
    unique_vals = set(col.dropna().unique())
    if unique_vals <= {True, False, 0, 1, "True", "False", "true", "false"}:
        if nu <= 2:
            return ColumnType.BOOLEAN

    # --- Datetime ---
    if pd.api.types.is_datetime64_any_dtype(col):
        return ColumnType.DATETIME

    # --- Numeric ---
    if pd.api.types.is_numeric_dtype(col):
        if nu == 1:
            return ColumnType.CONSTANT
        if nu == n and pd.api.types.is_integer_dtype(col):
            return ColumnType.UNIQUE_ID
        return ColumnType.NUMERIC

    # --- Object / string / category ---
    if nu == 1:
        return ColumnType.CONSTANT
    if nu == n:
        return ColumnType.UNIQUE_ID

    # Try parsing as datetime (suppress format-inference warning)
    try:
        import warnings
        sample = col.dropna().head(100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.8:
            return ColumnType.DATETIME
    except Exception:
        pass

    if nu <= config.categorical_threshold:
        return ColumnType.CATEGORICAL
    if nu > config.high_cardinality_threshold:
        return ColumnType.HIGH_CARDINALITY

    # Check average string length for text detection
    try:
        avg_len = col.dropna().astype(str).str.len().mean()
        if avg_len > config.text_min_avg_length:
            return ColumnType.TEXT
    except Exception:
        pass

    return ColumnType.CATEGORICAL


def infer_schema(
    df: pd.DataFrame,
    target: str | None,
    task: TaskType,
    config: InsightMLConfig,
) -> DataSchema:
    """Build a DataSchema by inferring the type of every column.

    Args:
        df: Input DataFrame.
        target: Target column name (may be None).
        task: Resolved task type.
        config: InsightML configuration.

    Returns:
        DataSchema TypedDict.
    """
    column_types: dict[str, ColumnType] = {}
    for col_name in df.columns:
        column_types[col_name] = infer_column_type(df[col_name], config)

    schema: DataSchema = {
        "column_types": column_types,
        "numeric_cols": [c for c, t in column_types.items() if t == ColumnType.NUMERIC],
        "categorical_cols": [c for c, t in column_types.items() if t == ColumnType.CATEGORICAL],
        "datetime_cols": [c for c, t in column_types.items() if t == ColumnType.DATETIME],
        "text_cols": [c for c, t in column_types.items() if t == ColumnType.TEXT],
        "boolean_cols": [c for c, t in column_types.items() if t == ColumnType.BOOLEAN],
        "high_cardinality_cols": [c for c, t in column_types.items() if t == ColumnType.HIGH_CARDINALITY],
        "constant_cols": [c for c, t in column_types.items() if t == ColumnType.CONSTANT],
        "unique_id_cols": [c for c, t in column_types.items() if t == ColumnType.UNIQUE_ID],
        "target_col": target,
        "task": task,
    }
    return schema


def validate_dataframe(df: pd.DataFrame, target: str | None = None) -> None:
    """Raise if the DataFrame is invalid for InsightML.

    Args:
        df: DataFrame to validate.
        target: Optional target column name to check existence.

    Raises:
        EmptyDataFrameError: If df has no rows or columns.
        TargetNotFoundError: If target is not in df.columns.
    """
    if df.empty or len(df.columns) == 0:
        raise EmptyDataFrameError("Input DataFrame is empty.")
    if len(df) == 0:
        raise EmptyDataFrameError("Input DataFrame has 0 rows.")
    if target is not None and target not in df.columns:
        raise TargetNotFoundError(
            f"Target column '{target}' not found. Available: {list(df.columns)}"
        )
