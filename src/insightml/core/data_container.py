"""DataContainer — the single data object flowing through the pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from insightml._config import InsightMLConfig, get_config
from insightml._io import read_data
from insightml._sampling import smart_sample
from insightml._types import DataSchema, TaskType
from insightml.exceptions import EmptyDataFrameError, TargetNotFoundError


class DataContainer:
    """Immutable wrapper around a DataFrame with metadata for the pipeline.

    Attributes:
        df: Full DataFrame (always pandas, always materialized).
        target: Target column name (None for unsupervised).
        task: TaskType (classification / regression / auto-detected).
        schema: Inferred DataSchema (column types, roles, cardinalities).
        sample: Subsampled DataFrame for expensive computations on large data.
                Equals `df` when the dataset is small enough.
        _original_path: Original file path if loaded from disk (for provenance).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str | None,
        task: TaskType,
        schema: DataSchema,
        sample: pd.DataFrame,
        original_path: str | None = None,
    ) -> None:
        self.df = df
        self.target = target
        self.task = task
        self.schema = schema
        self.sample = sample
        self._original_path = original_path

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_input(
        cls,
        data: str | Path | pd.DataFrame,
        target: str | None = None,
        task: str | TaskType = TaskType.AUTO,
        config: InsightMLConfig | None = None,
    ) -> DataContainer:
        """Create a DataContainer from multiple input types.

        Args:
            data: File path (str/Path), pandas DataFrame, Polars DataFrame,
                  dict, list of dicts, or numpy array.
            target: Name of the target column.
            task: 'classification', 'regression', or 'auto' (default).
            config: Configuration override. Uses global config if None.

        Returns:
            A fully initialised DataContainer.

        Raises:
            EmptyDataFrameError: If the DataFrame has no rows or columns.
            TargetNotFoundError: If `target` is specified but not in the data.
        """
        if config is None:
            config = get_config()

        original_path: str | None = None

        # --- Load data ---
        if isinstance(data, (str, Path)):
            original_path = str(data)
            df = read_data(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            # Delegate to to_pandas() — handles Polars DataFrames, dicts, lists, numpy arrays
            from insightml._compat import to_pandas
            df = to_pandas(data)

        # --- Validate ---
        if df.empty:
            raise EmptyDataFrameError(
                "Input DataFrame is empty (0 rows or 0 columns)."
            )
        if target is not None and target not in df.columns:
            raise TargetNotFoundError(
                f"Target column '{target}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        # --- Task detection ---
        if isinstance(task, str):
            task = TaskType(task)
        if task == TaskType.AUTO and target is not None:
            from insightml.core.validators import infer_task
            task = infer_task(df[target])

        # --- Schema inference ---
        from insightml.core.validators import infer_schema
        schema = infer_schema(df, target=target, task=task, config=config)

        # --- Sampling ---
        sample = smart_sample(df, target=target, config=config)

        return cls(
            df=df,
            target=target,
            task=task,
            schema=schema,
            sample=sample,
            original_path=original_path,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_cols(self) -> int:
        return len(self.df.columns)

    @property
    def feature_cols(self) -> list[str]:
        """All columns except the target."""
        if self.target is None:
            return list(self.df.columns)
        return [c for c in self.df.columns if c != self.target]

    @property
    def X(self) -> pd.DataFrame:
        """Feature matrix (all columns except target)."""
        return self.df[self.feature_cols]

    @property
    def y(self) -> pd.Series | None:
        """Target series, or None if no target was specified."""
        if self.target is None:
            return None
        return self.df[self.target]

    @property
    def is_large(self) -> bool:
        """True if the dataset exceeds the large-dataset threshold."""
        from insightml._config import get_config
        return self.n_rows > get_config().large_dataset_threshold

    def __repr__(self) -> str:
        task_str = self.task.value if isinstance(self.task, TaskType) else self.task
        return (
            f"DataContainer(rows={self.n_rows}, cols={self.n_cols}, "
            f"target={self.target!r}, task={task_str!r})"
        )
