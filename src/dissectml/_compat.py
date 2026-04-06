"""Compatibility layer — accept Polars DataFrames, file paths, and dicts in public API.

Usage::

    from dissectml._compat import to_pandas

    df = to_pandas(user_input)  # works for pd.DataFrame, pl.DataFrame, Path, str, dict, list

All public-facing functions that accept data should call ``to_pandas`` first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def to_pandas(data: Any, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Convert various input types to a pandas DataFrame.

    Supported inputs:

    * ``pandas.DataFrame`` — returned unchanged.
    * ``polars.DataFrame`` or ``polars.LazyFrame`` — converted via ``.to_pandas()``.
    * ``str`` / ``pathlib.Path`` — read as CSV (pass extra kwargs to ``pd.read_csv``).
    * ``dict`` — passed to ``pd.DataFrame.from_dict``.
    * ``list[dict]`` — passed to ``pd.DataFrame(data)``.
    * ``numpy.ndarray`` — passed to ``pd.DataFrame(data)``.

    Args:
        data: Input data.
        **read_csv_kwargs: Forwarded to :func:`pandas.read_csv` when *data* is a path.

    Returns:
        pandas DataFrame.

    Raises:
        TypeError: If *data* is an unsupported type.
    """
    if isinstance(data, pd.DataFrame):
        return data

    # --- Polars (optional dep) ---
    try:
        import polars as pl  # noqa: F401 (only imported if available)

        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
        if isinstance(data, pl.LazyFrame):
            return data.collect().to_pandas()
    except ImportError:
        pass  # Polars not installed — skip

    # --- File path ---
    if isinstance(data, (str, Path)):
        path = Path(data)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, **read_csv_kwargs)
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(path, **read_csv_kwargs)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path, **read_csv_kwargs)
        if suffix == ".json":
            return pd.read_json(path, **read_csv_kwargs)
        # Fallback: try CSV
        return pd.read_csv(path, **read_csv_kwargs)

    # --- dict / list / array-like ---
    if isinstance(data, dict):
        return pd.DataFrame(data)

    if isinstance(data, list):
        return pd.DataFrame(data)

    # --- numpy ndarray ---
    try:
        import numpy as np

        if isinstance(data, np.ndarray):
            return pd.DataFrame(data)
    except ImportError:
        pass

    raise TypeError(
        f"Unsupported data type: {type(data).__name__}. "
        "Expected pandas DataFrame, polars DataFrame, file path (str/Path), "
        "dict, or list of dicts."
    )


def is_polars_available() -> bool:
    """Return True if polars is importable."""
    try:
        import polars  # noqa: F401
        return True
    except ImportError:
        return False


def get_pandas_version() -> tuple[int, ...]:
    """Return pandas version as a tuple of ints, e.g. (2, 1, 0)."""
    parts = pd.__version__.split(".")
    result = []
    for p in parts:
        # Strip any non-numeric suffix like "1rc1"
        num = "".join(c for c in p if c.isdigit())
        result.append(int(num) if num else 0)
    return tuple(result)
