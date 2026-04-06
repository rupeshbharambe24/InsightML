"""File I/O helpers — load DataFrames from CSV, Excel, Parquet, or JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dissectml.exceptions import DependencyError, UnsupportedFormatError

# Supported extensions and their pandas readers
_READERS: dict[str, Any] = {
    ".csv": pd.read_csv,
    ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".parquet": pd.read_parquet,
    ".json": pd.read_json,
}

SUPPORTED_EXTENSIONS = list(_READERS.keys())


def read_data(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a DataFrame from a file path.

    Supports CSV, TSV, Excel (.xlsx/.xls), Parquet, and JSON.
    Extra kwargs are passed directly to the underlying pandas reader.

    Args:
        path: Path to the data file.
        **kwargs: Additional keyword arguments forwarded to the pandas reader.

    Returns:
        Loaded DataFrame.

    Raises:
        UnsupportedFormatError: If the file extension is not supported.
        DependencyError: If a required optional reader package is missing.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = path.suffix.lower()
    if ext not in _READERS:
        raise UnsupportedFormatError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: {SUPPORTED_EXTENSIONS}"
        )

    reader = _READERS[ext]
    try:
        return reader(path, **kwargs)
    except ImportError as exc:
        # e.g., openpyxl not installed for .xlsx, pyarrow missing for .parquet
        missing = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
        raise DependencyError(
            f"Reading '{ext}' files requires '{missing}'. "
            f"Install it with: pip install {missing}"
        ) from exc
