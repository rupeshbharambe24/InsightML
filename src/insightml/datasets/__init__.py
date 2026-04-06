"""Built-in demo datasets for InsightML.

Quick start::

    import insightml as iml

    df = iml.load_titanic()   # 891 rows, classification target: 'survived'
    df = iml.load_housing()   # ~20k rows, regression target: 'MedHouseVal'
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Titanic
# ---------------------------------------------------------------------------


def load_titanic() -> pd.DataFrame:
    """Load the Titanic passenger dataset.

    Returns a DataFrame with 891 rows and columns: ``survived`` (int, target),
    ``pclass``, ``sex``, ``age``, ``sibsp``, ``parch``, ``fare``, ``embarked``.

    Loads from a bundled CSV if present; falls back to seaborn's copy; then
    to a small synthetic stand-in for offline/CI environments.

    Returns:
        pandas DataFrame, target column: ``survived``.
    """
    bundled = _DATA_DIR / "titanic.csv"
    if bundled.exists():
        return pd.read_csv(bundled)

    # Seaborn ships a clean Titanic CSV
    try:
        import seaborn as sns  # type: ignore[import-untyped]

        df = sns.load_dataset("titanic")
        keep = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
        available = [c for c in keep if c in df.columns]
        return df[available].copy()
    except Exception:
        pass

    # Last resort: synthetic stand-in (for CI without network access)
    return _synthetic_titanic()


def _synthetic_titanic(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "survived": rng.choice([0, 1], n),
        "pclass": rng.choice([1, 2, 3], n),
        "sex": rng.choice(["male", "female"], n),
        "age": rng.uniform(1, 80, n).round(1),
        "sibsp": rng.integers(0, 5, n),
        "parch": rng.integers(0, 4, n),
        "fare": rng.exponential(30, n).round(2),
        "embarked": rng.choice(["S", "C", "Q"], n),
    })


# ---------------------------------------------------------------------------
# California Housing
# ---------------------------------------------------------------------------


def load_housing() -> pd.DataFrame:
    """Load the California Housing dataset.

    Returns a DataFrame with ~20 640 rows. Target column: ``MedHouseVal``
    (median house value, regression). Features: ``MedInc``, ``HouseAge``,
    ``AveRooms``, ``AveBedrms``, ``Population``, ``AveOccup``,
    ``Latitude``, ``Longitude``.

    Loads from a bundled CSV if present; falls back to sklearn's
    ``fetch_california_housing``; then to a small synthetic stand-in.

    Returns:
        pandas DataFrame, target column: ``MedHouseVal``.
    """
    bundled = _DATA_DIR / "housing.csv"
    if bundled.exists():
        return pd.read_csv(bundled)

    try:
        from sklearn.datasets import fetch_california_housing

        housing = fetch_california_housing(as_frame=True)
        return housing.frame.copy()
    except Exception:
        pass

    return _synthetic_housing()


def _synthetic_housing(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "MedInc": rng.uniform(0.5, 15.0, n).round(4),
        "HouseAge": rng.uniform(1, 52, n).round(1),
        "AveRooms": rng.uniform(1, 10, n).round(4),
        "AveBedrms": rng.uniform(0.5, 3, n).round(4),
        "Population": rng.integers(3, 35_000, n).astype(float),
        "AveOccup": rng.uniform(1, 5, n).round(4),
        "Latitude": rng.uniform(32, 42, n).round(4),
        "Longitude": rng.uniform(-124, -114, n).round(4),
        "MedHouseVal": rng.uniform(0.15, 5.0, n).round(4),
    })


__all__ = ["load_titanic", "load_housing"]
