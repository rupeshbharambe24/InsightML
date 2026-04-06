"""Shared pytest fixtures for InsightML tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Small synthetic DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def df_classification() -> pd.DataFrame:
    """150-row classification dataset (iris-like structure)."""
    rng = np.random.default_rng(42)
    n = 150
    return pd.DataFrame({
        "sepal_length": rng.normal(5.8, 0.8, n),
        "sepal_width": rng.normal(3.0, 0.4, n),
        "petal_length": np.concatenate([
            rng.normal(1.5, 0.2, 50),
            rng.normal(4.5, 0.5, 50),
            rng.normal(5.5, 0.6, 50),
        ]),
        "petal_width": np.concatenate([
            rng.normal(0.3, 0.1, 50),
            rng.normal(1.3, 0.2, 50),
            rng.normal(2.0, 0.3, 50),
        ]),
        "species": ["setosa"] * 50 + ["versicolor"] * 50 + ["virginica"] * 50,
    })


@pytest.fixture
def df_regression() -> pd.DataFrame:
    """200-row regression dataset."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.uniform(0, 10, n)
    noise = rng.normal(0, 0.5, n)
    return pd.DataFrame({
        "feature_a": x1,
        "feature_b": x2,
        "feature_c": x3,
        "category": rng.choice(["A", "B", "C"], n),
        "target": 2.5 * x1 - 1.2 * x2 + 0.3 * x3 + noise,
    })


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """100-row DataFrame with intentional missing values."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "num1": rng.normal(0, 1, n),
        "num2": rng.normal(5, 2, n),
        "cat1": rng.choice(["A", "B", "C"], n),
        "target": rng.normal(0, 1, n),
    })
    # Introduce missing values
    df.loc[rng.choice(n, 15, replace=False), "num1"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "num2"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "cat1"] = np.nan
    return df


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """100-row DataFrame with clear outliers."""
    rng = np.random.default_rng(42)
    n = 100
    col = rng.normal(0, 1, n)
    col[0] = 100.0   # extreme high
    col[1] = -100.0  # extreme low
    return pd.DataFrame({
        "normal_col": rng.normal(0, 1, n),
        "outlier_col": col,
        "target": rng.integers(0, 2, n),
    })


@pytest.fixture
def df_tiny() -> pd.DataFrame:
    """Minimal 10-row DataFrame for edge case tests."""
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [2.0, 4.1, 6.0, 8.2, 10.0, 11.9, 14.1, 16.0, 18.1, 20.0],
        "cat": ["A", "B"] * 5,
    })


@pytest.fixture
def df_all_types() -> pd.DataFrame:
    """DataFrame with one column of every detected ColumnType."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "numeric": rng.normal(0, 1, n),
        "categorical": rng.choice(["cat", "dog", "bird"], n),
        "boolean": rng.choice([True, False], n),
        "datetime": pd.date_range("2023-01-01", periods=n, freq="D"),
        "high_card": [f"item_{i}" for i in range(n)],  # n unique -> UNIQUE_ID
        "constant": ["same"] * n,
        "target": rng.integers(0, 2, n),
    })


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    """Return a fresh default InsightMLConfig."""
    from dissectml._config import InsightMLConfig
    return InsightMLConfig()
