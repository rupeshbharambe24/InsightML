"""Tests for DataOverview."""
import pandas as pd

from insightml._types import ColumnType
from insightml.eda.overview import DataOverview


def test_basic_shape(df_classification):
    ov = DataOverview(df_classification)
    ov._ensure_computed()
    assert ov._results["n_rows"] == 150
    assert ov._results["n_cols"] == 5


def test_column_types_detected(df_all_types):
    ov = DataOverview(df_all_types)
    ct = ov.column_types
    assert ct["numeric"] == ColumnType.NUMERIC
    assert ct["categorical"] == ColumnType.CATEGORICAL
    assert ct["boolean"] == ColumnType.BOOLEAN
    assert ct["constant"] == ColumnType.CONSTANT


def test_numeric_profile_keys(df_regression):
    ov = DataOverview(df_regression)
    profile = next(p for p in ov.profiles if p["name"] == "feature_a")
    for key in ("mean", "median", "std", "q1", "q3", "iqr", "skewness", "kurtosis"):
        assert key in profile


def test_missing_pct(df_with_missing):
    ov = DataOverview(df_with_missing)
    df_out = ov.to_dataframe()
    missing_row = df_out[df_out["name"] == "num1"].iloc[0]
    assert missing_row["missing_count"] == 15
    assert missing_row["missing_pct"] == 15.0


def test_figures_built(df_classification):
    ov = DataOverview(df_classification)
    figs = ov.plot()
    assert "column_types" in figs


def test_summary_string(df_regression):
    ov = DataOverview(df_regression)
    s = ov.summary()
    assert "rows" in s
    assert "columns" in s


def test_duplicate_detection():
    df = pd.DataFrame({"a": [1, 2, 1, 3], "b": ["x", "y", "x", "z"]})
    ov = DataOverview(df)
    ov._ensure_computed()
    assert ov._results["n_duplicates"] == 1
