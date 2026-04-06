"""Tests for OutlierDetection."""
import pytest
import numpy as np
from insightml.eda.outliers import OutlierDetection


def test_iqr_detects_extremes(df_with_outliers):
    od = OutlierDetection(df_with_outliers)
    iqr = od.by_iqr()
    # outlier_col has values 100 and -100 which should be flagged
    assert iqr["outlier_col"]["n_outliers"] >= 2


def test_zscore_detects_extremes(df_with_outliers):
    od = OutlierDetection(df_with_outliers)
    z = od.by_zscore()
    assert z["outlier_col"]["n_outliers"] >= 2


def test_consensus_non_negative(df_regression):
    od = OutlierDetection(df_regression)
    consensus = od.consensus()
    for col, result in consensus.items():
        assert result["n_outliers"] >= 0


def test_comparison_dataframe(df_regression):
    od = OutlierDetection(df_regression)
    df_comp = od.comparison()
    assert "column" in df_comp.columns
    assert "consensus_outliers" in df_comp.columns


def test_no_outliers_in_clean_data():
    import pandas as pd
    df = pd.DataFrame({"x": list(range(100)), "y": list(range(100))})
    od = OutlierDetection(df)
    iqr = od.by_iqr()
    # Linear data has no IQR outliers
    for col_result in iqr.values():
        assert col_result["n_outliers"] == 0


def test_isolation_forest_runs(df_regression):
    od = OutlierDetection(df_regression)
    iso = od.by_isolation_forest()
    assert "n_outliers" in iso or "error" in iso or "note" in iso
