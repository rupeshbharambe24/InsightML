"""Tests for TargetAnalysis."""
import pytest
import pandas as pd
import numpy as np
from insightml.eda.target_analysis import TargetAnalysis


def test_classification_balance(df_classification):
    ta = TargetAnalysis(df_classification, target="species")
    balance = ta.balance()
    assert "n_classes" in balance
    assert balance["n_classes"] == 3
    assert "imbalance_severity" in balance
    assert balance["imbalance_severity"] == "balanced"


def test_classification_severe_imbalance():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "x": rng.normal(0, 1, 100),
        "label": ["A"] * 95 + ["B"] * 5,
    })
    ta = TargetAnalysis(df, target="label")
    balance = ta.balance()
    assert balance["imbalance_severity"] == "severe"
    assert balance["recommendation"] is not None


def test_regression_distribution(df_regression):
    ta = TargetAnalysis(df_regression, target="target")
    dist = ta.distribution()
    assert "mean" in dist
    assert "skewness" in dist
    assert "is_normal" in dist


def test_feature_target_plots_built(df_classification):
    ta = TargetAnalysis(df_classification, target="species")
    plots = ta.feature_target_plots()
    assert isinstance(plots, dict)


def test_no_target_raises():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    ta = TargetAnalysis(df)  # no target
    with pytest.raises(ValueError, match="target"):
        ta._ensure_computed()
