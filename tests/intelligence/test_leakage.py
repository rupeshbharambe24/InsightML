"""Tests for intelligence/leakage.py."""

import numpy as np
import pandas as pd
import pytest

from dissectml.intelligence.leakage import detect_leakage


@pytest.fixture
def df_no_leakage():
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    target = rng.choice([0, 1], n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


@pytest.fixture
def df_with_leaky_feature():
    rng = np.random.default_rng(1)
    n = 300
    target = rng.normal(0, 1, n)
    leaky = target * 0.9999 + rng.normal(0, 0.001, n)  # near-perfect copy
    noise = rng.normal(0, 1, n)
    return pd.DataFrame({"leaky": leaky, "noise": noise, "target": target})


class TestDetectLeakage:
    def test_no_warnings_on_clean_data(self, df_no_leakage):
        warnings = detect_leakage(df_no_leakage, target="target")
        assert isinstance(warnings, list)
        # Clean random data should have no high-correlation leakage
        high_corr = [w for w in warnings if w["method"] == "high_correlation"]
        assert len(high_corr) == 0

    def test_detects_high_correlation(self, df_with_leaky_feature):
        warnings = detect_leakage(df_with_leaky_feature, target="target")
        flagged = {w["column"] for w in warnings}
        assert "leaky" in flagged

    def test_noise_not_flagged(self, df_with_leaky_feature):
        warnings = detect_leakage(df_with_leaky_feature, target="target")
        flagged = {w["column"] for w in warnings}
        assert "noise" not in flagged

    def test_warning_fields(self, df_with_leaky_feature):
        warnings = detect_leakage(df_with_leaky_feature, target="target")
        assert len(warnings) > 0
        w = warnings[0]
        assert "column" in w
        assert "score" in w
        assert "method" in w
        assert "severity" in w
        assert "explanation" in w

    def test_severity_levels(self, df_with_leaky_feature):
        warnings = detect_leakage(df_with_leaky_feature, target="target")
        for w in warnings:
            assert w["severity"] in ("low", "moderate", "high", "critical")

    def test_missing_target_raises(self, df_no_leakage):
        with pytest.raises(KeyError):
            detect_leakage(df_no_leakage, target="nonexistent")

    def test_sorted_by_score_desc(self, df_with_leaky_feature):
        warnings = detect_leakage(df_with_leaky_feature, target="target")
        scores = [w["score"] for w in warnings]
        assert scores == sorted(scores, reverse=True)

    def test_derived_feature_detected(self):
        rng = np.random.default_rng(42)
        n = 200
        target = rng.normal(10, 2, n)
        derived = target * 2.0 + 1.0  # linear transform = 100% R²
        noise = rng.normal(0, 1, n)
        df = pd.DataFrame({"derived": derived, "noise": noise, "target": target})
        warnings = detect_leakage(df, target="target")
        flagged = {w["column"] for w in warnings}
        assert "derived" in flagged
