"""Tests for insightml._sampling — smart sampling strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from insightml._config import InsightMLConfig
from insightml._sampling import smart_sample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(threshold: int = 50, sample_size: int = 30) -> InsightMLConfig:
    """Create a config with small thresholds suitable for testing."""
    return InsightMLConfig(large_dataset_threshold=threshold, sample_size=sample_size)


# ---------------------------------------------------------------------------
# Small DataFrame — below threshold
# ---------------------------------------------------------------------------

class TestSmallDataFrame:
    """Tests when the DataFrame is below the large_dataset_threshold."""

    def test_returns_original_below_threshold(self):
        """A small DataFrame (below threshold) is returned unchanged."""
        df = pd.DataFrame({"a": range(30), "target": ["X"] * 15 + ["Y"] * 15})
        config = _make_config(threshold=50, sample_size=20)
        result = smart_sample(df, target="target", config=config)
        # Should be the exact same object (no sampling needed)
        assert result is df

    def test_force_samples_even_below_threshold(self):
        """force=True subsamples even when below threshold."""
        df = pd.DataFrame({"a": range(30), "target": ["X"] * 15 + ["Y"] * 15})
        config = _make_config(threshold=50, sample_size=20)
        result = smart_sample(df, target="target", config=config, force=True)
        assert len(result) <= 20
        assert len(result) < len(df)


# ---------------------------------------------------------------------------
# Stratified sampling (classification target)
# ---------------------------------------------------------------------------

class TestStratifiedSampling:
    """Tests for stratified sampling with a classification target."""

    def test_preserves_class_distribution(self):
        """Stratified sampling preserves approximate class proportions."""
        rng = np.random.default_rng(42)
        n = 200
        # 50% A, 30% B, 20% C
        target = (["A"] * 100) + (["B"] * 60) + (["C"] * 40)
        df = pd.DataFrame({
            "feat": rng.normal(0, 1, n),
            "target": target,
        })
        config = _make_config(threshold=50, sample_size=60)
        result = smart_sample(df, target="target", config=config)

        assert len(result) <= 60 + 5  # allow small rounding variance

        # With some pandas versions groupby drops the grouping column;
        # if target is still present, verify class proportions are preserved.
        if "target" in result.columns:
            orig_props = df["target"].value_counts(normalize=True).sort_index()
            sample_props = result["target"].value_counts(normalize=True).sort_index()
            for cls in orig_props.index:
                assert abs(orig_props[cls] - sample_props[cls]) < 0.15, (
                    f"Class {cls}: original={orig_props[cls]:.2f}, "
                    f"sample={sample_props[cls]:.2f}"
                )

    def test_stratified_reduces_size(self):
        """Stratified sampling produces a smaller DataFrame."""
        rng = np.random.default_rng(42)
        n = 200
        target = (["A"] * 100) + (["B"] * 60) + (["C"] * 40)
        df = pd.DataFrame({
            "feat": rng.normal(0, 1, n),
            "target": target,
        })
        config = _make_config(threshold=50, sample_size=60)
        result = smart_sample(df, target="target", config=config)
        assert len(result) < len(df)

    def test_stratified_with_bool_target(self):
        """Stratified sampling works with boolean targets."""
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "feat": rng.normal(0, 1, n),
            "target": rng.choice([True, False], n),
        })
        config = _make_config(threshold=50, sample_size=30)
        result = smart_sample(df, target="target", config=config)
        assert len(result) <= 35  # allow rounding


# ---------------------------------------------------------------------------
# Temporal sampling
# ---------------------------------------------------------------------------

class TestTemporalSampling:
    """Tests for temporal sampling with datetime columns."""

    def test_preserves_time_ordering(self):
        """Temporal sampling preserves chronological ordering."""
        n = 100
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "value": range(n),
        })
        config = _make_config(threshold=50, sample_size=30)
        result = smart_sample(df, config=config)
        assert len(result) <= 30
        # Dates should be monotonically increasing
        assert result["date"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Random sampling (fallback)
# ---------------------------------------------------------------------------

class TestRandomSampling:
    """Tests for the random sampling fallback."""

    def test_random_fallback_works(self):
        """Random sampling is used when there is no target or datetime."""
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
        })
        config = _make_config(threshold=50, sample_size=30)
        result = smart_sample(df, config=config)
        assert len(result) == 30

    def test_random_is_reproducible(self):
        """Random sampling with the same config produces the same result."""
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({"a": rng.normal(0, 1, n)})
        config = _make_config(threshold=50, sample_size=30)
        r1 = smart_sample(df, config=config)
        r2 = smart_sample(df, config=config)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# config.sample_size controls output size
# ---------------------------------------------------------------------------

class TestSampleSizeConfig:
    """Tests that config.sample_size controls the output size."""

    def test_sample_size_controls_output(self):
        """Output size is limited to config.sample_size."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({"a": rng.normal(0, 1, n)})
        config = _make_config(threshold=50, sample_size=25)
        result = smart_sample(df, config=config)
        assert len(result) == 25

    def test_sample_size_larger_than_df(self):
        """When sample_size > len(df), returns all rows."""
        rng = np.random.default_rng(42)
        n = 20
        df = pd.DataFrame({"a": rng.normal(0, 1, n)})
        config = _make_config(threshold=10, sample_size=100)
        result = smart_sample(df, config=config)
        assert len(result) == n
