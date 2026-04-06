"""Tests for insightml.core.pipeline — InsightPipeline orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insightml._config import InsightMLConfig
from insightml.core.data_container import DataContainer
from insightml.core.pipeline import InsightPipeline
from insightml.eda.result import EDAResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df() -> pd.DataFrame:
    """Small classification DataFrame for pipeline tests."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.normal(5, 2, n),
        "target": rng.choice(["A", "B", "C"], n),
    })


@pytest.fixture
def container(small_df: pd.DataFrame) -> DataContainer:
    """DataContainer built from the small classification DataFrame."""
    return DataContainer.from_input(small_df, target="target", task="classification")


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInsightPipelineInit:
    """Tests for InsightPipeline.__init__."""

    def test_creates_with_default_config(self):
        """InsightPipeline() uses default config when none is provided."""
        pipe = InsightPipeline()
        assert pipe.config is not None
        assert isinstance(pipe.config, InsightMLConfig)

    def test_creates_with_custom_config(self):
        """InsightPipeline(config) stores the custom config."""
        cfg = InsightMLConfig(cv_folds=10, verbosity=0)
        pipe = InsightPipeline(config=cfg)
        assert pipe.config.cv_folds == 10
        assert pipe.config.verbosity == 0

    def test_context_is_created(self):
        """The pipeline creates a PipelineContext on init."""
        pipe = InsightPipeline()
        assert pipe.context is not None


# ---------------------------------------------------------------------------
# run_eda
# ---------------------------------------------------------------------------

class TestRunEDA:
    """Tests for InsightPipeline.run_eda."""

    def test_returns_eda_result(self, container: DataContainer):
        """run_eda() returns an EDAResult instance."""
        pipe = InsightPipeline()
        result = pipe.run_eda(container)
        assert isinstance(result, EDAResult)

    def test_stores_result_in_context(self, container: DataContainer):
        """run_eda() stores the result in context.eda_result."""
        pipe = InsightPipeline()
        result = pipe.run_eda(container)
        assert pipe.context.eda_result is result


# ---------------------------------------------------------------------------
# run (full pipeline stub)
# ---------------------------------------------------------------------------

class TestRun:
    """Tests for InsightPipeline.run (full pipeline — currently partial stub)."""

    def test_raises_not_implemented(self, small_df: pd.DataFrame):
        """run() raises NotImplementedError for stages 2-5."""
        pipe = InsightPipeline()
        with pytest.raises(NotImplementedError, match="Full pipeline"):
            pipe.run(small_df, target="target", task="classification")
