"""Tests for the top-level dissectml public API.

Covers attribute presence, types, callability, and lightweight
end-to-end smoke tests that do not require a full battle run.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import dissectml as dml

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df():
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "survived": rng.choice([0, 1], n),
    })


# ---------------------------------------------------------------------------
# Module-level attributes and version
# ---------------------------------------------------------------------------

class TestModuleAttributes:
    def test_import_succeeds(self):
        # If we reach this point dissectml imported cleanly above.
        assert dml is not None

    def test_version_is_string(self):
        assert isinstance(dml.__version__, str)

    def test_version_not_empty(self):
        assert len(dml.__version__) > 0


# ---------------------------------------------------------------------------
# Callables
# ---------------------------------------------------------------------------

class TestCallables:
    def test_explore_is_callable(self):
        assert callable(dml.explore)

    def test_battle_is_callable(self):
        assert callable(dml.battle)

    def test_analyze_is_callable(self):
        assert callable(dml.analyze)

    def test_analyze_intelligence_is_callable(self):
        assert callable(dml.analyze_intelligence)

    def test_load_titanic_is_callable(self):
        assert callable(dml.load_titanic)

    def test_load_housing_is_callable(self):
        assert callable(dml.load_housing)

    def test_to_pandas_is_callable(self):
        assert callable(dml.to_pandas)

    def test_get_config_is_callable(self):
        assert callable(dml.get_config)

    def test_set_config_is_callable(self):
        assert callable(dml.set_config)

    def test_config_context_is_callable(self):
        assert callable(dml.config_context)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class TestClasses:
    def test_model_comparator_is_class(self):
        assert inspect.isclass(dml.ModelComparator)

    def test_analysis_report_is_class(self):
        assert inspect.isclass(dml.AnalysisReport)

    def test_insight_ml_config_is_class(self):
        assert inspect.isclass(dml.DissectMLConfig)

    def test_insight_ml_error_is_exception_class(self):
        assert inspect.isclass(dml.DissectMLError)
        assert issubclass(dml.DissectMLError, Exception)


# ---------------------------------------------------------------------------
# Dataset loaders return DataFrames
# ---------------------------------------------------------------------------

class TestDatasetLoaders:
    def test_load_titanic_returns_dataframe(self):
        df = dml.load_titanic()
        assert isinstance(df, pd.DataFrame)

    def test_load_housing_returns_dataframe(self):
        df = dml.load_housing()
        assert isinstance(df, pd.DataFrame)

    def test_to_pandas_from_dict_returns_dataframe(self):
        result = dml.to_pandas({"a": [1, 2, 3]})
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# analyze() behaviour
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_analyze_raises_key_error_for_missing_target(self, small_df):
        with pytest.raises(KeyError):
            dml.analyze(small_df, target="nonexistent")

    def test_analyze_no_battle_returns_analysis_report(self, small_df):
        report = dml.analyze(small_df, target="survived", run_battle=False)
        assert isinstance(report, dml.AnalysisReport)

    def test_analyze_no_battle_report_has_correct_task_inferred(self, small_df):
        report = dml.analyze(small_df, target="survived", run_battle=False)
        # survived is binary 0/1 — should be inferred as classification
        assert report.task == "classification"

    def test_analyze_no_battle_report_stores_target(self, small_df):
        report = dml.analyze(small_df, target="survived", run_battle=False)
        assert report.target == "survived"

    def test_analyze_no_battle_report_has_eda(self, small_df):
        report = dml.analyze(small_df, target="survived", run_battle=False)
        assert report.eda is not None

    def test_analyze_no_battle_report_has_intelligence(self, small_df):
        report = dml.analyze(small_df, target="survived", run_battle=False)
        assert report.intelligence is not None

    def test_analyze_no_battle_models_is_none(self, small_df):
        report = dml.analyze(small_df, target="survived", run_battle=False)
        assert report.models is None

    def test_analyze_no_battle_compare_is_none(self, small_df):
        report = dml.analyze(small_df, target="survived", run_battle=False)
        assert report.compare is None
