"""Additional EDAResult tests for coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insightml.eda.result import EDAResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_simple():
    """Simple 40-row DataFrame for testing EDAResult."""
    rng = np.random.default_rng(42)
    n = 40
    return pd.DataFrame({
        "num1": rng.normal(0, 1, n),
        "num2": rng.normal(5, 2, n),
        "cat1": rng.choice(["a", "b", "c"], n),
        "target": rng.choice([0, 1], n),
    })


@pytest.fixture
def eda(df_simple):
    return EDAResult(df_simple, target="target")


@pytest.fixture
def eda_no_target(df_simple):
    return EDAResult(df_simple)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_creates_eda_result(self, df_simple):
        result = EDAResult(df_simple)
        assert isinstance(result, EDAResult)

    def test_with_target(self, df_simple):
        result = EDAResult(df_simple, target="target")
        assert result._target == "target"

    def test_stage_name(self, eda):
        assert eda.stage_name == "EDA"

    def test_repr(self, eda):
        r = repr(eda)
        assert "EDAResult" in r
        assert "rows=40" in r
        assert "cols=4" in r
        assert "target='target'" in r

    def test_repr_no_target(self, eda_no_target):
        r = repr(eda_no_target)
        assert "target=None" in r


# ---------------------------------------------------------------------------
# Lazy properties accessible
# ---------------------------------------------------------------------------

class TestLazyProperties:
    def test_overview_accessible(self, eda):
        ov = eda.overview
        assert ov is not None
        from insightml.eda.overview import DataOverview
        assert isinstance(ov, DataOverview)

    def test_univariate_accessible(self, eda):
        u = eda.univariate
        assert u is not None
        from insightml.eda.univariate import UnivariateAnalysis
        assert isinstance(u, UnivariateAnalysis)

    def test_bivariate_accessible(self, eda):
        bv = eda.bivariate
        assert bv is not None

    def test_correlations_accessible(self, eda):
        c = eda.correlations
        assert c is not None

    def test_missing_accessible(self, eda):
        m = eda.missing
        assert m is not None

    def test_outliers_accessible(self, eda):
        o = eda.outliers
        assert o is not None

    def test_tests_accessible(self, eda):
        t = eda.tests
        assert t is not None

    def test_clusters_accessible(self, eda):
        cl = eda.clusters
        assert cl is not None

    def test_interactions_accessible(self, eda):
        inter = eda.interactions
        assert inter is not None

    def test_target_accessible_with_target(self, eda):
        t = eda.target
        assert t is not None

    def test_target_none_without_target(self, eda_no_target):
        assert eda_no_target.target is None

    def test_properties_are_cached(self, eda):
        """Accessing twice returns the same object."""
        assert eda.overview is eda.overview
        assert eda.univariate is eda.univariate


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------

class TestToDict:
    def test_returns_dict(self, eda):
        d = eda.to_dict()
        assert isinstance(d, dict)

    def test_has_stage_name(self, eda):
        d = eda.to_dict()
        assert "stage_name" in d
        assert d["stage_name"] == "EDA"

    def test_has_duration(self, eda):
        d = eda.to_dict()
        assert "duration_seconds" in d

    def test_includes_accessed_submodules(self, eda):
        """Only accessed sub-modules appear in to_dict()."""
        # Access overview
        _ = eda.overview
        d = eda.to_dict()
        assert "overview" in d

    def test_excludes_unaccessed_submodules(self, df_simple):
        """Sub-modules not accessed should not appear in to_dict()."""
        result = EDAResult(df_simple, target="target")
        d = result.to_dict()
        # Only stage_name and duration_seconds should be present
        assert "univariate" not in d
        assert "correlations" not in d


# ---------------------------------------------------------------------------
# _repr_html_()
# ---------------------------------------------------------------------------

class TestReprHTML:
    def test_returns_string(self, eda):
        html = eda._repr_html_()
        assert isinstance(html, str)

    def test_contains_eda_result(self, eda):
        html = eda._repr_html_()
        assert "EDAResult" in html

    def test_contains_row_count(self, eda):
        html = eda._repr_html_()
        assert "40" in html

    def test_contains_col_count(self, eda):
        html = eda._repr_html_()
        assert "4" in html


# ---------------------------------------------------------------------------
# show()
# ---------------------------------------------------------------------------

class TestShow:
    def test_show_does_not_crash(self, eda, monkeypatch):
        """show() should not raise; monkeypatch display_html to no-op."""
        monkeypatch.setattr(
            "insightml.viz.display.display_html", lambda html: None
        )
        monkeypatch.setattr(
            "insightml.eda.result.display_html", lambda html: None
        )
        eda.show()

    def test_show_no_target_does_not_crash(self, eda_no_target, monkeypatch):
        monkeypatch.setattr(
            "insightml.viz.display.display_html", lambda html: None
        )
        monkeypatch.setattr(
            "insightml.eda.result.display_html", lambda html: None
        )
        eda_no_target.show()
