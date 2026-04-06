"""Tests for ModelRegistry."""

import pytest

from dissectml.battle.catalog import MODEL_CATALOG, ModelEntry, get_classifiers, get_regressors
from dissectml.battle.registry import ModelRegistry, get_registry


class TestModelEntry:
    def test_build_returns_estimator(self):
        from sklearn.ensemble import RandomForestClassifier
        entry = next(e for e in MODEL_CATALOG.values() if e.name == "RandomForestClassifier")
        est = entry.build()
        assert hasattr(est, "fit")
        assert isinstance(est, RandomForestClassifier)

    def test_core_models_available(self):
        entry = next(e for e in MODEL_CATALOG.values() if e.name == "LogisticRegression")
        assert entry.is_available()

    def test_optional_model_stub_raises(self):
        # CatBoost may or may not be installed; if not, build() should raise ImportError
        entry = next(e for e in MODEL_CATALOG.values() if e.name == "CatBoostClassifier")
        if not entry.is_available():
            with pytest.raises(ImportError):
                entry.build()


class TestCatalogFunctions:
    def test_get_classifiers(self):
        classifiers = get_classifiers(include_optional=False)
        assert len(classifiers) >= 10
        for e in classifiers:
            assert e.task in ("classification", "both")

    def test_get_regressors(self):
        regressors = get_regressors(include_optional=False)
        assert len(regressors) >= 10
        for e in regressors:
            assert e.task in ("regression", "both")


class TestModelRegistry:
    def test_default_registry_has_models(self):
        reg = get_registry()
        assert len(reg.names()) > 20

    def test_available_classification(self):
        reg = ModelRegistry()
        entries = reg.available("classification", include_optional=False)
        assert len(entries) >= 10
        assert all(e.task in ("classification", "both") for e in entries)

    def test_available_regression(self):
        reg = ModelRegistry()
        entries = reg.available("regression", include_optional=False)
        assert len(entries) >= 10
        assert all(e.task in ("regression", "both") for e in entries)

    def test_get_existing(self):
        reg = ModelRegistry()
        entry = reg.get("RandomForestClassifier")
        assert entry.name == "RandomForestClassifier"

    def test_get_missing_raises(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError, match="not in registry"):
            reg.get("NonExistentModel")

    def test_register_custom(self):
        from sklearn.dummy import DummyClassifier
        reg = ModelRegistry()
        custom = ModelEntry(
            name="DummyClassifier",
            family="baseline",
            task="classification",
            estimator_cls=DummyClassifier,
            default_params={"strategy": "most_frequent"},
        )
        reg.register(custom)
        assert reg.get("DummyClassifier").name == "DummyClassifier"

    def test_unregister(self):
        reg = ModelRegistry()
        reg.unregister("GaussianNB")
        assert "GaussianNB" not in reg.names()

    def test_unregister_missing_raises(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.unregister("DoesNotExist")

    def test_filter_by_family(self):
        reg = ModelRegistry()
        entries = reg.filter(task="classification", families=["linear"])
        assert all(e.family == "linear" for e in entries)

    def test_filter_exclude(self):
        reg = ModelRegistry()
        entries = reg.filter(task="classification", exclude=["LogisticRegression"])
        names = [e.name for e in entries]
        assert "LogisticRegression" not in names

    def test_repr(self):
        reg = ModelRegistry()
        r = repr(reg)
        assert "classifiers" in r
        assert "regressors" in r
