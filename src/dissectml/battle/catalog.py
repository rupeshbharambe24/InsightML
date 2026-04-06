"""MODEL_CATALOG — all supported classifiers and regressors with metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelEntry:
    """Metadata + constructor for one model in the catalog."""
    name: str                        # Unique display name, e.g. "RandomForest"
    family: str                      # Algorithm family, e.g. "tree", "linear", "kernel"
    task: str                        # "classification" | "regression" | "both"
    estimator_cls: type              # sklearn-compatible estimator class
    default_params: dict[str, Any]   # params passed to estimator constructor
    tree_based: bool = False         # skip scaling for tree-based models
    linear: bool = False             # use LinearExplainer for SHAP
    is_optional: bool = False        # requires an optional extra package
    optional_package: str = ""       # import name of optional package

    def build(self) -> Any:
        """Instantiate the estimator with default params."""
        return self.estimator_cls(**self.default_params)

    def is_available(self) -> bool:
        """Check if optional package is importable (always True for core models)."""
        if not self.is_optional:
            return True
        try:
            import importlib
            importlib.import_module(self.optional_package)
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# Build catalog
# ---------------------------------------------------------------------------

def _build_catalog() -> dict[str, ModelEntry]:
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.ensemble import (
        AdaBoostClassifier,
        AdaBoostRegressor,
        BaggingClassifier,
        BaggingRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import (
        ElasticNet,
        HuberRegressor,
        Lasso,
        LinearRegression,
        LogisticRegression,
        Ridge,
        RidgeClassifier,
        SGDClassifier,
        SGDRegressor,
    )
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    entries: list[ModelEntry] = [
        # ---- CLASSIFIERS ----
        ModelEntry("LogisticRegression", "linear", "classification",
                   LogisticRegression,
                   {"max_iter": 1000, "random_state": 42, "n_jobs": -1},
                   linear=True),
        ModelEntry("RidgeClassifier", "linear", "classification",
                   RidgeClassifier, {"random_state": 42}, linear=True),
        ModelEntry("SGDClassifier", "linear", "classification",
                   SGDClassifier,
                   {"max_iter": 1000, "random_state": 42, "n_jobs": -1},
                   linear=True),
        ModelEntry("GaussianNB", "naive_bayes", "classification",
                   GaussianNB, {}),
        ModelEntry("LinearDiscriminantAnalysis", "linear", "classification",
                   LinearDiscriminantAnalysis, {}, linear=True),
        ModelEntry("QuadraticDiscriminantAnalysis", "kernel", "classification",
                   QuadraticDiscriminantAnalysis, {}),
        ModelEntry("KNeighborsClassifier", "neighbors", "classification",
                   KNeighborsClassifier, {"n_neighbors": 5, "n_jobs": -1}),
        ModelEntry("DecisionTreeClassifier", "tree", "classification",
                   DecisionTreeClassifier, {"random_state": 42}, tree_based=True),
        ModelEntry("RandomForestClassifier", "tree", "classification",
                   RandomForestClassifier,
                   {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
                   tree_based=True),
        ModelEntry("ExtraTreesClassifier", "tree", "classification",
                   ExtraTreesClassifier,
                   {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
                   tree_based=True),
        ModelEntry("GradientBoostingClassifier", "tree", "classification",
                   GradientBoostingClassifier,
                   {"n_estimators": 100, "random_state": 42},
                   tree_based=True),
        ModelEntry("BaggingClassifier", "ensemble", "classification",
                   BaggingClassifier,
                   {"n_estimators": 20, "random_state": 42, "n_jobs": -1}),
        ModelEntry("AdaBoostClassifier", "ensemble", "classification",
                   AdaBoostClassifier,
                   {"n_estimators": 50, "random_state": 42, "algorithm": "SAMME"}),
        ModelEntry("SVC_linear", "kernel", "classification",
                   SVC, {"kernel": "linear", "probability": True, "random_state": 42}),
        ModelEntry("SVC_rbf", "kernel", "classification",
                   SVC, {"kernel": "rbf", "probability": True, "random_state": 42}),
        ModelEntry("MLPClassifier", "neural_net", "classification",
                   MLPClassifier,
                   {"hidden_layer_sizes": (100,), "max_iter": 300, "random_state": 42}),
        # Optional boost classifiers
        ModelEntry("XGBClassifier", "tree", "classification",
                   _lazy_cls("xgboost", "XGBClassifier"),
                   {"n_estimators": 100, "random_state": 42,
                    "use_label_encoder": False, "eval_metric": "logloss",
                    "verbosity": 0, "n_jobs": -1},
                   tree_based=True, is_optional=True, optional_package="xgboost"),
        ModelEntry("LGBMClassifier", "tree", "classification",
                   _lazy_cls("lightgbm", "LGBMClassifier"),
                   {"n_estimators": 100, "random_state": 42,
                    "verbosity": -1, "n_jobs": -1},
                   tree_based=True, is_optional=True, optional_package="lightgbm"),
        ModelEntry("CatBoostClassifier", "tree", "classification",
                   _lazy_cls("catboost", "CatBoostClassifier"),
                   {"iterations": 100, "random_seed": 42, "verbose": 0},
                   tree_based=True, is_optional=True, optional_package="catboost"),

        # ---- REGRESSORS ----
        ModelEntry("LinearRegression", "linear", "regression",
                   LinearRegression, {"n_jobs": -1}, linear=True),
        ModelEntry("Ridge", "linear", "regression",
                   Ridge, {"random_state": 42}, linear=True),
        ModelEntry("Lasso", "linear", "regression",
                   Lasso, {"max_iter": 2000, "random_state": 42}, linear=True),
        ModelEntry("ElasticNet", "linear", "regression",
                   ElasticNet, {"max_iter": 2000, "random_state": 42}, linear=True),
        ModelEntry("HuberRegressor", "linear", "regression",
                   HuberRegressor, {"max_iter": 300}, linear=True),
        ModelEntry("SGDRegressor", "linear", "regression",
                   SGDRegressor, {"max_iter": 1000, "random_state": 42}, linear=True),
        ModelEntry("KNeighborsRegressor", "neighbors", "regression",
                   KNeighborsRegressor, {"n_neighbors": 5, "n_jobs": -1}),
        ModelEntry("DecisionTreeRegressor", "tree", "regression",
                   DecisionTreeRegressor, {"random_state": 42}, tree_based=True),
        ModelEntry("RandomForestRegressor", "tree", "regression",
                   RandomForestRegressor,
                   {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
                   tree_based=True),
        ModelEntry("ExtraTreesRegressor", "tree", "regression",
                   ExtraTreesRegressor,
                   {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
                   tree_based=True),
        ModelEntry("GradientBoostingRegressor", "tree", "regression",
                   GradientBoostingRegressor,
                   {"n_estimators": 100, "random_state": 42},
                   tree_based=True),
        ModelEntry("BaggingRegressor", "ensemble", "regression",
                   BaggingRegressor,
                   {"n_estimators": 20, "random_state": 42, "n_jobs": -1}),
        ModelEntry("AdaBoostRegressor", "ensemble", "regression",
                   AdaBoostRegressor,
                   {"n_estimators": 50, "random_state": 42}),
        ModelEntry("SVR_linear", "kernel", "regression",
                   SVR, {"kernel": "linear"}),
        ModelEntry("SVR_rbf", "kernel", "regression",
                   SVR, {"kernel": "rbf"}),
        ModelEntry("MLPRegressor", "neural_net", "regression",
                   MLPRegressor,
                   {"hidden_layer_sizes": (100,), "max_iter": 300, "random_state": 42}),
        # Optional boost regressors
        ModelEntry("XGBRegressor", "tree", "regression",
                   _lazy_cls("xgboost", "XGBRegressor"),
                   {"n_estimators": 100, "random_state": 42, "verbosity": 0,
                    "n_jobs": -1},
                   tree_based=True, is_optional=True, optional_package="xgboost"),
        ModelEntry("LGBMRegressor", "tree", "regression",
                   _lazy_cls("lightgbm", "LGBMRegressor"),
                   {"n_estimators": 100, "random_state": 42,
                    "verbosity": -1, "n_jobs": -1},
                   tree_based=True, is_optional=True, optional_package="lightgbm"),
        ModelEntry("CatBoostRegressor", "tree", "regression",
                   _lazy_cls("catboost", "CatBoostRegressor"),
                   {"iterations": 100, "random_seed": 42, "verbose": 0},
                   tree_based=True, is_optional=True, optional_package="catboost"),
    ]

    return {e.name: e for e in entries}


def _lazy_cls(package: str, cls_name: str) -> type:
    """Return a placeholder class that imports lazily. Falls back to a stub."""
    try:
        import importlib
        mod = importlib.import_module(package)
        return getattr(mod, cls_name)
    except (ImportError, AttributeError):
        # Return a stub class so catalog builds without the optional package
        class _Stub:
            def __init__(self, **kwargs):
                raise ImportError(
                    f"Optional package '{package}' is not installed. "
                    f"Install with: pip install dissectml[boost]"
                )
        _Stub.__name__ = cls_name
        _Stub.__qualname__ = cls_name
        return _Stub


# Singleton catalog
MODEL_CATALOG: dict[str, ModelEntry] = _build_catalog()


def get_classifiers(include_optional: bool = True) -> list[ModelEntry]:
    """Return all classification model entries."""
    return [
        e for e in MODEL_CATALOG.values()
        if e.task in ("classification", "both")
        and (not e.is_optional or (include_optional and e.is_available()))
    ]


def get_regressors(include_optional: bool = True) -> list[ModelEntry]:
    """Return all regression model entries."""
    return [
        e for e in MODEL_CATALOG.values()
        if e.task in ("regression", "both")
        and (not e.is_optional or (include_optional and e.is_available()))
    ]
