"""Default hyperparameter search spaces for RandomizedSearchCV."""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform

# Each entry maps model name -> param_grid dict
# Keys must match Pipeline step naming: "model__<param_name>"

CLASSIFICATION_GRIDS: dict[str, dict] = {
    "LogisticRegression": {
        "model__C": loguniform(1e-3, 1e3),
        "model__solver": ["lbfgs", "liblinear", "saga"],
        "model__penalty": ["l1", "l2", "elasticnet", None],
    },
    "RidgeClassifier": {
        "model__alpha": loguniform(1e-3, 1e3),
    },
    "SGDClassifier": {
        "model__alpha": loguniform(1e-5, 1e-1),
        "model__loss": ["hinge", "log_loss", "modified_huber"],
        "model__learning_rate": ["optimal", "adaptive"],
    },
    "KNeighborsClassifier": {
        "model__n_neighbors": randint(3, 20),
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan", "minkowski"],
    },
    "DecisionTreeClassifier": {
        "model__max_depth": [None, 3, 5, 10, 20],
        "model__min_samples_split": randint(2, 20),
        "model__min_samples_leaf": randint(1, 10),
        "model__criterion": ["gini", "entropy"],
    },
    "RandomForestClassifier": {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": randint(2, 15),
        "model__max_features": ["sqrt", "log2", None],
    },
    "ExtraTreesClassifier": {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": randint(2, 15),
        "model__max_features": ["sqrt", "log2", None],
    },
    "GradientBoostingClassifier": {
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": loguniform(1e-3, 0.5),
        "model__max_depth": randint(2, 8),
        "model__subsample": uniform(0.6, 0.4),
    },
    "AdaBoostClassifier": {
        "model__n_estimators": randint(30, 200),
        "model__learning_rate": loguniform(1e-2, 2.0),
    },
    "BaggingClassifier": {
        "model__n_estimators": randint(10, 100),
        "model__max_samples": uniform(0.5, 0.5),
        "model__max_features": uniform(0.5, 0.5),
    },
    "SVC_linear": {
        "model__C": loguniform(1e-3, 1e3),
    },
    "SVC_rbf": {
        "model__C": loguniform(1e-3, 1e3),
        "model__gamma": ["scale", "auto"] + list(loguniform(1e-4, 1.0).rvs(3, random_state=42)),
    },
    "MLPClassifier": {
        "model__hidden_layer_sizes": [(50,), (100,), (200,), (100, 50), (200, 100)],
        "model__alpha": loguniform(1e-5, 1e-1),
        "model__learning_rate_init": loguniform(1e-4, 1e-2),
        "model__activation": ["relu", "tanh"],
    },
    "XGBClassifier": {
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": loguniform(1e-3, 0.5),
        "model__max_depth": randint(2, 10),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
    },
    "LGBMClassifier": {
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": loguniform(1e-3, 0.5),
        "model__num_leaves": randint(15, 63),
        "model__subsample": uniform(0.6, 0.4),
    },
    "CatBoostClassifier": {
        "model__iterations": randint(50, 300),
        "model__learning_rate": loguniform(1e-3, 0.5),
        "model__depth": randint(3, 8),
    },
}

REGRESSION_GRIDS: dict[str, dict] = {
    "LinearRegression": {},  # No hyperparameters
    "Ridge": {
        "model__alpha": loguniform(1e-3, 1e3),
    },
    "Lasso": {
        "model__alpha": loguniform(1e-3, 10.0),
    },
    "ElasticNet": {
        "model__alpha": loguniform(1e-3, 10.0),
        "model__l1_ratio": uniform(0.0, 1.0),
    },
    "HuberRegressor": {
        "model__alpha": loguniform(1e-5, 1.0),
        "model__epsilon": uniform(1.1, 1.9),
    },
    "SGDRegressor": {
        "model__alpha": loguniform(1e-5, 1e-1),
        "model__loss": ["squared_error", "huber", "epsilon_insensitive"],
        "model__learning_rate": ["optimal", "adaptive", "invscaling"],
    },
    "KNeighborsRegressor": {
        "model__n_neighbors": randint(3, 20),
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan", "minkowski"],
    },
    "DecisionTreeRegressor": {
        "model__max_depth": [None, 3, 5, 10, 20],
        "model__min_samples_split": randint(2, 20),
        "model__min_samples_leaf": randint(1, 10),
        "model__criterion": ["squared_error", "absolute_error"],
    },
    "RandomForestRegressor": {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": randint(2, 15),
        "model__max_features": ["sqrt", "log2", None],
    },
    "ExtraTreesRegressor": {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": [None, 5, 10, 20],
        "model__max_features": ["sqrt", "log2", None],
    },
    "GradientBoostingRegressor": {
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": loguniform(1e-3, 0.5),
        "model__max_depth": randint(2, 8),
        "model__subsample": uniform(0.6, 0.4),
    },
    "AdaBoostRegressor": {
        "model__n_estimators": randint(30, 200),
        "model__learning_rate": loguniform(1e-2, 2.0),
        "model__loss": ["linear", "square", "exponential"],
    },
    "BaggingRegressor": {
        "model__n_estimators": randint(10, 100),
        "model__max_samples": uniform(0.5, 0.5),
        "model__max_features": uniform(0.5, 0.5),
    },
    "SVR_linear": {
        "model__C": loguniform(1e-3, 1e3),
        "model__epsilon": loguniform(1e-3, 1.0),
    },
    "SVR_rbf": {
        "model__C": loguniform(1e-3, 1e3),
        "model__gamma": ["scale", "auto"],
        "model__epsilon": loguniform(1e-3, 1.0),
    },
    "MLPRegressor": {
        "model__hidden_layer_sizes": [(50,), (100,), (200,), (100, 50), (200, 100)],
        "model__alpha": loguniform(1e-5, 1e-1),
        "model__learning_rate_init": loguniform(1e-4, 1e-2),
        "model__activation": ["relu", "tanh"],
    },
    "XGBRegressor": {
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": loguniform(1e-3, 0.5),
        "model__max_depth": randint(2, 10),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
    },
    "LGBMRegressor": {
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": loguniform(1e-3, 0.5),
        "model__num_leaves": randint(15, 63),
        "model__subsample": uniform(0.6, 0.4),
    },
    "CatBoostRegressor": {
        "model__iterations": randint(50, 300),
        "model__learning_rate": loguniform(1e-3, 0.5),
        "model__depth": randint(3, 8),
    },
}


def get_param_grid(model_name: str, task: str) -> dict:
    """Return the param grid for a given model name and task."""
    grids = CLASSIFICATION_GRIDS if task == "classification" else REGRESSION_GRIDS
    return grids.get(model_name, {})
