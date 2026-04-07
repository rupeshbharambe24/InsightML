# Model Battle

`dml.battle(df, target)` runs parallel cross-validation across a configurable model
catalogue and returns a `BattleResult` with sorted leaderboard and OOF predictions.

## Basic Usage

```python
models = dml.battle(df, target="survived")
models.leaderboard()          # Sorted by primary metric
models.best                   # Best ModelScore
models.get("RandomForest")    # Specific model score
```

## Filtering Models

```python
# By family
models = dml.battle(df, target="survived", families=["tree", "linear"])

# By name
models = dml.battle(df, target="survived", models=["RandomForest", "LogisticRegression"])

# Excluding models
models = dml.battle(df, target="survived", exclude=["SVC", "KNN"])
```

## Task Inference

DissectML infers `"classification"` or `"regression"` automatically:

- Boolean/categorical/object dtypes → classification
- Numeric with ≤ 20 unique values → classification
- Otherwise → regression

Override with `task="regression"`.

## Preprocessing

Preprocessing is informed by EDA results when `dml.analyze()` is used. In standalone
`battle()` calls, heuristics are applied:

- **> 20% missing values** → KNN imputer (instead of mean/most-frequent)
- **> 30% columns with |skew| > 2** → Robust scaler (instead of standard)
- **High-cardinality categoricals (> 15 unique)** → Ordinal encoder
- **Tree-based models** skip scaling entirely

## Metrics

| Classification | Regression |
|---|---|
| Accuracy | R² |
| F1 (weighted) | MAE |
| Precision (weighted) | RMSE |
| Recall (weighted) | MAPE |
| ROC-AUC (OVR weighted) | |

## Custom Models

```python
from dissectml.battle.catalog import ModelEntry
from dissectml.battle.registry import get_registry

registry = get_registry()
registry.register(ModelEntry(
    name="MyModel",
    task="classification",
    family="custom",
    cls=MyEstimator,
    params={"alpha": 0.1},
))

models = dml.battle(df, target="survived", registry=registry)
```

## Hyperparameter Tuning

```python
models = dml.battle(df, target="survived", tune=True, top_n=3, n_iter=20)
```

Runs `RandomizedSearchCV` on the top-N models using default search spaces from
`battle/param_grids.py`.
