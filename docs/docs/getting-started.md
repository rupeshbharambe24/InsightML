# Getting Started

## Installation

```bash
pip install insightml
```

## Your First Analysis

```python
import insightml as iml

# Load a built-in demo dataset
df = iml.load_titanic()

# Full pipeline: EDA → Intelligence → Battle → Compare → Report
report = iml.analyze(df, target="survived")

# Inspect interactively
report.eda.overview.show()
report.eda.correlations.heatmap()
report.models.leaderboard()

# Export a self-contained HTML report
report.export("titanic_report.html")
```

## Individual Stages

### Stage 1 — Deep EDA

```python
eda = iml.explore(df)

eda.overview.show()              # Dataset shape, types, memory
eda.correlations.unified()       # Mixed-type correlation matrix
eda.missing.patterns()           # MCAR/MAR/MNAR classification
eda.outliers.consensus()         # Multi-method outlier detection
eda.clusters.auto()              # Auto K-Means + DBSCAN
```

### Stage 2 — Intelligence Bridge

```python
intel = iml.analyze_intelligence(df, target="survived")

intel.leakage              # List of leakage warnings
intel.readiness.score      # 0–100 data readiness score
intel.recommendations      # Algorithm recommendations
```

### Stage 3 — Model Battle

```python
models = iml.battle(df, target="survived", task="classification")

models.leaderboard()       # Sorted CV scores table
models.best                # Best ModelScore
```

### Stage 4 — Comparative Analysis

```python
comp = iml.ModelComparator(models, y=df["survived"])

comp.pareto                # Accuracy vs speed Pareto front
comp.significance          # McNemar p-value matrix
comp.error_analysis        # Cross-model disagreement
```

## Configuration

```python
iml.set_config(cv_folds=10, n_jobs=4, random_state=0)

# Or use a context manager for a single call
with iml.config_context(cv_folds=3):
    report = iml.analyze(df, target="survived")
```
