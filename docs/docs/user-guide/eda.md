# Deep EDA

`dml.explore(df)` returns an `EDAResult` object with lazy `@cached_property` sub-modules.
Computation only triggers when you access a sub-module — `explore()` returns instantly.

## Sub-modules

### Overview

```python
eda.overview.show()
eda.overview.to_dataframe()  # Per-column stats
```

Per-column stats: dtype, count, missing%, unique count, mean/median/std (numeric),
top value/frequency (categorical), date range (datetime).

### Correlations

```python
eda.correlations.unified()       # Auto-select measure per cell
eda.correlations.pearson()       # Numeric only
eda.correlations.cramers_v()     # Categorical only
eda.correlations.heatmap()       # Interactive Plotly heatmap
eda.correlations.top_correlations(n=20)
```

The **unified matrix** uses statistically appropriate measures per type pair:

| Cell Types | Measure |
|---|---|
| Numeric–Numeric | Pearson r |
| Categorical–Categorical | Cramér's V |
| Numeric–Binary | Point-biserial r |
| Numeric–Categorical | Correlation ratio (η) |

### Missing Data

```python
eda.missing.summary()      # Per-column missing %
eda.missing.patterns()     # Pattern matrix heatmap
eda.missing.mcar_test()    # Little's MCAR test result
```

Classification: MCAR → MAR → MNAR with imputation recommendations per column.

### Outliers

```python
eda.outliers.by_iqr()
eda.outliers.by_zscore()
eda.outliers.by_isolation_forest()
eda.outliers.consensus(min_methods=2)   # Agreement across methods
eda.outliers.comparison()              # Side-by-side bar chart
```

### Statistical Tests

```python
eda.tests.normality()          # Shapiro-Wilk + D'Agostino + Anderson-Darling
eda.tests.independence()       # Chi-square contingency
eda.tests.group_comparison()   # ANOVA or Kruskal-Wallis (auto-selected)
```

### Clusters

```python
eda.clusters.auto()            # Auto K-Means (silhouette-optimised) + DBSCAN
eda.clusters.plot()            # PCA / t-SNE 2D scatter
eda.clusters.profiles()        # Mean / mode per cluster
```

### Interactions

```python
eda.interactions.with_target(target="price")
eda.interactions.pairwise()
eda.interactions.nonlinear()   # Linear vs polynomial R² comparison
```
