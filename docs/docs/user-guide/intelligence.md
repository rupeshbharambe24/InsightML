# Intelligence Bridge

`dml.analyze_intelligence(df, target)` connects EDA findings to modelling decisions.
All sub-modules are lazy — no computation until accessed.

## Leakage Detection

DissectML uses a four-pronged scan:

| Method | Threshold | Description |
|---|---|---|
| High correlation | > 0.95 | Pearson/Cramér's V/point-biserial |
| Mutual information | > 1.0 or top 1% | `sklearn.feature_selection` MI |
| Temporal leakage | Δcorr > 0.30 | Future vs past correlation difference |
| Derived feature | OLS R² > 0.98 | Near-perfect linear reconstruction |

```python
intel = dml.analyze_intelligence(df, target="price", datetime_col="sale_date")

for w in intel.leakage:
    print(f"{w.column}: {w.method} (score={w.score:.3f}, severity={w.severity})")
```

## Multicollinearity

```python
intel.vif                          # DataFrame with VIF per feature
intel.condition_number             # Eigenvalue-based condition number
intel.multicollinearity_recommendations  # Which features to remove
```

VIF thresholds: < 5 low, 5–10 moderate, > 10 high.

## Feature Importance

Pre-model composite ranking combining:

1. Mutual information
2. Absolute correlation with target
3. ANOVA F-score (numeric) / chi-square (categorical, classification)

```python
fi = intel.feature_importance
fi.sort_values("composite_rank").head(10)
```

## Data Readiness Score

A 0–100 score with letter grade (A–F):

```python
r = intel.readiness
print(f"Score: {r.score}/100 (grade {r.grade})")
r.waterfall_figure().show()   # Penalty/bonus breakdown
r.gauge_figure().show()       # Gauge chart
```

**Scoring breakdown:**

| Category | Max Penalty |
|---|---|
| Missing values | −25 |
| Class imbalance | −20 |
| Multicollinearity | −15 |
| Outliers | −10 |
| Constant features | −2 each |
| Sample size bonus | +10 |
| Feature diversity bonus | +5 |

## Algorithm Recommendations

```python
for rec in intel.recommendations.ranked:
    print(f"{rec.algorithm}: score={rec.score:.0f} — {rec.reasoning}")
```
