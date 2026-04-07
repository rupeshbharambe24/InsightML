# Comparative Analysis

`ModelComparator` provides a unified facade for all comparison views.
All properties are lazy `@cached_property` — computed on first access.

## Basic Usage

```python
import dissectml as dml

comp = dml.ModelComparator(battle_result, y=df["target"])

comp.table            # ComparisonTable (styled DataFrame)
comp.pareto           # Plotly scatter: accuracy vs speed Pareto front
comp.metric_bar       # Horizontal bar chart of primary metric
comp.significance     # Statistical significance tests
comp.error_analysis   # Cross-model disagreement and hard samples
```

## Comparison Table

```python
t = comp.table
t.styled()        # Jupyter-styled with gradient + bold best values
t.to_latex()      # LaTeX table for papers
t.to_markdown()   # Markdown
```

## Statistical Significance

```python
sig = comp.significance

# McNemar test (classification)
sig["mcnemar"]["p_matrix"]    # n_models × n_models p-value matrix
sig["mcnemar"]["figure"]      # Heatmap

# Corrected paired t-test (Nadeau & Bengio, works for all tasks)
sig["ttest"]["p_matrix"]
sig["ttest"]["figure"]
```

The corrected t-test adjusts for the overlap between training sets in CV folds:

```
var_corrected = (1/k + n_test/n_train) × var(score_diffs)
```

## Error Analysis

```python
ea = comp.error_analysis

ea.disagreement          # Pairwise disagreement rate matrix
ea.hard_indices          # Indices of hardest samples (all models wrong)
ea.hard_sample_profile   # Feature statistics for hard vs easy samples
ea.ensemble_candidates() # Model pairs with high complementarity

ea.disagreement_figure().show()
ea.hard_sample_figure().show()
```

## Pareto Front

Interactive scatter plot: X = training time, Y = primary metric.
Pareto-optimal models are highlighted with a connecting step line.

```python
comp.pareto.show()
pareto_names = comp.pareto_models  # List of model names on the front
```

## Curve Plots

```python
# Classification
comp.roc_curves.show()       # All models overlaid
comp.pr_curves.show()        # Precision-Recall curves
comp.confusion_matrices.show()

# Regression
comp.residual_plots.show()
comp.actual_vs_predicted.show()
```
