"""Stage 4: Comparative analysis — metrics, curves, significance, error analysis."""

from dissectml.compare.comparator import ModelComparator
from dissectml.compare.curves import (
    actual_vs_predicted,
    confusion_matrices,
    metric_bar_chart,
    pr_curves,
    residual_plots,
    roc_curves,
)
from dissectml.compare.error_analysis import ErrorAnalysisResult, analyze_errors
from dissectml.compare.metrics_table import ComparisonTable
from dissectml.compare.pareto import get_pareto_models, pareto_front
from dissectml.compare.significance import corrected_ttest_matrix, mcnemar_matrix

__all__ = [
    "ModelComparator",
    "ComparisonTable",
    "analyze_errors",
    "ErrorAnalysisResult",
    "pareto_front",
    "get_pareto_models",
    "roc_curves",
    "pr_curves",
    "confusion_matrices",
    "residual_plots",
    "actual_vs_predicted",
    "metric_bar_chart",
    "mcnemar_matrix",
    "corrected_ttest_matrix",
]
