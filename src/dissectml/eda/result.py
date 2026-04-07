"""EDAResult — lazy orchestrator for all EDA sub-modules."""

from __future__ import annotations

import time
from functools import cached_property
from typing import Any

import pandas as pd

from dissectml._config import DissectMLConfig, get_config
from dissectml._sampling import smart_sample
from dissectml.core.base import StageResult
from dissectml.viz.display import display_html


class EDAResult(StageResult):
    """Lazy-evaluated EDA result.

    Each sub-module is a ``@cached_property`` — computation only triggers
    when the attribute is first accessed.

    Usage::

        eda = dml.explore(df, target="survived")
        eda.overview.show()          # Type detection, column profiles
        eda.correlations.heatmap()   # Unified correlation matrix
        eda.missing.patterns()       # MCAR/MAR/MNAR analysis
        eda.outliers.plot()          # IQR + Z-score + Isolation Forest
        eda.tests.normality()        # Shapiro-Wilk for all numeric cols
        eda.clusters.scatter_2d()    # Auto K-Means cluster plot
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target: str | None = None,
        task: str = "auto",
        config: DissectMLConfig | None = None,
    ) -> None:
        super().__init__(stage_name="EDA", duration_seconds=0.0)
        self._df = df
        self._target = target
        self._task = task
        self._config = config or get_config()
        # Sampled DF for expensive operations on large datasets
        self._sample = smart_sample(df, target=target, config=self._config)

    # ------------------------------------------------------------------
    # Lazy sub-module properties
    # ------------------------------------------------------------------

    @cached_property
    def overview(self):
        """Dataset overview: type detection, column profiles, memory usage."""
        from dissectml.eda.overview import DataOverview
        return DataOverview(self._df, target=self._target, config=self._config)

    @cached_property
    def univariate(self):
        """Univariate analysis: distributions, descriptive stats, normality flags."""
        from dissectml.eda.univariate import UnivariateAnalysis
        return UnivariateAnalysis(self._sample, target=self._target, config=self._config)

    @cached_property
    def bivariate(self):
        """Bivariate analysis: scatter, ANOVA, chi-square by type pairs."""
        from dissectml.eda.bivariate import BivariateAnalysis
        return BivariateAnalysis(self._sample, target=self._target, config=self._config)

    @cached_property
    def correlations(self):
        """Unified correlation matrix: Pearson/Spearman/Cramer's V/eta."""
        from dissectml.eda.correlations import CorrelationAnalysis
        return CorrelationAnalysis(self._sample, target=self._target, config=self._config)

    @cached_property
    def missing(self):
        """Missing data intelligence: Little's MCAR test, MAR/MNAR classification."""
        from dissectml.eda.missing import MissingDataIntelligence
        # Missing counts use full DF; pattern analysis uses sample
        return MissingDataIntelligence(self._df, target=self._target, config=self._config)

    @cached_property
    def outliers(self):
        """Outlier detection: IQR, Z-score, Isolation Forest, consensus."""
        from dissectml.eda.outliers import OutlierDetection
        return OutlierDetection(self._sample, target=self._target, config=self._config)

    @cached_property
    def tests(self):
        """Statistical tests: normality, independence, variance, group comparison."""
        from dissectml.eda.statistical_tests import StatisticalTests
        return StatisticalTests(self._sample, target=self._target, config=self._config)

    @cached_property
    def clusters(self):
        """Cluster discovery: auto K-Means + DBSCAN, profiling, PCA/t-SNE viz."""
        from dissectml.eda.clusters import ClusterDiscovery
        return ClusterDiscovery(self._sample, target=self._target, config=self._config)

    @cached_property
    def interactions(self):
        """Feature interactions: interaction strength, non-linearity detection."""
        from dissectml.eda.interactions import FeatureInteractions
        return FeatureInteractions(self._sample, target=self._target, config=self._config)

    @cached_property
    def target(self):
        """Target-specific analysis: class balance, distribution, feature-target plots."""
        if self._target is None:
            return None
        from dissectml.eda.target_analysis import TargetAnalysis
        return TargetAnalysis(self._df, target=self._target, config=self._config)

    # ------------------------------------------------------------------
    # Aggregated display
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Display a dashboard-style overview highlighting key findings from all modules."""
        # Always-available: overview
        ov = self.overview
        html_parts = [
            "<div style='font-family:system-ui;max-width:900px'>",
            "<h2>DissectML EDA Report</h2>",
            f"<p>Dataset: <b>{len(self._df):,} rows × {len(self._df.columns)} columns</b>",
        ]
        if self._target:
            html_parts.append(f" | Target: <code>{self._target}</code>")
        html_parts.append("</p>")

        # Overview figures
        ov._ensure_computed()
        for fig in ov._figures.values():
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        html_parts.append("</div>")
        display_html("\n".join(html_parts))

    def to_dict(self) -> dict[str, Any]:
        """Serialize all computed sub-modules (only those already accessed)."""
        result: dict[str, Any] = {
            "stage_name": self.stage_name,
            "duration_seconds": self.duration_seconds,
        }
        # Only serialize sub-modules that have been accessed (cached)
        for attr in ("overview", "univariate", "bivariate", "correlations",
                     "missing", "outliers", "tests", "clusters", "interactions", "target"):
            if attr in self.__dict__:  # cached_property stores in __dict__
                module = self.__dict__[attr]
                if module is not None:
                    result[attr] = module.to_dict()
        return result

    def _repr_html_(self) -> str:
        ov = self.overview
        ov._ensure_computed()
        n_rows, n_cols = len(self._df), len(self._df.columns)
        missing_total = self._df.isnull().sum().sum()
        missing_pct = 100 * missing_total / (n_rows * n_cols)

        fig_html = ""
        if ov._figures:
            first_fig = next(iter(ov._figures.values()))
            fig_html = first_fig.to_html(full_html=False, include_plotlyjs="cdn")

        return (
            f"<div style='font-family:system-ui;padding:12px;border-left:4px solid #4c78a8;"
            f"background:#f8f9fa'>"
            f"<b>EDAResult</b> — {n_rows:,} rows × {n_cols} cols"
            f" | Missing: {missing_pct:.1f}%"
            f"<br><small style='color:#666'>Access sub-modules: "
            f".overview · .correlations · .missing · .outliers · .tests · .clusters</small>"
            f"</div>"
            + fig_html
        )

    def __repr__(self) -> str:
        return (
            f"EDAResult(rows={len(self._df)}, cols={len(self._df.columns)}, "
            f"target={self._target!r})"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def explore(
    df: pd.DataFrame | str,
    target: str | None = None,
    task: str = "auto",
    **config_kwargs,
) -> EDAResult:
    """Run deep EDA on a DataFrame and return a lazy EDAResult.

    The result is returned instantly. Sub-module computation only happens
    when you access a property (e.g., ``eda.correlations.heatmap()``).

    Args:
        df: pandas DataFrame or path to a data file (CSV, Excel, Parquet, JSON).
        target: Name of the target column (optional). Enables target analysis
                and class balance checks when provided.
        task: 'classification', 'regression', or 'auto' (default).
        **config_kwargs: Override any DissectMLConfig field for this call only.

    Returns:
        EDAResult — lazy, exploration-ready result object.

    Example::

        eda = dml.explore(df, target="survived")
        eda.overview.show()
        eda.correlations.heatmap()
        eda.missing.patterns()
    """
    import pandas as _pd

    from dissectml._config import get_config
    from dissectml._io import read_data

    # Resolve config
    config = get_config()
    if config_kwargs:
        config = config.copy_with(**config_kwargs)

    # Load from file if path given
    if isinstance(df, str):
        df = read_data(df)
    elif not isinstance(df, _pd.DataFrame):
        raise TypeError(
            f"'df' must be a pandas DataFrame or file path string, got {type(df).__name__}"
        )

    # Basic validation
    from dissectml.core.validators import validate_dataframe
    validate_dataframe(df, target=target)

    start = time.perf_counter()
    result = EDAResult(df, target=target, task=task, config=config)
    result.duration_seconds = time.perf_counter() - start  # ~0s, lazy
    return result
