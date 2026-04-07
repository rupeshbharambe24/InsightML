"""InsightPipeline — orchestrates all 5 stages end-to-end."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dissectml._config import DissectMLConfig, get_config
from dissectml.core.base import PipelineContext
from dissectml.core.data_container import DataContainer


class InsightPipeline:
    """Runs all pipeline stages in sequence, passing context between them.

    Used internally by ``dml.analyze()``. Individual stages can also be
    accessed directly via ``dml.explore()`` or ``dml.battle()``.

    Stages (v0.4 full pipeline):
        1. EDA         — Deep exploratory data analysis
        2. Intelligence — Pre-model intelligence (leakage, readiness, recommendations)
        3. Battle       — Multi-model parallel training + CV scoring
        4. Compare      — Statistical comparison, error analysis, SHAP, Pareto
        5. Report       — HTML report generation
    """

    def __init__(self, config: DissectMLConfig | None = None) -> None:
        self.config = config or get_config()
        self.context = PipelineContext(config=self.config)

    def run_eda(self, container: DataContainer) -> Any:
        """Run Stage 1: Deep EDA."""
        from dissectml.eda import EDAStage
        stage = EDAStage()
        result = stage.timed_run(container, self.context)
        self.context.eda_result = result
        return result

    def run(
        self,
        data: str | pd.DataFrame,
        target: str | None = None,
        task: str = "auto",
    ) -> Any:
        """Run the complete pipeline and return an AnalysisReport.

        Available in v0.4+. Currently raises NotImplementedError for stages 2-5.
        """
        container = DataContainer.from_input(data, target=target, task=task, config=self.config)

        # Stage 1: EDA (v0.1+)
        self.run_eda(container)

        # Stages 2-5: to be implemented in v0.2-v0.4
        raise NotImplementedError(
            "Full pipeline (dml.analyze) is available in v0.4+. "
            "Use dml.explore(df) for EDA."
        )
