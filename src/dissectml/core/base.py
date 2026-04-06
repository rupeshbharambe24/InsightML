"""Base classes for InsightML's pipeline architecture."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from dissectml.viz.display import display_html

if TYPE_CHECKING:
    from dissectml._config import InsightMLConfig
    from dissectml.core.data_container import DataContainer
    from dissectml.core.progress import ProgressTracker


# ---------------------------------------------------------------------------
# StageResult — base class for all stage outputs
# ---------------------------------------------------------------------------

class StageResult:
    """Base class for the output of every pipeline stage.

    Provides Jupyter display, `.show()`, serialization, and timing.
    """

    def __init__(
        self,
        stage_name: str,
        duration_seconds: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.duration_seconds = duration_seconds
        self.metadata: dict[str, Any] = metadata or {}

    # --- Jupyter/display integration ---

    def _repr_html_(self) -> str:
        """Called automatically by Jupyter to render this result."""
        return (
            f"<div style='font-family:monospace;padding:8px;background:#f8f9fa;"
            f"border-left:4px solid #4c78a8'>"
            f"<b>{self.stage_name}</b> result "
            f"<span style='color:#666'>(computed in {self.duration_seconds:.2f}s)</span>"
            f"</div>"
        )

    def show(self) -> None:
        """Display this result — adapts to Jupyter, VS Code, or terminal."""
        html = self._repr_html_()
        display_html(html)

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "stage_name": self.stage_name,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a summary DataFrame. Override in subclasses for richer output."""
        return pd.DataFrame([self.to_dict()])


# ---------------------------------------------------------------------------
# BaseStage — ABC for every pipeline stage
# ---------------------------------------------------------------------------

class BaseStage(ABC):
    """Abstract base class for a pipeline stage.

    Every stage (EDA, Intelligence, Battle, Compare, Report) subclasses this.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name, e.g., 'EDA'."""
        ...

    @abstractmethod
    def run(
        self,
        container: DataContainer,
        context: PipelineContext,
    ) -> StageResult:
        """Execute the stage and return its result.

        Args:
            container: Immutable data wrapper (df, target, task, schema, sample).
            context: Shared mutable pipeline state (stores results between stages).

        Returns:
            StageResult subclass instance.
        """
        ...

    def timed_run(
        self,
        container: DataContainer,
        context: PipelineContext,
    ) -> StageResult:
        """Wrap `run()` with timing and progress reporting."""
        start = time.perf_counter()
        result = self.run(container, context)
        result.duration_seconds = time.perf_counter() - start
        return result


# ---------------------------------------------------------------------------
# PipelineContext — shared mutable state flowing between stages
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """Shared mutable state passed between pipeline stages.

    Stages read results from earlier stages and write their own results here.
    """
    eda_result: Any | None = None           # EDAResult (typed as Any to avoid circular import)
    intelligence_result: Any | None = None  # IntelligenceResult
    battle_result: Any | None = None        # BattleResult
    compare_result: Any | None = None       # CompareResult
    config: InsightMLConfig = field(
        default_factory=lambda: _default_config()
    )
    progress: ProgressTracker | None = None

    def __post_init__(self) -> None:
        if self.progress is None:
            from dissectml.core.progress import ProgressTracker
            self.progress = ProgressTracker()


def _default_config() -> InsightMLConfig:
    from dissectml._config import get_config
    return get_config()
