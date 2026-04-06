"""BaseAnalysisModule — ABC for all EDA sub-modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from dissectml._config import InsightMLConfig, get_config
from dissectml.viz.display import display_html


class BaseAnalysisModule(ABC):
    """Abstract base class for every EDA sub-module.

    Sub-modules use lazy computation: results are only computed when first
    accessed (triggered by ``_ensure_computed()``).

    Subclasses MUST implement:
        - ``_compute()``     — performs all computation, stores in ``self._results``
        - ``_build_figures()`` — builds Plotly figures from computed results
        - ``summary()``      — returns a one-paragraph text summary

    Subclasses SHOULD NOT call ``_compute()`` directly; use ``_ensure_computed()``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target: str | None = None,
        config: InsightMLConfig | None = None,
    ) -> None:
        self._df = df
        self._target = target
        self._config = config or get_config()
        self._computed = False
        self._results: dict[str, Any] = {}
        self._figures: dict[str, go.Figure] = {}
        self._warnings: list[str] = []

    # ------------------------------------------------------------------
    # Abstract interface (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute(self) -> None:
        """Perform all computations and store results in ``self._results``."""
        ...

    @abstractmethod
    def _build_figures(self) -> dict[str, go.Figure]:
        """Build Plotly figures from ``self._results``.

        Returns:
            Dict mapping figure name -> go.Figure.
        """
        ...

    @abstractmethod
    def summary(self) -> str:
        """Return a concise one-paragraph text summary of key findings."""
        ...

    # ------------------------------------------------------------------
    # Concrete shared behaviour
    # ------------------------------------------------------------------

    def _ensure_computed(self) -> None:
        """Trigger computation if not already done. Thread-safe via flag check."""
        if not self._computed:
            self._compute()
            self._figures = self._build_figures()
            self._computed = True

    def show(self, kind: str | None = None) -> None:
        """Display chart(s) in Jupyter or browser.

        Args:
            kind: Specific figure name to show. If None, shows all figures.
        """
        self._ensure_computed()
        if kind is not None:
            fig = self._figures.get(kind)
            if fig is None:
                available = list(self._figures.keys())
                raise KeyError(f"Figure '{kind}' not found. Available: {available}")
            display_html(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        else:
            for fig in self._figures.values():
                display_html(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    def plot(self, kind: str | None = None) -> go.Figure | dict[str, go.Figure]:
        """Return Plotly figure(s) without displaying.

        Args:
            kind: Specific figure name. If None, returns all figures as dict.

        Returns:
            A single go.Figure if kind is specified, else dict of all figures.
        """
        self._ensure_computed()
        if kind is not None:
            fig = self._figures.get(kind)
            if fig is None:
                raise KeyError(
                    f"Figure '{kind}' not found. Available: {list(self._figures.keys())}"
                )
            return fig
        return dict(self._figures)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of computed results."""
        self._ensure_computed()
        return {k: _make_serializable(v) for k, v in self._results.items()}

    def to_dataframe(self) -> pd.DataFrame:
        """Return computed results as a DataFrame. Override for richer output."""
        self._ensure_computed()
        try:
            return pd.DataFrame(self._results)
        except ValueError:
            return pd.DataFrame([self._results])

    def _repr_html_(self) -> str:
        """Jupyter rich display — shows summary + first figure."""
        self._ensure_computed()
        parts = [
            f"<div style='font-family:system-ui;padding:12px;"
            f"border-left:4px solid #4c78a8;background:#f8f9fa'>"
            f"<b>{self.__class__.__name__}</b><br>"
            f"<p style='margin:4px 0;color:#444'>{self.summary()}</p>"
        ]
        if self._warnings:
            for w in self._warnings:
                parts.append(
                    f"<p style='color:#e45756;font-size:0.9em'>⚠ {w}</p>"
                )
        parts.append("</div>")

        # Embed first figure (Plotly CDN loaded once per notebook)
        if self._figures:
            first_fig = next(iter(self._figures.values()))
            parts.append(
                first_fig.to_html(full_html=False, include_plotlyjs="cdn")
            )
        return "\n".join(parts)

    def _warn(self, message: str) -> None:
        """Record a warning (shown in _repr_html_ and summary)."""
        self._warnings.append(message)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj
