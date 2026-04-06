"""AnalysisReport — top-level result object combining all pipeline stages."""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AnalysisReport:
    """Unified report combining EDA, Intelligence, Battle, and Compare stages.

    Produced by ``iml.analyze()``.

    Attributes:
        task: "classification" or "regression".
        target: Target column name.
        n_samples: Number of training rows.
        n_features: Number of feature columns.
        eda: EDAResult (Stage 1).
        intelligence: IntelligenceResult (Stage 2).
        models: BattleResult (Stage 3).
        compare: ModelComparator (Stage 4).
    """

    task: str
    target: str
    n_samples: int = 0
    n_features: int = 0
    eda: Any = None             # EDAResult
    intelligence: Any = None    # IntelligenceResult
    models: Any = None          # BattleResult
    compare: Any = None         # ModelComparator

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, path: str = "insightml_report.html") -> str:
        """Render and save a self-contained HTML report.

        Args:
            path: Output file path (default: ``insightml_report.html``).

        Returns:
            Absolute path to the written file.
        """
        from insightml.report.html_renderer import render_html_report
        html = render_html_report(self)
        out = Path(path).resolve()
        out.write_text(html, encoding="utf-8")
        return str(out)

    def show(self, path: str | None = None) -> str:
        """Export to HTML and open in the default browser.

        Args:
            path: Optional path to save to (default: temp file).

        Returns:
            Path of the rendered HTML file.
        """
        import tempfile
        if path is None:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, prefix="insightml_"
            )
            path = tmp.name
            tmp.close()
        out = self.export(path)
        webbrowser.open(f"file://{out}")
        return out

    # ------------------------------------------------------------------
    # Quick accessors
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a plain-text summary of the full analysis."""
        parts: list[str] = [
            "=== InsightML Analysis Report ===",
            f"Task: {self.task}  |  Target: {self.target}",
            f"Dataset: {self.n_samples:,} samples × {self.n_features} features",
        ]

        if self.intelligence is not None:
            try:
                r = self.intelligence.readiness
                parts.append(f"Data Readiness: {r.score:.0f}/100 (Grade {r.grade})")
            except Exception:
                pass
            try:
                n_leakage = len(self.intelligence.leakage)
                if n_leakage:
                    parts.append(f"Leakage Warnings: {n_leakage}")
            except Exception:
                pass

        if self.models is not None:
            best = self.models.best
            if best:
                parts.append(
                    f"Best Model: {best.name} "
                    f"({self.models.primary_metric}={best.primary_metric:.4f})"
                )

        if self.compare is not None:
            try:
                parts.append(f"Pareto-optimal: {self.compare.pareto_models}")
            except Exception:
                pass

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _repr_html_(self) -> str:
        from insightml.report.html_renderer import render_html_report
        return render_html_report(self)

    def __repr__(self) -> str:
        best_str = ""
        if self.models and self.models.best:
            best_str = f", best={self.models.best.name}"
        return (
            f"AnalysisReport(task={self.task!r}, target={self.target!r}"
            f"{best_str})"
        )
