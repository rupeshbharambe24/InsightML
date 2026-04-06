"""Battle/leaderboard section builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from insightml.report.builder import AnalysisReport


def build_battle_section(report: "AnalysisReport") -> dict[str, Any] | None:
    """Build the model battle section."""
    if report.models is None:
        return None

    result = report.models
    parts: list[str] = []

    # Leaderboard table
    try:
        lb = result.leaderboard()
        parts.append("<h3>Leaderboard</h3>")
        parts.append(_df_to_html(lb, best_row=0))
    except Exception:
        pass

    # Metric bar chart via compare
    if report.compare is not None:
        try:
            fig = report.compare.metric_bar
            parts.append("<h3>Metric Comparison</h3>")
            parts.append(_fig_html(fig))
        except Exception:
            pass

    # Best model narrative
    try:
        from insightml.report.narrative import model_narrative

        best = result.best
        if best is not None:
            text = model_narrative(
                model_name=best.name,
                metrics=best.metrics,
                primary_metric=result.primary_metric,
                rank=1,
                n_models=len(result.successful),
            )
            parts.append(f'<p class="model-narrative">{text}</p>')
    except Exception:
        pass

    # Failed models
    failed = result.failed
    if failed:
        parts.append(f"<p><em>{len(failed)} model(s) failed during training.</em></p>")
        parts.append("<ul>")
        for f in failed:
            short_err = (f.error or "").split("\n")[0][:120]
            parts.append(f"<li><strong>{f.name}</strong>: {short_err}</li>")
        parts.append("</ul>")

    return {
        "title": "Model Battle",
        "anchor": "battle",
        "content": "\n".join(parts),
    }


def _fig_html(fig, height: int = 420) -> str:
    try:
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": False},
            default_height=height,
        )
    except Exception:
        return "<p><em>Chart unavailable.</em></p>"


def _df_to_html(df, best_row: int | None = None) -> str:
    try:
        html = df.to_html(classes="table", border=0, index=False)
        if best_row == 0:
            # Highlight first row as best
            html = html.replace("<tr>", '<tr class="best-row">', 2).replace(
                '<tr class="best-row">', "<tr>", 1
            )
        return '<div class="table-wrap">' + html + "</div>"
    except Exception:
        return ""
