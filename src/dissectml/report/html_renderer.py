"""Self-contained HTML report renderer using an embedded Jinja2 template."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from dissectml.report.builder import AnalysisReport

# ---------------------------------------------------------------------------
# HTML template (embedded to avoid package-data complexity)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {
    --primary: #4c78a8;
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --border: #dee2e6;
    --text: #212529;
    --muted: #6c757d;
    --success: #54a24b;
    --warning: #f58518;
    --danger: #e45756;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }

  /* Layout */
  .layout { display: flex; min-height: 100vh; }
  .sidebar {
    width: 240px; min-width: 240px; background: #2d3748; color: #e2e8f0;
    padding: 1.5rem 1rem; position: sticky; top: 0; height: 100vh;
    overflow-y: auto; font-size: 0.875rem;
  }
  .sidebar h2 { color: #90cdf4; margin-bottom: 1rem; font-size: 1rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }
  .sidebar ul { list-style: none; }
  .sidebar li { margin: 0.25rem 0; }
  .sidebar a { color: #cbd5e0; text-decoration: none; display: block; padding: 0.25rem 0.5rem; border-radius: 4px; transition: background 0.15s; }
  .sidebar a:hover { background: #4a5568; color: #fff; }
  .main { flex: 1; padding: 2rem; max-width: 1100px; }

  /* Header */
  .report-header { background: var(--primary); color: white; padding: 1.5rem 2rem; border-radius: 8px; margin-bottom: 2rem; }
  .report-header h1 { font-size: 1.75rem; font-weight: 700; }
  .report-header p { opacity: 0.85; margin-top: 0.25rem; }
  .badge { display: inline-block; padding: 0.2em 0.6em; border-radius: 20px; font-size: 0.75rem; font-weight: 600; background: rgba(255,255,255,0.2); }

  /* Cards */
  .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 1.5rem; overflow: hidden; }
  .card-header { padding: 1rem 1.25rem; background: #f1f3f5; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 0.5rem; }
  .card-header h2 { font-size: 1.1rem; font-weight: 600; }
  .card-body { padding: 1.25rem; }

  /* Collapsible */
  details > summary { cursor: pointer; list-style: none; padding: 1rem 1.25rem; background: #f1f3f5; border-bottom: 1px solid var(--border); font-weight: 600; user-select: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::before { content: "▶ "; font-size: 0.7em; color: var(--muted); transition: transform 0.2s; display: inline-block; }
  details[open] > summary::before { content: "▼ "; }
  details > summary:hover { background: #e9ecef; }
  details .card-body { padding: 1.25rem; }

  /* Tables */
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 0.875rem; }
  th { background: #f1f3f5; font-weight: 600; padding: 0.5rem 0.75rem; text-align: left; border-bottom: 2px solid var(--border); white-space: nowrap; }
  td { padding: 0.45rem 0.75rem; border-bottom: 1px solid #f1f3f5; }
  tr:hover td { background: #f8f9fa; }
  .best-row td { font-weight: 700; color: var(--success); }

  /* Metrics grid */
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
  .metric-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; text-align: center; }
  .metric-card .value { font-size: 1.8rem; font-weight: 700; color: var(--primary); }
  .metric-card .label { font-size: 0.75rem; color: var(--muted); margin-top: 0.25rem; }

  /* Recommendations */
  .rec-list { list-style: none; }
  .rec-list li { padding: 0.5rem 0.75rem; margin: 0.4rem 0; border-left: 3px solid var(--primary); background: #f0f4ff; border-radius: 0 4px 4px 0; font-size: 0.9rem; }
  .rec-list li.warning { border-color: var(--warning); background: #fff8ec; }
  .rec-list li.danger { border-color: var(--danger); background: #fff5f5; }

  /* Readiness gauge */
  .grade-badge { display: inline-block; width: 48px; height: 48px; border-radius: 50%; font-size: 1.5rem; font-weight: 800; line-height: 48px; text-align: center; color: white; }
  .grade-A { background: var(--success); }
  .grade-B { background: #4c78a8; }
  .grade-C { background: var(--warning); }
  .grade-D { background: #f58518; }
  .grade-F { background: var(--danger); }

  /* Plotly containers */
  .plotly-container { width: 100%; min-height: 300px; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 768px) { .chart-grid { grid-template-columns: 1fr; } .sidebar { display: none; } }

  /* Footer */
  .report-footer { margin-top: 3rem; padding: 1rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.8rem; text-align: center; }
</style>
</head>
<body>
<div class="layout">

<!-- Sidebar TOC -->
<nav class="sidebar">
  <h2>Contents</h2>
  <ul>
    {% for section in sections %}
    <li><a href="#{{ section.id }}">{{ section.title }}</a></li>
    {% endfor %}
  </ul>
</nav>

<!-- Main content -->
<main class="main">

<!-- Header -->
<div class="report-header">
  <h1>{{ title }}</h1>
  <p>{{ subtitle }} &nbsp;
    <span class="badge">{{ task|title }}</span>
    <span class="badge">target: {{ target }}</span>
    <span class="badge">{{ n_samples|int }} samples</span>
    <span class="badge">{{ n_features }} features</span>
  </p>
</div>

{% for section in sections %}
<div class="card" id="{{ section.id }}">
  <details {% if section.open %}open{% endif %}>
    <summary>{{ section.emoji }} {{ section.title }}</summary>
    <div class="card-body">
      {{ section.body | safe }}
    </div>
  </details>
</div>
{% endfor %}

<div class="report-footer">
  Generated by <strong>InsightML</strong> v{{ version }} &mdash; {{ generated_at }}
</div>
</main>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def _fig_html(fig: go.Figure, height: int | None = None) -> str:
    """Embed a Plotly figure as an inline div."""
    if height:
        fig = fig.update_layout(height=height)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=None)


def _table_html(df, best_col: str | None = None, best_row_idx: int | None = None) -> str:
    """Render a DataFrame as an HTML table with optional best-row highlight."""
    if df is None or df.empty:
        return "<p><em>No data available.</em></p>"

    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        tr_class = "best-row" if i == best_row_idx else ""
        cells = "".join(
            f"<td>{v:.4f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
            for v in row
        )
        rows_html += f"<tr class='{tr_class}'>{cells}</tr>"

    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table></div>"


def _rec_list_html(recs: list[str], level: str = "info") -> str:
    """Render a list of recommendations."""
    if not recs:
        return "<p><em>None.</em></p>"
    items = "".join(f"<li>{r}</li>" for r in recs)
    return f"<ul class='rec-list'>{items}</ul>"


def _metric_cards_html(metrics: dict[str, float]) -> str:
    """Render a grid of metric cards."""
    cards = "".join(
        f"<div class='metric-card'>"
        f"<div class='value'>{v:.4f}</div>"
        f"<div class='label'>{k}</div>"
        f"</div>"
        for k, v in metrics.items()
    )
    return f"<div class='metrics-grid'>{cards}</div>"


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------


def render_html_report(report: AnalysisReport) -> str:  # noqa: F821
    """Render a self-contained HTML report from an AnalysisReport.

    Args:
        report: :class:`~dissectml.report.builder.AnalysisReport`

    Returns:
        HTML string (self-contained, embeds all figures).
    """
    from datetime import datetime

    from dissectml._version import __version__

    try:
        from jinja2 import Environment
        env = Environment(autoescape=False)
        template = env.from_string(_HTML_TEMPLATE)
    except ImportError as exc:
        raise ImportError("jinja2 is required for HTML reports: pip install dissectml") from exc

    sections = _build_sections(report)

    html = template.render(
        title=f"InsightML Report — {report.target}",
        subtitle="Automated ML analysis report",
        task=report.task or "unknown",
        target=report.target or "—",
        n_samples=report.n_samples,
        n_features=report.n_features,
        sections=sections,
        version=__version__,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    return html


def _build_sections(report: AnalysisReport) -> list[dict]:
    sections = []

    # --- Executive Summary ---
    sections.append(_section_executive_summary(report))

    # --- Data Readiness ---
    if report.intelligence is not None:
        sections.append(_section_readiness(report))

    # --- EDA ---
    if report.eda is not None:
        sections.append(_section_eda(report))

    # --- Intelligence ---
    if report.intelligence is not None:
        sections.append(_section_intelligence(report))

    # --- Model Battle ---
    if report.models is not None:
        sections.append(_section_battle(report))

    # --- Comparative Analysis ---
    if report.compare is not None:
        sections.append(_section_compare(report))

    return sections


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _section_executive_summary(report) -> dict:
    from dissectml.report.narrative import (
        data_recommendations,
        ensemble_recommendation,
        executive_summary,
    )

    leakage_cols = []
    high_vif_cols = []
    missing_pct = 0.0
    readiness_score = None
    readiness_grade = None
    n_leakage = 0
    pareto_models = []
    imbalance_severity = None

    if report.intelligence is not None:
        try:
            readiness_score = report.intelligence.readiness.score
            readiness_grade = report.intelligence.readiness.grade
            leakage_cols = [w["column"] for w in report.intelligence.leakage]
            n_leakage = len(leakage_cols)
            vif_df = report.intelligence.vif
            if not vif_df.empty:
                high_vif_cols = vif_df[vif_df["vif"] >= 10]["feature"].tolist()
        except Exception:
            pass

    if report.compare is not None:
        try:
            pareto_models = report.compare.pareto_models
        except Exception:
            pass

    best_model = report.models.best.name if (report.models and report.models.best) else None
    best_score = report.models.best.primary_metric if (report.models and report.models.best) else None
    primary_metric = report.models.primary_metric if report.models else ""

    summary_text = executive_summary(
        task=report.task or "classification",
        target=report.target or "target",
        n_samples=report.n_samples,
        n_features=report.n_features,
        best_model=best_model,
        best_score=best_score,
        primary_metric=primary_metric,
        readiness_score=readiness_score,
        readiness_grade=readiness_grade,
        n_leakage_warnings=n_leakage,
        pareto_models=pareto_models,
    )

    recs = data_recommendations(
        readiness_score=readiness_score or 100,
        leakage_columns=leakage_cols,
        high_vif_columns=high_vif_cols,
        missing_pct=missing_pct,
        imbalance_severity=imbalance_severity,
    )

    ensemble_rec = ""
    if report.compare is not None:
        try:
            candidates = report.compare.error_analysis.ensemble_candidates()
            ensemble_rec = f"<p>{ensemble_recommendation(candidates, best_model, pareto_models)}</p>"
        except Exception:
            pass

    body = (
        f"<p style='font-size:1.05em;line-height:1.8'>{summary_text}</p>"
        f"<h3 style='margin-top:1.5rem'>Recommendations</h3>"
        + _rec_list_html(recs)
        + ensemble_rec
    )

    return {"id": "summary", "title": "Executive Summary", "emoji": "📋", "body": body, "open": True}


def _section_readiness(report) -> dict:
    try:
        r = report.intelligence.readiness
        grade_color = {"A": "grade-A", "B": "grade-B", "C": "grade-C", "D": "grade-D", "F": "grade-F"}
        gclass = grade_color.get(r.grade, "grade-F")

        breakdown_rows = "".join(
            f"<tr><td>{cat.replace('_',' ').title()}</td>"
            f"<td style='color:{'#e45756' if v['delta']<0 else '#54a24b'};font-weight:600'>"
            f"{v['delta']:+.1f}</td>"
            f"<td>{v.get('description','')}</td></tr>"
            for cat, v in r.breakdown.items()
        )
        breakdown_table = (
            "<div class='table-wrap'><table><thead><tr>"
            "<th>Category</th><th>Delta</th><th>Details</th>"
            "</tr></thead><tbody>" + breakdown_rows + "</tbody></table></div>"
        )

        gauge_html = _fig_html(r.gauge_figure(), height=280)
        waterfall_html = _fig_html(r.waterfall_figure(), height=380)

        body = (
            f"<div style='display:flex;align-items:center;gap:1.5rem;margin-bottom:1.5rem'>"
            f"<span class='grade-badge {gclass}'>{r.grade}</span>"
            f"<div><strong style='font-size:2rem'>{r.score:.0f}</strong><span style='color:var(--muted)'>/100</span>"
            f"<div style='color:var(--muted);font-size:0.85rem'>{r.n_samples:,} samples · {r.n_features} features</div>"
            f"</div></div>"
            f"<div class='chart-grid'>{gauge_html}{waterfall_html}</div>"
            f"<h4 style='margin:1rem 0 0.5rem'>Score Breakdown</h4>{breakdown_table}"
        )
    except Exception as e:
        body = f"<p>Readiness analysis unavailable: {e}</p>"

    return {"id": "readiness", "title": "Data Readiness", "emoji": "🎯", "body": body, "open": True}


def _section_eda(report) -> dict:
    body_parts = []
    try:
        overview = report.eda.overview
        summary_str = overview.summary()
        body_parts.append(f"<p>{summary_str}</p>")
        for _name, fig in list(overview.figures.items())[:4]:
            body_parts.append(_fig_html(fig))
    except Exception:
        body_parts.append("<p>EDA overview unavailable.</p>")

    return {
        "id": "eda",
        "title": "Exploratory Data Analysis",
        "emoji": "🔍",
        "body": "\n".join(body_parts),
        "open": False,
    }


def _section_intelligence(report) -> dict:
    body_parts = []
    intel = report.intelligence

    # Leakage
    try:
        leakage = intel.leakage
        if leakage:
            rows = "".join(
                f"<tr><td>{w['column']}</td><td>{w['method']}</td>"
                f"<td>{w['score']:.4f}</td><td>{w['severity']}</td>"
                f"<td>{w['explanation']}</td></tr>"
                for w in leakage[:10]
            )
            body_parts.append(
                "<h4>⚠ Leakage Warnings</h4>"
                "<div class='table-wrap'><table><thead><tr>"
                "<th>Column</th><th>Method</th><th>Score</th><th>Severity</th><th>Explanation</th>"
                "</tr></thead><tbody>" + rows + "</tbody></table></div>"
            )
        else:
            body_parts.append("<p style='color:var(--success)'>✓ No leakage warnings detected.</p>")
    except Exception:
        pass

    # VIF
    try:
        vif_df = intel.vif
        if not vif_df.empty:
            high_vif = vif_df[vif_df["severity"] == "high"]
            if not high_vif.empty:
                body_parts.append(
                    "<h4>Multicollinearity (VIF≥10)</h4>"
                    + _table_html(high_vif)
                )
    except Exception:
        pass

    # Feature importance
    try:
        fi = intel.feature_importance
        if not fi.empty:
            body_parts.append("<h4>Feature Importance (composite ranking, top 10)</h4>")
            body_parts.append(_table_html(fi.head(10)))
    except Exception:
        pass

    # Recommendations
    try:
        recs_html = intel.recommendations._repr_html_()
        body_parts.append(f"<h4>Algorithm Recommendations</h4>{recs_html}")
    except Exception:
        pass

    return {
        "id": "intelligence",
        "title": "Pre-Model Intelligence",
        "emoji": "🧠",
        "body": "\n".join(body_parts) or "<p>Intelligence analysis unavailable.</p>",
        "open": False,
    }


def _section_battle(report) -> dict:
    body_parts = []
    result = report.models

    try:
        best = result.best
        if best:
            metrics_html = _metric_cards_html(
                dict(list(best.metrics.items())[:4])
            )
            body_parts.append(
                f"<h4>Best Model: {best.name}</h4>{metrics_html}"
            )
    except Exception:
        pass

    try:
        lb = result.leaderboard()
        body_parts.append(f"<h4>Leaderboard ({len(lb)} models)</h4>")
        body_parts.append(_table_html(lb, best_row_idx=0))
    except Exception:
        body_parts.append("<p>Leaderboard unavailable.</p>")

    if report.compare is not None:
        try:
            body_parts.append("<h4>Metric Comparison</h4>")
            body_parts.append(_fig_html(report.compare.metric_bar))
        except Exception:
            pass

    return {
        "id": "battle",
        "title": "Model Battle",
        "emoji": "⚔️",
        "body": "\n".join(body_parts),
        "open": True,
    }


def _section_compare(report) -> dict:
    body_parts = []
    comp = report.compare

    # Pareto
    try:
        body_parts.append("<h4>Pareto Front</h4>")
        body_parts.append(_fig_html(comp.pareto, height=400))
        body_parts.append(f"<p>Pareto-optimal: <strong>{', '.join(comp.pareto_models)}</strong></p>")
    except Exception:
        pass

    # ROC / residuals
    task = report.task
    try:
        if task == "classification" and comp.roc_curves:
            body_parts.append("<h4>ROC Curves</h4>")
            body_parts.append(_fig_html(comp.roc_curves))
        if task == "regression" and comp.residual_plots:
            body_parts.append("<h4>Residual Plots</h4>")
            body_parts.append(_fig_html(comp.residual_plots))
    except Exception:
        pass

    # Confusion matrices
    try:
        if task == "classification" and comp.confusion_matrices:
            body_parts.append("<h4>Confusion Matrices</h4>")
            body_parts.append(_fig_html(comp.confusion_matrices))
    except Exception:
        pass

    # Significance
    try:
        sig = comp.significance
        if sig:
            key = "mcnemar" if "mcnemar" in sig else "ttest"
            if key in sig and "figure" in sig[key]:
                body_parts.append("<h4>Statistical Significance</h4>")
                body_parts.append(_fig_html(sig[key]["figure"]))
    except Exception:
        pass

    # Error analysis
    try:
        ea = comp.error_analysis
        if not ea.disagreement.empty:
            body_parts.append("<h4>Model Disagreement</h4>")
            body_parts.append(_fig_html(ea.disagreement_figure()))
    except Exception:
        pass

    return {
        "id": "compare",
        "title": "Comparative Analysis",
        "emoji": "📊",
        "body": "\n".join(body_parts) or "<p>Comparison analysis unavailable.</p>",
        "open": False,
    }
