# HTML Reports

InsightML generates self-contained interactive HTML reports — a single file with
embedded CSS, Plotly charts (via CDN), collapsible sections, and a sticky sidebar TOC.

## Generating a Report

```python
report = iml.analyze(df, target="survived")

# Export to file
report.export("report.html")   # Returns absolute path

# Open in browser immediately
report.show("report.html")

# Get HTML string (e.g. to embed in a web app)
from insightml.report.html_renderer import render_html_report
html = render_html_report(report)
```

## Jupyter Integration

Every result object implements `_repr_html_()`, so they render automatically in notebooks:

```python
report           # Renders full report inline
report.eda       # Renders EDA sub-report
report.models    # Renders leaderboard table
```

## Report Sections

1. **Executive Summary** — dataset size, task, best model, key findings
2. **Data Readiness** — score gauge + penalty waterfall
3. **EDA Findings** — correlation heatmap, missing patterns, outlier chart
4. **Intelligence** — leakage warnings, feature importance, algorithm recommendations
5. **Model Battle** — leaderboard table, metric bar chart, training time
6. **Comparative Analysis** — Pareto front, significance matrix, error analysis

## PDF Export (Optional)

Requires WeasyPrint:

```bash
pip install insightml[report]
```

```python
from insightml.report.pdf_renderer import export_pdf

export_pdf(report, "report.pdf")
```

## Customisation

```python
iml.set_config(plotly_template="plotly_dark")
```

Available Plotly templates: `plotly`, `plotly_white`, `plotly_dark`,
`ggplot2`, `seaborn`, `simple_white`.
