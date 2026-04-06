# InsightML

**The missing middle layer between EDA and AutoML.**

InsightML (`insightml` on PyPI) gives data scientists a single, unified pipeline from raw data to publication-ready insights — in as few as 3 function calls.

```python
import insightml as iml

df = iml.load_titanic()
report = iml.analyze(df, target="survived")
report.export("report.html")
```

---

## Why InsightML?

| Without InsightML | With InsightML |
|---|---|
| YData Profiling + PyCaret + SHAP + matplotlib | One library |
| 30+ lines of boilerplate | 3 function calls |
| No cross-tool insights | Pipeline stages inform each other |
| Static outputs | Interactive Plotly charts |

---

## Key Features

- **Deep EDA** — unified correlation matrix, MCAR/MAR/MNAR missing analysis, auto clustering, interaction detection
- **Pre-model Intelligence** — 4-pronged leakage detection, VIF, data readiness score (0–100), algorithm recommendations
- **Model Battle** — parallel CV across 19 classifiers / 17 regressors with EDA-informed preprocessing
- **Comparative Analysis** — McNemar test, corrected paired t-test, Pareto front, cross-model error analysis
- **HTML Report** — self-contained interactive report with Plotly charts, collapsible sections, sidebar TOC

---

## Installation

```bash
pip install insightml
```

With optional extras:
```bash
pip install insightml[boost]    # XGBoost, LightGBM, CatBoost
pip install insightml[explain]  # SHAP explanations
pip install insightml[report]   # PDF export (WeasyPrint)
pip install insightml[full]     # Everything
```

---

## Quick Start

See [Getting Started](getting-started.md) for a full walkthrough.
