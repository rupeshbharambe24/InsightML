# Changelog

All notable changes to DissectML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-06

### Added
- `dml.explore(df)` — Deep EDA with lazy evaluation
- Dataset overview: type detection, column profiles, memory stats
- Univariate analysis: distributions, KDE, descriptive stats
- Bivariate analysis: cross-type pair analysis
- Correlation analysis: unified matrix (Pearson/Spearman/Cramer's V/eta)
- Missing data intelligence: Little's MCAR test, MAR/MNAR classification
- Outlier detection: IQR, Z-score, Isolation Forest with consensus
- Statistical tests: normality, independence, variance, group comparison
- Cluster discovery: auto K-Means + DBSCAN with profiling
- Feature interactions: interaction strength, non-linearity detection
- Target analysis: class balance, distribution, feature-target relationships
- `dml.battle(df, target)` — parallel CV across 19 classifiers / 17 regressors
- EDA-informed preprocessing (KNN imputer, Robust scaler, OrdinalEncoder)
- ModelRegistry, MODEL_CATALOG, ModelTuner (quick/tuned/custom modes)
- `dml.analyze_intelligence(df, target)` — 4-pronged leakage detection, VIF, condition number
- Data readiness score 0–100 with grade (A–F) and penalty waterfall
- Composite feature importance ranking (MI + correlation + F-score)
- Algorithm recommendations engine (7 algorithm profiles)
- `ModelComparator` — McNemar test, corrected paired t-test, Pareto front, error analysis
- ROC/PR curves, confusion matrices, residual plots, actual vs predicted
- SHAP model comparison (TreeExplainer / LinearExplainer / KernelExplainer)
- `dml.analyze(df, target)` — full 5-stage pipeline, returns `AnalysisReport`
- `AnalysisReport.export(path)` — self-contained interactive HTML report
- `AnalysisReport.show()` — export + open in browser
- `dml.load_titanic()` / `dml.load_housing()` — built-in demo datasets
- `dml.to_pandas()` — Polars DataFrame / file path / dict / numpy array conversion
- `report/pdf_renderer.py` — optional PDF export via WeasyPrint
- Report sections module (`report/sections/`), Jinja2 templates, CSS/JS assets
- GitHub Actions CI/CD (`.github/workflows/ci.yml`, `release.yml`)
- MkDocs Material documentation site
- 4 example Jupyter notebooks (quickstart, deep EDA, model battle, full pipeline)
- 472 tests (453 passing, 3 skipped — tabulate/weasyprint optional deps)
