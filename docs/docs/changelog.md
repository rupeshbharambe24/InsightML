# Changelog

## v0.5.0 (unreleased)

### Added
- `_compat.py` — `to_pandas()` conversion layer (accepts Polars DataFrames, file paths, dicts)
- `datasets/` — built-in `load_titanic()` and `load_housing()` demo datasets
- `report/pdf_renderer.py` — optional PDF export via WeasyPrint (requires `dissectml[report]`)
- GitHub Actions CI/CD workflows (`.github/workflows/ci.yml`, `release.yml`)
- MkDocs Material documentation site

## v0.4.0

### Added
- `compare/` — `ModelComparator`, McNemar test, corrected paired t-test, Pareto front, error analysis
- `report/` — `AnalysisReport`, `AnalysisReport.export()`, self-contained HTML renderer
- `dml.analyze()` — full 5-stage pipeline entry point

## v0.3.0

### Added
- `intelligence/` — leakage detection, VIF, feature importance, readiness score, algorithm recommendations
- `dml.analyze_intelligence()` entry point

## v0.2.0

### Added
- `battle/` — `ModelRegistry`, `BattleRunner`, `ModelTuner`, EDA-informed preprocessing
- `dml.battle()` entry point
- 19 classifiers + 17 regressors in default catalogue

## v0.1.0

### Added
- `eda/` — `EDAResult` with lazy sub-modules: overview, correlations, missing, outliers,
  statistical tests, clusters, interactions, target analysis
- `dml.explore()` entry point
- `DissectMLConfig`, `config_context()`, `set_config()`
