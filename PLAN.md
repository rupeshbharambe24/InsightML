# InsightML - Final Implementation Plan

> **This is the single source of truth for building InsightML.**
> All decisions, architecture, algorithms, and implementation details are here.
> Reference: `InsightML.html` (market research), this file (how to build it).

---

## 0. Decisions Log

| Decision | Choice | Rationale |
|---|---|---|
| Package name (PyPI) | `dissectml` | Clean, readable, matches "InsightML" |
| Import name | `dissectml` | `import dissectml as iml` |
| Python environment | Dedicated `.venv` in project | Shared D:\commonenv has NumPy/SciPy conflict |
| Python version | `>=3.10` | Needed for `match`, `X \| Y` union types |
| Build system | hatchling (src-layout) | Modern, clean, no setup.py needed |
| v0.1 scope | EDA-only (`iml.explore()`) | Ship fast, validate EDA engine first |
| Deep Learning | sklearn MLP only | MLPClassifier/MLPRegressor, no PyTorch/TF |
| Core deps | Keep pydantic, statsmodels, rich | All useful, common in DS environments |
| File formats | CSV, Excel, Parquet, JSON | All four via pandas readers |
| Plugins | Deferred to v0.5+ | No plugin system in initial releases |
| Git | Init now, remote later | Local repo immediately |
| Visualization | Plotly only, no matplotlib | Interactive HTML natively, report-ready |
| Polars | Deferred to v0.5 | pandas-only initially |

---

## 1. Environment Setup

```bash
# 1. Create dedicated venv
cd D:\Projects\insightML
python -m venv .venv

# 2. Activate
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Unix

# 3. Install in editable mode with dev deps
pip install -e ".[dev]"
```

The `.venv` isolates InsightML from the broken `D:\commonenv` (NumPy 2.4.4 vs SciPy 1.11.4 conflict). All dependency versions are pinned via `pyproject.toml`.

---

## 2. Project Structure

```
D:\Projects\insightML\
├── pyproject.toml                     # Build config, deps, tool config
├── README.md
├── LICENSE                            # MIT
├── CHANGELOG.md
├── PLAN.md                            # This file
├── InsightML.html                     # Market research (existing)
├── .gitignore
├── .pre-commit-config.yaml
│
├── src/
│   └── dissectml/                     # The installable package
│       ├── __init__.py                # Public API: analyze(), explore(), battle()
│       ├── _version.py                # __version__ = "0.1.0"
│       ├── _types.py                  # Enums, TypedDicts, type aliases
│       ├── _config.py                 # InsightMLConfig dataclass + get/set/context
│       ├── _lazy.py                   # Optional dependency guard
│       ├── _sampling.py               # Smart sampling (stratified, temporal, random)
│       ├── _io.py                     # File loading: CSV, Excel, Parquet, JSON
│       ├── exceptions.py              # InsightMLError hierarchy
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base.py                # BaseStage ABC, StageResult, PipelineContext
│       │   ├── pipeline.py            # InsightPipeline orchestrator (stages 1-5)
│       │   ├── data_container.py      # DataContainer: wraps df + metadata + schema
│       │   ├── validators.py          # Input validation, task inference
│       │   └── progress.py            # ProgressTracker (tqdm/rich/notebook auto-detect)
│       │
│       ├── eda/
│       │   ├── __init__.py            # explore() entry point + re-exports
│       │   ├── _base.py               # BaseAnalysisModule ABC
│       │   ├── result.py              # EDAResult: lazy @cached_property orchestrator
│       │   ├── overview.py            # Shape, dtypes, memory, type detection
│       │   ├── univariate.py          # Distributions, descriptive stats, KDE
│       │   ├── bivariate.py           # Cross-type analysis: scatter, ANOVA, chi-sq
│       │   ├── correlations.py        # Pearson/Spearman/Cramer's V/unified matrix
│       │   ├── missing.py             # Missing patterns, Little's MCAR, MAR/MNAR
│       │   ├── outliers.py            # IQR, Z-score, Isolation Forest, consensus
│       │   ├── statistical_tests.py   # Normality, independence, variance, group comparison
│       │   ├── clusters.py            # Auto K-Means/DBSCAN, profiles, PCA/t-SNE viz
│       │   ├── interactions.py        # Interaction strength, non-linearity detection
│       │   └── target_analysis.py     # Target-specific: balance, distribution, vs features
│       │
│       ├── intelligence/
│       │   ├── __init__.py
│       │   ├── leakage.py             # Target leakage: high-corr, MI, temporal, derived
│       │   ├── multicollinearity.py   # VIF, condition number, eigenvalue analysis
│       │   ├── feature_importance.py  # MI, correlation, F-scores, composite ranking
│       │   ├── readiness.py           # Data readiness score 0-100 with breakdown
│       │   └── recommendations.py     # Algorithm recommender (rules engine)
│       │
│       ├── battle/
│       │   ├── __init__.py            # battle() entry point
│       │   ├── catalog.py             # MODEL_CATALOG: 19 classifiers + 17 regressors
│       │   ├── registry.py            # ModelRegistry: register/unregister/filter
│       │   ├── preprocessing.py       # EDA-informed ColumnTransformer builder
│       │   ├── runner.py              # BattleRunner: parallel CV training + timing
│       │   ├── tuner.py               # ModelTuner: RandomizedSearchCV on top-N
│       │   ├── param_grids.py         # Default hyperparameter search spaces
│       │   └── result.py              # BattleResult, ModelScore dataclasses
│       │
│       ├── compare/
│       │   ├── __init__.py
│       │   ├── comparator.py          # ModelComparator facade
│       │   ├── metrics_table.py       # ComparisonTable with styled Jupyter display
│       │   ├── curves.py              # ROC, PR, calibration, learning curves
│       │   ├── error_analysis.py      # Cross-model disagreement, complementarity
│       │   ├── significance.py        # McNemar, corrected paired t-test, Wilcoxon
│       │   ├── pareto.py              # Accuracy vs speed Pareto front
│       │   └── shap_compare.py        # Side-by-side SHAP across top-N models
│       │
│       ├── report/
│       │   ├── __init__.py
│       │   ├── builder.py             # ReportBuilder: assembles AnalysisReport
│       │   ├── narrative.py           # Template-driven natural language summaries
│       │   ├── html_renderer.py       # Jinja2 -> self-contained HTML with Plotly
│       │   ├── pdf_renderer.py        # Optional: weasyprint HTML-to-PDF
│       │   ├── sections/
│       │   │   ├── eda_section.py
│       │   │   ├── intelligence_section.py
│       │   │   ├── battle_section.py
│       │   │   ├── compare_section.py
│       │   │   └── summary_section.py
│       │   ├── templates/
│       │   │   ├── base.html.j2       # Master layout (Plotly CDN, responsive)
│       │   │   ├── section.html.j2    # Section wrapper
│       │   │   └── components/
│       │   │       ├── table.html.j2
│       │   │       ├── chart_container.html.j2
│       │   │       ├── collapsible.html.j2
│       │   │       └── toc.html.j2
│       │   └── assets/
│       │       ├── style.css          # Inlined into final HTML
│       │       └── script.js          # Collapse/TOC logic, inlined
│       │
│       ├── viz/
│       │   ├── __init__.py
│       │   ├── theme.py               # InsightML Plotly theme/template
│       │   ├── charts.py              # Chart factory functions
│       │   └── display.py             # Jupyter _repr_html_ mixin, env detection
│       │
│       └── datasets/
│           ├── __init__.py            # load_titanic(), load_housing()
│           └── data/
│               ├── titanic.csv
│               └── housing.csv
│
├── tests/
│   ├── conftest.py                    # Shared fixtures (synthetic DataFrames)
│   ├── test_api.py                    # Top-level: analyze(), explore(), battle()
│   ├── test_io.py                     # File loading: CSV, Excel, Parquet, JSON
│   ├── eda/
│   │   ├── test_overview.py
│   │   ├── test_univariate.py
│   │   ├── test_correlations.py
│   │   ├── test_missing.py
│   │   ├── test_outliers.py
│   │   ├── test_statistical_tests.py
│   │   ├── test_clusters.py
│   │   └── test_target_analysis.py
│   ├── intelligence/
│   │   ├── test_leakage.py
│   │   ├── test_readiness.py
│   │   └── test_recommendations.py
│   ├── battle/
│   │   ├── test_registry.py
│   │   ├── test_runner.py
│   │   └── test_preprocessing.py
│   ├── compare/
│   │   ├── test_significance.py
│   │   ├── test_error_analysis.py
│   │   └── test_pareto.py
│   ├── report/
│   │   └── test_html_renderer.py
│   └── integration/
│       ├── test_full_classification.py
│       └── test_full_regression.py
│
├── docs/
│   ├── mkdocs.yml
│   └── docs/
│       ├── index.md
│       ├── getting-started.md
│       └── user-guide/
│           ├── eda.md
│           ├── model-battle.md
│           ├── comparative-analysis.md
│           └── reports.md
│
└── examples/
    ├── 01_quickstart.ipynb
    ├── 02_deep_eda.ipynb
    ├── 03_model_battle.ipynb
    └── 04_full_pipeline.ipynb
```

---

## 3. Architecture & Data Flow

### 3.1 Dream API

```python
import dissectml as iml

# === Full pipeline (3 lines) ===
report = iml.analyze(data="dataset.csv", target="price", task="regression")
report.eda.show()             # Interactive EDA dashboard
report.models.compare()       # Side-by-side model comparison
report.insights.summary()     # Natural language key findings
report.export("report.html")  # Full interactive report

# === Individual stages ===
eda = iml.explore(df)                     # Deep EDA only
models = iml.battle(df, target="price")   # Model comparison only

# === Granular access ===
eda.outliers.plot()             # Outlier analysis with IQR + Z-score + IF
eda.correlations.heatmap()      # Multi-type correlation matrix
eda.tests.normality()           # Shapiro-Wilk for all numerical cols
eda.missing.patterns()          # MCAR/MAR/MNAR analysis
eda.target.balance()            # Class balance / target distribution

models.error_analysis()         # Where models disagree and why
models.significance_test()      # McNemar / paired t-test between top models
models.pareto()                 # Accuracy vs. training time Pareto front
models.shap_compare()           # SHAP importance across all models
```

### 3.2 Pipeline Data Flow

```
User Input (CSV/Excel/Parquet/JSON/DataFrame)
     |
     v
DataContainer.from_input()  -->  DataContainer {df, target, task, schema, sample}
     |
     v
[Stage 1: EDA]  --->  EDAResult  --->  stored in PipelineContext
     |
     v  (reads: missing patterns, outlier severity, distributions, types)
[Stage 2: Intelligence]  --->  IntelligenceResult {leakage, readiness, recommendations}
     |
     v  (reads: algo recommendations -> prioritize models; data profile -> preprocessing)
[Stage 3: Battle]  --->  BattleResult {trained models, CV scores, OOF predictions, timing}
     |
     v  (reads: fitted estimators + predictions for all comparisons)
[Stage 4: Compare]  --->  CompareResult {significance, error analysis, SHAP, Pareto}
     |
     v  (reads: ALL stage results, assembles into unified report)
[Stage 5: Report]  --->  AnalysisReport {.eda, .models, .compare, .insights, .export()}
```

### 3.3 Core Base Classes

#### BaseStage (ABC) — `core/base.py`
Every pipeline stage implements:
```python
class BaseStage(ABC):
    @abstractmethod
    def run(self, container: DataContainer, context: PipelineContext) -> StageResult: ...
    @property
    @abstractmethod
    def name(self) -> str: ...
```

#### StageResult (base) — `core/base.py`
Every stage produces one:
```python
class StageResult:
    stage_name: str
    duration_seconds: float
    metadata: dict
    def _repr_html_(self) -> str: ...   # Jupyter rendering
    def show(self) -> None: ...         # Jupyter or browser
    def to_dict(self) -> dict: ...      # Serialization
    def to_dataframe(self) -> pd.DataFrame: ...
```

#### DataContainer — `core/data_container.py`
Single data object flowing through stages:
```python
class DataContainer:
    df: pd.DataFrame           # Always materialized to pandas
    target: str | None         # Target column name
    task: TaskType             # classification / regression / auto
    schema: DataSchema         # Column types, cardinalities, inferred roles
    sample: pd.DataFrame       # Subsample for expensive ops on large data
    _original_path: str | None # If loaded from file

    @classmethod
    def from_input(cls, data, target, task) -> "DataContainer":
        """Accepts: str/Path (file), pd.DataFrame, pl.DataFrame (future)"""
```

#### PipelineContext — `core/base.py`
Shared mutable state between stages:
```python
@dataclass
class PipelineContext:
    eda_result: EDAResult | None = None
    intelligence_result: IntelligenceResult | None = None
    battle_result: BattleResult | None = None
    compare_result: CompareResult | None = None
    config: InsightMLConfig = field(default_factory=InsightMLConfig)
    progress: ProgressTracker = field(default_factory=default_progress)
```

#### BaseAnalysisModule (ABC) — `eda/_base.py`
Every EDA sub-module implements:
```python
class BaseAnalysisModule(ABC):
    def __init__(self, df, *, target=None, config=None):
        self._df = df
        self._target = target
        self._config = config or InsightMLConfig()
        self._computed = False
        self._results: dict = {}
        self._warnings: list[str] = []

    @abstractmethod
    def _compute(self) -> None: ...
    @abstractmethod
    def _build_figures(self) -> dict[str, go.Figure]: ...
    @abstractmethod
    def summary(self) -> str: ...

    # Concrete (shared by all):
    def _ensure_computed(self) -> None: ...  # Lazy trigger with error resilience
    def show(self) -> None: ...              # Display all charts
    def plot(self, kind=None) -> go.Figure | dict[str, go.Figure]: ...
    def to_dict(self) -> dict: ...           # JSON-serializable
    def to_dataframe(self) -> pd.DataFrame: ...
    def _repr_html_(self) -> str: ...        # Jupyter rich display
```

Uses `@cached_property` — `explore()` returns instantly; computation only triggers on attribute access.

---

## 4. File I/O (`_io.py`)

Supports CSV, Excel, Parquet, JSON — inferred from file extension:

```python
def read_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    readers = {
        ".csv": pd.read_csv,
        ".tsv": lambda p: pd.read_csv(p, sep="\t"),
        ".xlsx": pd.read_excel,    # requires openpyxl
        ".xls": pd.read_excel,
        ".parquet": pd.read_parquet,  # requires pyarrow or fastparquet
        ".json": pd.read_json,
    }
    if ext not in readers:
        raise UnsupportedFormatError(f"Unsupported file format: {ext}. Supported: {list(readers)}")
    try:
        return readers[ext](path)
    except ImportError as e:
        # e.g., openpyxl not installed for .xlsx
        raise DependencyError(f"Reading {ext} files requires additional package: {e}") from e
```

**Dependency note**: `openpyxl` (Excel) and `pyarrow` (Parquet) are **not** core dependencies. They are installed by the user when needed. The error message tells them what to install.

---

## 5. Exception Hierarchy (`exceptions.py`)

```python
class InsightMLError(Exception):
    """Base exception for all InsightML errors."""

# Input/Validation errors
class ValidationError(InsightMLError): ...
class EmptyDataFrameError(ValidationError): ...
class TargetNotFoundError(ValidationError): ...
class UnsupportedFormatError(ValidationError): ...
class InvalidTaskError(ValidationError): ...

# Dependency errors
class DependencyError(InsightMLError): ...
class OptionalDependencyError(DependencyError): ...

# Computation errors
class ComputationError(InsightMLError): ...
class ModelTrainingError(ComputationError): ...
class TimeoutError(ComputationError): ...

# Report errors
class ReportError(InsightMLError): ...
class TemplateError(ReportError): ...
class ExportError(ReportError): ...
```

---

## 6. Type System (`_types.py`)

```python
from enum import Enum

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    AUTO = "auto"

class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    HIGH_CARDINALITY = "high_cardinality"
    CONSTANT = "constant"
    UNIQUE_ID = "unique_id"

class MissingnessType(str, Enum):
    MCAR = "MCAR"
    MAR = "MAR"
    MNAR = "MNAR"
    UNKNOWN = "unknown"

class TuningMode(str, Enum):
    QUICK = "quick"        # Default params only
    TUNED = "tuned"        # RandomizedSearchCV on top-N
    CUSTOM = "custom"      # User-provided param grids
```

**Task auto-detection** (`core/validators.py`):
1. dtype is object/category/bool -> classification
2. dtype is int AND nunique <= max(20, 0.05 * len) -> classification
3. dtype is float AND nunique <= 20 AND all whole numbers -> classification
4. Otherwise -> regression
5. Emit warning when uncertain (int target with 15-25 unique values)

---

## 7. Configuration (`_config.py`)

```python
@dataclass
class InsightMLConfig:
    # --- EDA ---
    categorical_threshold: int = 50         # nunique <= this = categorical
    high_cardinality_threshold: int = 100   # nunique > this = high_cardinality
    text_min_avg_length: int = 30           # avg str length > this = text
    significance_level: float = 0.05        # p-value threshold for all tests
    iqr_multiplier: float = 1.5             # outlier IQR fence
    zscore_threshold: float = 3.0           # outlier z-score cutoff
    isolation_forest_contamination: float = 0.05
    max_k_clusters: int = 10                # max K for auto K-Means
    correlation_methods: list[str] = field(default_factory=lambda: ["pearson", "spearman", "cramers_v"])

    # --- Battle ---
    cv_folds: int = 5
    timeout_per_model: int = 300            # seconds
    n_jobs: int = -1                        # joblib parallelism
    random_state: int = 42

    # --- Scale ---
    large_dataset_threshold: int = 100_000  # rows; triggers auto-sampling
    sample_size: int = 50_000               # subsample size

    # --- Report ---
    report_theme: str = "default"
    plotly_template: str = "plotly_white"

    # --- General ---
    verbosity: int = 1                      # 0=silent, 1=progress, 2=debug
```

**Three override levels** (resolution: per-call > context > global > defaults):
```python
# Global
iml.set_config(cv_folds=10)
cfg = iml.get_config()

# Temporary
with iml.config_context(cv_folds=3):
    result = iml.battle(df, target="y")

# Per-call
result = iml.battle(df, target="y", cv_folds=10)
```

---

## 8. Stage 1: Deep EDA (`iml.explore(df)`)

### 8.0 EDAResult Orchestrator (`eda/result.py`)

```python
class EDAResult(StageResult):
    """Lazy-evaluated EDA. Each property computes on first access."""

    # --- @cached_property sub-modules ---
    overview:     DataOverview           # Always computed first
    univariate:   UnivariateAnalysis
    bivariate:    BivariateAnalysis
    correlations: CorrelationAnalysis
    missing:      MissingDataIntelligence
    outliers:     OutlierDetection
    tests:        StatisticalTests
    clusters:     ClusterDiscovery
    interactions: FeatureInteractions
    target:       TargetAnalysis | None  # None when no target provided

    def show(self):
        """Display overview + key highlights from each module."""
        # Shows: overview table, missing summary, top correlations,
        # outlier counts, normality flags, cluster count

    def to_dict(self) -> dict:
        """Calls to_dict() on each sub-module."""
```

**Data routing**: `overview` and `missing.counts()` receive **full DataFrame** (exact counts needed). All other modules receive **sampled DataFrame** if len > `config.large_dataset_threshold`.

### 8.1 Type Detection & Overview (`overview.py`)

**Class**: `DataOverview` extends `BaseAnalysisModule`

**Type detection algorithm** per column:
1. dtype is bool OR unique values subset of {True, False, 0, 1} -> `BOOLEAN`
2. dtype is datetime64 -> `DATETIME`
3. dtype is numeric (int/float):
   - nunique == 1 -> `CONSTANT`
   - nunique == len(df) AND dtype is int -> `UNIQUE_ID`
   - else -> `NUMERIC`
4. dtype is object/string/category:
   - nunique == 1 -> `CONSTANT`
   - nunique == len(df) -> `UNIQUE_ID`
   - Try `pd.to_datetime` on sample of 100; if >80% parse -> `DATETIME`
   - nunique <= `config.categorical_threshold` -> `CATEGORICAL`
   - nunique > `config.high_cardinality_threshold` -> `HIGH_CARDINALITY`
   - avg string length > `config.text_min_avg_length` -> `TEXT`
   - fallback -> `CATEGORICAL`

**Per-column profile** (`ColumnProfile` TypedDict):
- All: name, dtype, inferred_type, count, unique, missing_count, missing_pct, memory_bytes
- Numeric: mean, median, std, variance, min, max, range, IQR, Q1/Q3, skewness, kurtosis
- Categorical: top_value, top_freq, cardinality_ratio, value_counts (top 20)
- DateTime: min, max, range_days, inferred_frequency

**Dataset overview**: shape, total_memory_mb, n_duplicates, column_type_counts

**Charts**: type distribution bar, missing overview bar

### 8.2 Univariate Analysis (`univariate.py`)

**Class**: `UnivariateAnalysis`

- **Numeric**: histogram + KDE overlay (scipy.stats.gaussian_kde), box plots, all descriptive stats, quick normality flag
- **Categorical**: frequency bars (top 20, sorted desc), Shannon entropy (scipy.stats.entropy on normalized value counts)
- **DateTime**: time range, gap detection (consecutive diffs > 3x median diff)

**Charts**:
- `go.Histogram(opacity=0.7)` + `go.Scatter` for KDE overlay
- `go.Box(boxpoints="outliers")`
- `go.Bar` for categorical frequencies

### 8.3 Bivariate Analysis (`bivariate.py`)

**Class**: `BivariateAnalysis`

| Type Pair | Analysis | Chart |
|---|---|---|
| Num-Num | Pearson r, Spearman rho, p-values | `go.Scatter` + OLS trendline |
| Num-Cat | Group means, ANOVA F-stat, p-value | `go.Violin` grouped by category |
| Cat-Cat | Chi-square, p-value, Cramer's V, contingency | `go.Heatmap` of counts |
| Any-Target | Auto-selects appropriate analysis | Varies |

**Pair limiting**: when N > 30 columns, only compute pairs involving: (a) target column, (b) top 20 features by variance (numeric) or cardinality (categorical).

### 8.4 Correlation Analysis (`correlations.py`) — KEY DIFFERENTIATOR

**Class**: `CorrelationAnalysis`

**Unified correlation matrix** — the single biggest differentiator vs competitors. One matrix with the statistically appropriate measure per cell:

| Cell Types | Measure | Implementation |
|---|---|---|
| Numeric-Numeric | Pearson r | `df.corr(method="pearson")` |
| Numeric-Numeric (rank) | Spearman rho | `df.corr(method="spearman")` |
| Categorical-Categorical | Cramer's V | `chi2_contingency` -> `sqrt(chi2 / (n * (min_dim-1)))` |
| Numeric-Binary | Point-biserial r | `scipy.stats.pointbiserialr` |
| Numeric-Categorical | Correlation ratio (eta) | `sqrt(SS_between / SS_total)` from groupby |

**Correlation ratio (eta)** implementation:
```python
def correlation_ratio(categories, values):
    groups = values.groupby(categories)
    grand_mean = values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in groups)
    ss_total = ((values - grand_mean)**2).sum()
    return np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0
```

**Methods**: `pearson()`, `spearman()`, `cramers_v_matrix()`, `point_biserial()`, `unified()`, `heatmap(method="unified")`, `top_correlations(n=20)`

**Heatmap**: `go.Heatmap(colorscale="RdBu_r", zmid=0)`, annotated, dynamic sizing `max(600, 40*n_cols)`

### 8.5 Missing Data Intelligence (`missing.py`) — UNIQUE FEATURE

**Class**: `MissingDataIntelligence`

**Little's MCAR Test** algorithm:
1. Restrict to numeric columns with missing values
2. Identify unique missing-data patterns (binary null/not-null mask per row)
3. Group rows by pattern. For each group j:
   - Compute mean vector of observed columns within group
   - Extract corresponding submatrix from pooled covariance
   - Accumulate: `d2 += n_j * (mean_j - grand_mean)^T @ pinv(cov_j) @ (mean_j - grand_mean)`
   - Accumulate degrees of freedom
4. `df_chi2 = df_total - p` (p = number of numeric columns)
5. `p_value = 1 - chi2.cdf(d2, df_chi2)`
6. `p > 0.05` -> likely MCAR

**Per-column MAR detection**:
For each column with missing values, for each other column X:
- Split rows into "column is missing" vs "column is present"
- Run t-test (if X numeric) or chi-square (if X categorical)
- If ANY X shows significant difference (p < 0.05) -> MAR

**MNAR**: residual category when neither MCAR nor MAR detected.

**Imputation recommendations** by (MissingnessType, ColumnType):
- MCAR + numeric: mean, median, KNN
- MCAR + categorical: mode, KNN
- MAR + numeric: KNN, IterativeImputer (MICE)
- MAR + categorical: KNN, IterativeImputer
- MNAR + any: missing-indicator column + fill, flag for domain knowledge

**Methods**: `counts()`, `patterns()`, `pairwise_correlations()`, `littles_test()`, `classify()`, `recommendations()`

**Missing pattern chart**: `go.Heatmap` with binary z=df.isnull().astype(int), colorscale green/red, rows sorted by total missingness.

### 8.6 Outlier Detection (`outliers.py`)

**Class**: `OutlierDetection`

Three methods per numeric column:
- **IQR**: `Q1 - 1.5*IQR` to `Q3 + 1.5*IQR` (configurable multiplier)
- **Z-score**: `|z| > 3.0` (configurable threshold)
- **Isolation Forest**: multi-dimensional, `contamination=0.05`, `n_estimators=100`

**Methods**: `by_iqr()`, `by_zscore()`, `by_isolation_forest()`, `comparison()`, `consensus(min_methods=2)`, `plot(col)`, `plot_comparison()`

**Charts**: `go.Box(boxpoints="outliers")` per column; grouped bar chart comparing methods.

### 8.7 Statistical Tests (`statistical_tests.py`) — UNIQUE FEATURE

**Class**: `StatisticalTests`

**Normality** (per numeric column):
- Shapiro-Wilk: `scipy.stats.shapiro` (sample 5000 if column larger)
- D'Agostino-Pearson: `scipy.stats.normaltest` (requires n >= 20)
- Anderson-Darling: `scipy.stats.anderson(dist="norm")` (compare to 5% critical)
- **Consensus**: 2 out of 3 tests agree = the verdict

**Independence**: `scipy.stats.chi2_contingency` on contingency table of two categoricals

**Variance**: Levene's (`scipy.stats.levene`, robust) + Bartlett's (`scipy.stats.bartlett`, assumes normality)

**Group comparison** (auto-selects):
- All groups normal AND Levene passes -> ANOVA (`scipy.stats.f_oneway`)
- Otherwise -> Kruskal-Wallis (`scipy.stats.kruskal`)
- Returns `{test_used, statistic, p_value, significant, reason_for_test_choice}`

**Methods**: `normality(col=None)`, `independence(col_a, col_b)`, `variance(num_col, group_col)`, `group_comparison(num_col, group_col)`, `all_tests()`

**Chart**: QQ plot (`go.Scatter` of sample vs theoretical quantiles + 45-degree line)

### 8.8 Cluster Discovery (`clusters.py`) — UNIQUE FEATURE

**Class**: `ClusterDiscovery`

**Auto K-Means**:
1. StandardScaler on all numeric features
2. Run K-Means for K = 2..min(config.max_k_clusters, n_samples//2)
3. Record inertia + silhouette score at each K
4. Select K with highest `silhouette_score`; elbow as tiebreaker
5. Final fit with optimal K

**Auto DBSCAN**:
1. `min_samples = max(2 * n_features, 5)`
2. Fit `NearestNeighbors(n_neighbors=min_samples)`, get k-distances sorted
3. Find knee in k-distance curve (max of second derivative) = `eps`

**Cluster profiling**: per-cluster mean (numeric) / mode (categorical) / size / percentage

**Methods**: `kmeans(k=None)`, `dbscan(eps=None, min_samples=None)`, `profiles()`, `elbow_plot()`, `silhouette_plot()`, `scatter_2d(method="pca")`

**Chart**: PCA or t-SNE 2D scatter, `go.Scatter(mode="markers")` per cluster, marker size=5, opacity=0.6

### 8.9 Feature Interactions (`interactions.py`) — UNIQUE FEATURE

**Class**: `FeatureInteractions`

**Interaction strength (with target)**:
1. Create interaction term: `A * B`
2. Fit OLS: `target ~ A + B` -> get residuals
3. Residual-ize interaction term against A + B
4. `strength = |corr(target_residuals, interaction_residuals)|`

**Interaction strength (without target)**:
`strength = 1 - |corr(A*B, A+B)|` (high = non-additive interaction)

**Non-linearity detection** per pair:
1. Fit linear: `B = a*A + b` -> `r2_linear`
2. Fit polynomial (deg 2): `B = a*A^2 + b*A + c` -> `r2_poly`
3. `improvement = r2_poly - r2_linear`; if > 0.05 -> non-linear

**Methods**: `strengths()`, `nonlinear_pairs()`, `interaction_plot(col_a, col_b)`, `top_interactions(n=10)`

### 8.10 Target Analysis (`target_analysis.py`)

**Class**: `TargetAnalysis`

Only available when `target` is provided. Analyzes the target column specifically.

**Classification targets**:
- Class distribution: counts, percentages
- Imbalance ratio: minority_class_count / majority_class_count
- Imbalance severity: "balanced" (>0.8), "mild" (0.4-0.8), "moderate" (0.2-0.4), "severe" (<0.2)
- Recommendation: "Consider SMOTE/class_weight" if moderate/severe
- Chart: `go.Bar` of class frequencies with imbalance annotation

**Regression targets**:
- Distribution: mean, median, std, skew, kurtosis
- Normality test (Shapiro-Wilk)
- If highly skewed (|skew| > 1): recommend log/sqrt transform
- Chart: histogram + KDE of target values

**Feature-Target relationships** (top N features):
- Numeric features vs target: scatter plots (regression) or box/violin (classification)
- Categorical features vs target: grouped bars

**Methods**: `balance()` (classification), `distribution()` (regression), `feature_target_plots(top_n=10)`, `show()`

---

## 9. Stage 2: Pre-Model Intelligence

### 9.1 Target Leakage Detection (`leakage.py`) — UNIQUE FEATURE

**Class**: `LeakageDetector`

Four-pronged scan:

1. **High correlation scan** (threshold > 0.95):
   Pearson (num-num), Cramer's V (cat-cat), point-biserial (num-binary)

2. **Mutual information**: `mutual_info_classif` / `mutual_info_regression`
   Flag columns in top 1% or absolute MI > 1.0

3. **Temporal leakage** (if datetime column present):
   Sort by datetime. Compare `corr(feature, target.shift(-1))` vs `corr(feature, target.shift(1))`.
   If `future_corr > past_corr + 0.3` -> temporal leakage warning

4. **Derived feature detection**:
   Fit `target ~ feature` (OLS). If R2 > 0.98 -> likely derived from target

**Returns**: `list[LeakageWarning]` with `{column, score, method, severity, explanation}`

### 9.2 Multicollinearity (`multicollinearity.py`)

**VIF computation** (manual, numpy-only):
For each numeric feature i: set y=feature_i, X=all_others+intercept -> OLS via `np.linalg.pinv` -> `VIF = 1/(1-R2)`. VIF > 10 = collinear.

**Condition number**: `max(eigenvalues)/min(eigenvalues)` of standardized covariance matrix. < 30 low, 30-100 moderate, > 100 severe.

**Removal recommendations**: for each VIF>10 pair, compare both features' correlation with target, suggest removing the one with lower target correlation.

**Chart**: horizontal bar chart sorted by VIF desc, threshold line at 10.

### 9.3 Feature Importance Pre-Model (`feature_importance.py`)

Four scoring methods:
1. **Mutual information**: `mutual_info_classif` / `mutual_info_regression`
2. **Absolute correlation with target**: `df.corrwith(target).abs()`
3. **ANOVA F-values**: `f_classif` / `f_regression`
4. **Chi-square scores**: `chi2` (only when features non-negative)

**Composite ranking**: rank per method, average ranks, sort by avg_rank ascending.

**Chart**: horizontal bar of top-N features by composite score.

### 9.4 Data Readiness Score (`readiness.py`) — UNIQUE FEATURE

Start at 100, apply deductions/bonuses. Clamped to [0, 100].

**Deductions**:
| Category | Max Penalty | Formula |
|---|---|---|
| Missing values | -25 | `min(25, total_missing_pct * 0.5) + n_cols_over_50pct_missing * 2` |
| Class imbalance | -20 | `max(0, (1 - minority_ratio) * 20)` — classification only |
| Multicollinearity | -15 | `min(15, n_features_VIF_gt_10 * 3)` |
| Outliers | -10 | `min(10, avg_outlier_pct_across_cols * 2)` |
| Constant features | varies | `n_constant_cols * 2` |

**Bonuses**:
| Category | Max Bonus | Formula |
|---|---|---|
| Sample size | +10 | `min(10, samples_per_feature / 10)` — ideal > 100 per feature |
| Feature diversity | +5 | `min(5, n_distinct_column_types * 1.5)` |

**Grades**: A (90+), B (80+), C (70+), D (60+), F (<60)

**Charts**: gauge (`go.Indicator(mode="gauge+number")`) + waterfall breakdown (`go.Waterfall`)

**Accessing other module results**: readiness scorer receives reference to parent EDAResult, accesses `.outliers`, `.missing`, etc. via their `@cached_property` (triggers computation on demand).

### 9.5 Algorithm Recommendations (`recommendations.py`) — UNIQUE FEATURE

Each algorithm has a capability profile:
```python
PROFILES = {
    "LogisticRegression": (nonlinear=F, high_dim=T, missing=F, speed="fast", interpretability="high"),
    "RandomForest":       (nonlinear=T, high_dim=T, missing=F, speed="medium", interpretability="medium"),
    "XGBoost":            (nonlinear=T, high_dim=T, missing=T, speed="medium", interpretability="low"),
    # ... etc for all models
}
```

**Scoring** (starts at 50, adjusts based on EDA findings):
- Non-linear relationships detected: +15 if handles, -20 if doesn't
- High dimensionality (>100 features): +10 if handles, -10 if not
- Missing values present: +10 if native handling
- Class imbalance: +10 if supports, -5 if not
- Small dataset (<1000): -5 for complex models (overfitting risk)
- Large dataset (>100K): -10 for slow models, +10 for fast

Final scores clamped [0, 100], sorted descending.

---

## 10. Stage 3: Multi-Model Training (`iml.battle(df, target)`)

### 10.1 Model Registry (`registry.py` + `catalog.py`)

**19 classifiers** (core sklearn + optional boosting):

| # | Name | Class | Family | Optional | supports_proba |
|---|---|---|---|---|---|
| 1 | Logistic Regression | LogisticRegression | linear | No | Yes |
| 2 | Ridge Classifier | RidgeClassifier | linear | No | No |
| 3 | SGD Classifier | SGDClassifier(loss="log_loss") | linear | No | Yes |
| 4 | K-Nearest Neighbors | KNeighborsClassifier | neighbors | No | Yes |
| 5 | SVC | SVC(probability=True) | svm | No | Yes |
| 6 | Decision Tree | DecisionTreeClassifier | tree | No | Yes |
| 7 | Random Forest | RandomForestClassifier | ensemble | No | Yes |
| 8 | Extra Trees | ExtraTreesClassifier | ensemble | No | Yes |
| 9 | Gradient Boosting | GradientBoostingClassifier | ensemble | No | Yes |
| 10 | Hist Gradient Boosting | HistGradientBoostingClassifier | ensemble | No | Yes |
| 11 | AdaBoost | AdaBoostClassifier | ensemble | No | Yes |
| 12 | Bagging | BaggingClassifier | ensemble | No | Yes |
| 13 | Gaussian NB | GaussianNB | naive_bayes | No | Yes |
| 14 | MLP | MLPClassifier | neural | No | Yes |
| 15 | LDA | LinearDiscriminantAnalysis | discriminant | No | Yes |
| 16 | QDA | QuadraticDiscriminantAnalysis | discriminant | No | Yes |
| 17 | XGBoost | XGBClassifier | ensemble | Yes (boost) | Yes |
| 18 | LightGBM | LGBMClassifier | ensemble | Yes (boost) | Yes |
| 19 | CatBoost | CatBoostClassifier | ensemble | Yes (boost) | Yes |

**17 regressors**: parallel structure — Linear, Ridge, Lasso, ElasticNet, SGD, KNN, SVR, DT, RF, ET, GB, HistGB, AdaBoost, Bagging, MLP, XGB, LGBM, CatBoost.

**ModelEntry dataclass**:
```python
@dataclass
class ModelEntry:
    name: str
    model_class: type
    task: Literal["classification", "regression"]
    family: str
    is_optional: bool
    requires_extra: str | None     # e.g., "boost"
    supports_proba: bool
    default_params: dict[str, Any]
    param_grid: dict[str, list]    # For RandomizedSearchCV
    tags: set[str]                 # {"fast", "interpretable", "handles_missing"}
```

**Registry** is extensible: `iml.registry.register(ModelEntry(...))` for custom models.

### 10.2 EDA-Informed Preprocessing (`preprocessing.py`)

`build_preprocessing_plan()` reads EDA findings:
- Many outliers -> `RobustScaler` instead of `StandardScaler`
- MNAR missing pattern -> `KNNImputer` instead of `SimpleImputer`
- High skewness on target (regression) -> suggest log transform
- High cardinality (>50 unique) -> `OrdinalEncoder`; low (<10) -> `OneHotEncoder`
- Falls back to heuristics if no EDA provided (standalone `battle()` call)

**ColumnTransformer structure**:
```
ColumnTransformer
    numerical: Pipeline([Imputer(median), Scaler(standard or robust)])
    categorical: Pipeline([Imputer(most_frequent), OneHotEncoder(handle_unknown="ignore")])
    high_cardinality: Pipeline([Imputer(most_frequent), OrdinalEncoder(unknown=-1)])
```

**Per-model-family overrides**:
- Tree-based (RF, XGB, LGBM, etc.) -> skip scaling (reduced pipeline)
- CatBoost -> pass `cat_features=` natively, skip encoding
- LightGBM -> `categorical_feature` param, skip encoding
- SVC/KNN -> always get full pipeline with scaling

### 10.3 Training (`runner.py`)

**BattleRunner.run() algorithm**:
1. **Task detection**: auto if not specified
2. **Build preprocessing**: from EDA insights or heuristics
3. **Get models**: from registry, filtered by user selection, skip unavailable optionals
4. **Prepare CV**: `StratifiedKFold(n_splits=cv)` (classification) / `KFold(n_splits=cv)` (regression)
5. **Split**: `X = df.drop(target)`, `y = df[target]`; LabelEncoder for string classification targets
6. **Custom CV loop** (not just `cross_validate`):
   ```python
   for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
       pipeline_clone = clone(pipeline)
       pipeline_clone.fit(X.iloc[train_idx], y.iloc[train_idx])
       fold_predictions[val_idx] = pipeline_clone.predict(X.iloc[val_idx])
       if hasattr(pipeline_clone, "predict_proba"):
           fold_probabilities[val_idx] = pipeline_clone.predict_proba(X.iloc[val_idx])
   ```
   This collects OOF predictions needed for Stage 4 error analysis + ROC curves.
7. **Parallel execution**: `joblib.Parallel(n_jobs=-1)` wrapping model training
8. **Progress**: `tqdm` progress bar (auto-detects notebook vs terminal)
9. **Timeout**: per-model timeout (300s default); failures logged and skipped
10. **Return** `BattleResult`

### 10.4 Metrics

**Classification** (computed via custom CV loop):
| Metric | Implementation |
|---|---|
| Accuracy | `accuracy_score` |
| Precision (weighted) | `precision_score(average="weighted")` |
| Recall (weighted) | `recall_score(average="weighted")` |
| F1 (weighted) | `f1_score(average="weighted")` |
| ROC-AUC | `roc_auc_score(multi_class="ovr", average="weighted")` — requires proba |
| Log Loss | `log_loss` — requires proba |
| MCC | `matthews_corrcoef` |
| Cohen's Kappa | `cohen_kappa_score` |

**Regression**:
| Metric | Implementation |
|---|---|
| MAE | `mean_absolute_error` |
| MSE | `mean_squared_error` |
| RMSE | `sqrt(MSE)` |
| R2 | `r2_score` |
| Adjusted R2 | `1 - (1-R2)*(n-1)/(n-p-1)` |
| MAPE | `mean_absolute_percentage_error` |
| Explained Variance | `explained_variance_score` |

**Multiclass handling**: All classification metrics use `average="weighted"` by default. ROC-AUC uses `multi_class="ovr"`. Models without `predict_proba` (RidgeClassifier) skip probability-dependent metrics.

### 10.5 Result Objects (`result.py`)

```python
@dataclass
class ModelScore:
    name: str
    family: str
    task: str
    cv_scores: dict[str, np.ndarray]     # metric -> per-fold scores
    cv_mean: dict[str, float]            # metric -> mean
    cv_std: dict[str, float]             # metric -> std
    train_time: float                     # wall-clock seconds
    predict_time: float
    fitted_estimator: Pipeline            # best fold's fitted pipeline
    fold_predictions: np.ndarray          # OOF predictions (all samples)
    fold_probabilities: np.ndarray | None # OOF probabilities
    failed: bool = False
    error_message: str | None = None

@dataclass
class BattleResult(StageResult):
    task: str
    target_name: str
    model_scores: list[ModelScore]
    preprocessing_plan: PreprocessingPlan
    feature_names: list[str]
    label_encoder: LabelEncoder | None
    cv_folds: int

    def compare(self, sort_by=None) -> ComparisonTable: ...
    def best(self, metric=None) -> ModelScore: ...
    def to_dataframe(self) -> pd.DataFrame: ...
    def compact(self) -> None:  # Release memory
```

### 10.6 Hyperparameter Tuning (`tuner.py`)

- **Quick** (default): default params only, no search
- **Tuned**: `RandomizedSearchCV` on top-5 models, `n_iter=20`, `cv=3`
- **Custom**: user-provided `param_grids: dict[str, dict]`

Param grids in `param_grids.py` for all models (e.g., RF: n_estimators, max_depth, min_samples_split, etc.)

---

## 11. Stage 4: Comparative Analysis

### 11.1 Comparison Table (`metrics_table.py`)

`ComparisonTable`: DataFrame with models as rows, all metrics + timing as columns. Sorted by primary metric. Styled Jupyter display (bold best, background gradients). `.to_latex()` for papers.

### 11.2 Visualization Suite (`curves.py`)

- **ROC curves**: all models overlaid on single plot — classification
- **Precision-Recall curves**: all models overlaid — classification
- **Confusion matrices**: grid of subplots (top-N models) — classification
- **Residual plots**: grid (predicted vs residual) — regression
- **Actual vs Predicted**: scatter grid — regression
- **Learning curves**: selected models, train/val score vs training size
- **Metric comparison bars**: grouped bar chart

### 11.3 Cross-Model Error Analysis (`error_analysis.py`) — UNIQUE FEATURE

Algorithm:
1. Collect OOF predictions from each model's `fold_predictions`
2. Build disagreement matrix `D[sample, model]` = 1 if correct, 0 if wrong (classification); absolute error (regression)
3. **Pairwise disagreement rate**: fraction where A correct & B wrong or vice versa
4. **Hard samples**: sort by number of models wrong; take bottom 10% (or bottom 50, whichever smaller)
5. **Easy samples**: all models correct; take top 10%
6. **Complementarity matrix**: `C[A][B] = mean((D[:,A]==1) & (D[:,B]==0))` — suggests ensemble candidates
7. **Cluster hard samples**: K-Means (k=3-5, by silhouette) on hard sample features -> reveal shared profiles
8. **Profile hard vs easy**: compare feature distributions, flag significant differences (KS-test or t-test)

### 11.4 Statistical Significance (`significance.py`) — UNIQUE FEATURE

- **McNemar's test**: `statsmodels.stats.contingency_tables.mcnemar` on 2x2 table of paired predictions — binary classification
- **Corrected paired t-test** (Nadeau & Bengio 2003):
  ```python
  corrected_var = (1/k + n_test/n_train) * var(diffs, ddof=1)
  t_stat = mean(diffs) / sqrt(corrected_var)
  p_value = 2 * t.cdf(-abs(t_stat), df=k-1)
  ```
  Works for both classification (on accuracy folds) and regression (on R2 folds).
- **Wilcoxon signed-rank**: `scipy.stats.wilcoxon` — non-parametric alternative
- Returns: n_models x n_models p-value matrix, rendered as `go.Heatmap`

**Multiclass handling**: McNemar extends to multiclass via the multiclass generalization (Stuart-Maxwell test, available in statsmodels). Alternatively, use corrected t-test on per-fold accuracy scores which works regardless of class count.

### 11.5 Pareto Front (`pareto.py`) — UNIQUE FEATURE

Interactive Plotly scatter: X=train_time (or predict_time), Y=primary metric. All models as labeled dots. Pareto-optimal models connected by step line and highlighted in accent color.

**2D Pareto algorithm**: sort by first objective, sweep through tracking best for second objective. O(n log n).

### 11.6 SHAP Comparison (`shap_compare.py`) — UNIQUE FEATURE

**Explainer selection by model family**:
- Tree-based (RF, XGB, LGBM, GB, etc.) -> `shap.TreeExplainer` (fast)
- Linear (LogReg, Ridge, Lasso, etc.) -> `shap.LinearExplainer` (fast)
- Others (SVC, KNN, MLP, etc.) -> `shap.KernelExplainer` (subsample to 100-500 rows)

**To access raw model inside Pipeline**: `model = pipeline[-1]`, `X_transformed = pipeline[:-1].transform(X)`

Side-by-side summary plots for top-N models. Feature importance ranking comparison. Spearman rank correlation between model importance rankings.

---

## 12. Stage 5: Report Generation

### 12.1 HTML Report (`html_renderer.py`)

- **Jinja2** templates via `PackageLoader("dissectml", "report/templates")`
- **Single self-contained HTML file**: CSS inlined in `<style>`, Plotly.js from CDN (option `inline_plotly=True` for offline)
- **Sections**: Executive Summary, EDA Findings, Pre-Model Intelligence, Model Comparison, Detailed Analysis, Recommendations
- **Collapsible**: `<details>/<summary>` with JS enhancement
- **Responsive**: max-width 1100px, flexbox/grid with auto-fill
- **Sticky TOC sidebar**: scroll-spy navigation
- **Charts**: `fig.to_html(full_html=False, include_plotlyjs=False)` per chart

### 12.2 Narrative Summaries (`narrative.py`)

Template-driven (no LLM dependency):
- **Executive summary**: "The dataset contains {n_rows} samples with {n_features} features. {best_model} achieved {score:.4f} {metric}, significantly better than {second_model} (p={p:.3f})."
- **Model narrative**: per-model paragraph with strengths/weaknesses
- **Recommendations**:
  - If top-2 not statistically significant -> recommend simpler one
  - If Pareto shows faster model within 1% accuracy -> mention it
  - If error analysis shows complementary models -> suggest ensemble
  - If EDA found issues (leakage, missing, imbalance) -> warn

### 12.3 PDF Export (`pdf_renderer.py`) — Optional

Requires `pip install dissectml[report]` (weasyprint + kaleido). Renders HTML to PDF with static chart images. Falls back gracefully with clear install instructions.

### 12.4 Jupyter Integration (`viz/display.py`)

**Environment detection**:
```python
def detect_environment():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell": return "jupyter"
        if "google.colab" in str(get_ipython()): return "colab"
    except NameError: pass
    if "VSCODE_PID" in os.environ: return "vscode"
    return "terminal"
```

Every result object implements `_repr_html_()`. `.show()` adapts: notebook -> `IPython.display.HTML`, terminal -> `webbrowser.open(temp_html)`.

**Progress bars**: tqdm.notebook in notebooks, rich.progress in terminals.

---

## 13. Packaging (`pyproject.toml`)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "dissectml"
dynamic = ["version"]
description = "The missing middle layer between EDA and AutoML — deep data understanding meets model comparison"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [{ name = "Rupesh Bharambe" }]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
keywords = ["machine-learning", "eda", "automl", "data-science", "model-comparison"]

dependencies = [
    "pandas>=2.0",
    "numpy>=1.24,<2.0",
    "scikit-learn>=1.3",
    "scipy>=1.10",
    "plotly>=5.18",
    "jinja2>=3.1",
    "joblib>=1.3",
    "tqdm>=4.65",
    "statsmodels>=0.14",
    "pydantic>=2.0",
    "rich>=13.0",
]

[project.optional-dependencies]
boost = ["xgboost>=2.0", "lightgbm>=4.0", "catboost>=1.2"]
explain = ["shap>=0.44"]
report = ["weasyprint>=60.0", "kaleido>=0.2"]
scale = ["polars>=0.20", "optuna>=3.4"]
full = ["dissectml[boost,explain,report,scale]"]
dev = [
    "dissectml[full]",
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "pytest-xdist>=3.5",
    "mypy>=1.8",
    "ruff>=0.3",
    "pre-commit>=3.6",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.24",
    "hatchling",
]

[project.urls]
Homepage = "https://github.com/rupeshbharambe/dissectml"
Documentation = "https://dissectml.readthedocs.io"
Repository = "https://github.com/rupeshbharambe/dissectml"

[tool.hatch.version]
path = "src/dissectml/_version.py"

[tool.hatch.build.targets.wheel]
packages = ["src/dissectml"]

[tool.ruff]
target-version = "py310"
line-length = 100
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.10"
packages = ["dissectml"]
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = [
    "slow: tests requiring >10s (deselect with '-m not slow')",
    "integration: full-pipeline integration tests",
    "optional: tests requiring optional dependencies",
]
```

**Note**: `numpy>=1.24,<2.0` pins numpy below 2.0 to avoid the scipy/sklearn compatibility issues seen in the shared env. We can relax this once all deps support numpy 2.x.

---

## 14. Visualization Theme (`viz/theme.py`)

Shared Plotly template for all charts:
- Template: `plotly_white`
- Font: `Inter, system-ui, sans-serif`; monospace for annotations: `JetBrains Mono, monospace`
- Color palette: `["#6366f1", "#22c55e", "#ef4444", "#f59e0b", "#06b6d4", "#ec4899", "#f97316", "#8b5cf6", "#14b8a6", "#f43f5e"]`
- Margins: `dict(l=60, r=30, t=50, b=40)`
- Title font size: 16, body: 12
- Grid: light gray, zeroline hidden

All chart factory functions in `viz/charts.py` apply this theme automatically.

---

## 15. Large Dataset Strategy

| Operation | Uses Full Data | Uses Sampled Data | Reason |
|---|---|---|---|
| Type detection | Yes | | Must check all values |
| Missing counts/percentages | Yes | | Must be exact |
| Dataset overview (shape, memory) | Yes | | Must be exact |
| Univariate aggregates (mean, std) | Yes | | O(n), cheap |
| Charts (histograms, scatter) | | Yes | Visual quality OK at 50K |
| Correlations | | Yes | Stable at >10K samples |
| Statistical tests | | Yes | Designed for samples |
| Outlier detection (IQR, Z-score) | Yes | | O(n), cheap |
| Outlier detection (Isolation Forest) | | Yes | Tree ensemble, expensive |
| Clustering, interactions | | Yes | Expensive algorithms |
| VIF computation | | Yes | OLS per feature, expensive |
| SHAP computation | | Yes (500 rows) | KernelExplainer is O(2^M) |

**Smart sampling** (`_sampling.py`):
```python
class SmartSampler:
    def sample(self, df, target=None):
        if len(df) <= config.large_dataset_threshold:
            return df
        if target and target in df.columns:
            # Stratified: preserve class proportions
            return stratified_sample(df, target, config.sample_size)
        datetime_cols = df.select_dtypes(include="datetime64").columns
        if len(datetime_cols) > 0:
            # Temporal: evenly spaced rows
            step = max(1, len(df) // config.sample_size)
            return df.iloc[::step].head(config.sample_size)
        # Default: random
        return df.sample(n=config.sample_size, random_state=config.random_state)
```

---

## 16. Phased Implementation Roadmap

### Phase 0: Scaffolding (implement first, ~1 session)

**Goal**: Project skeleton that installs and imports.

- [ ] Init git repo, `.gitignore`
- [ ] `pyproject.toml` with all deps/groups
- [ ] Create `.venv`, install in editable mode
- [ ] `src/dissectml/__init__.py` — stub `analyze()`, `explore()`, `battle()`
- [ ] `src/dissectml/_version.py` — `__version__ = "0.1.0"`
- [ ] `src/dissectml/_types.py` — all enums + TypedDicts
- [ ] `src/dissectml/_config.py` — InsightMLConfig + get/set/context
- [ ] `src/dissectml/_lazy.py` — optional dependency guard
- [ ] `src/dissectml/_io.py` — file loading (CSV, Excel, Parquet, JSON)
- [ ] `src/dissectml/_sampling.py` — SmartSampler
- [ ] `src/dissectml/exceptions.py` — full exception hierarchy
- [ ] `src/dissectml/core/base.py` — BaseStage, StageResult, PipelineContext
- [ ] `src/dissectml/core/data_container.py` — DataContainer + from_input()
- [ ] `src/dissectml/core/validators.py` — input validation + task inference
- [ ] `src/dissectml/core/progress.py` — ProgressTracker
- [ ] `src/dissectml/viz/theme.py` — Plotly theme
- [ ] `src/dissectml/viz/charts.py` — chart factory stubs
- [ ] `src/dissectml/viz/display.py` — env detection + _repr_html_ mixin
- [ ] `src/dissectml/eda/_base.py` — BaseAnalysisModule ABC
- [ ] `tests/conftest.py` — shared fixtures (synthetic DataFrames)
- [ ] Verify: `pip install -e .` works, `import dissectml` works

### Phase 1: Deep EDA — v0.1 (~3-4 sessions)

**Goal**: `iml.explore(df)` returns full EDA with interactive output.

- [ ] `eda/result.py` — EDAResult lazy orchestrator
- [ ] `eda/overview.py` — type detection + dataset stats
- [ ] `eda/univariate.py` — distributions, descriptive stats, KDE
- [ ] `eda/bivariate.py` — cross-type analysis
- [ ] `eda/correlations.py` — unified correlation matrix (KEY)
- [ ] `eda/missing.py` — Little's MCAR + MAR/MNAR + recommendations
- [ ] `eda/outliers.py` — IQR + Z-score + Isolation Forest + consensus
- [ ] `eda/statistical_tests.py` — normality, independence, variance, group comparison
- [ ] `eda/clusters.py` — auto K-Means/DBSCAN + profiles + PCA/t-SNE
- [ ] `eda/interactions.py` — interaction strength + non-linearity
- [ ] `eda/target_analysis.py` — class balance / regression distribution / feature-target
- [ ] Wire up `iml.explore(df)` in `__init__.py`
- [ ] Tests for all EDA modules
- [ ] Manual test: run in notebook, verify `_repr_html_()` renders

### Phase 2: Model Battle — v0.2 (~3-4 sessions)

**Goal**: `iml.battle(df, target)` trains and compares 15+ models.

- [ ] `battle/catalog.py` — MODEL_CATALOG with all model entries
- [ ] `battle/registry.py` — ModelRegistry with register/unregister/filter
- [ ] `battle/preprocessing.py` — EDA-informed ColumnTransformer builder
- [ ] `battle/runner.py` — BattleRunner with parallel CV + timing + OOF collection
- [ ] `battle/result.py` — BattleResult + ModelScore
- [ ] `battle/tuner.py` + `battle/param_grids.py` — hyperparameter tuning
- [ ] Wire up `iml.battle(df, target)` in `__init__.py`
- [ ] Tests for registry, runner, preprocessing

### Phase 3: Intelligence Bridge — v0.3 (~2 sessions)

**Goal**: EDA insights feed into model selection and data readiness scoring.

- [ ] `intelligence/leakage.py` — 4-pronged target leakage scan
- [ ] `intelligence/multicollinearity.py` — VIF + condition number
- [ ] `intelligence/feature_importance.py` — composite ranking
- [ ] `intelligence/readiness.py` — 0-100 score with gauge + waterfall
- [ ] `intelligence/recommendations.py` — algorithm rules engine
- [ ] Wire EDA -> Intelligence -> Battle (context propagation)
- [ ] Tests

### Phase 4: Compare + Report — v0.4 (~3-4 sessions)

**Goal**: Full `iml.analyze()` pipeline with HTML report export.

- [ ] `compare/comparator.py` — ModelComparator facade
- [ ] `compare/metrics_table.py` — styled comparison table
- [ ] `compare/curves.py` — ROC, PR, confusion matrices, residuals
- [ ] `compare/error_analysis.py` — cross-model disagreement + clustering
- [ ] `compare/significance.py` — McNemar + corrected t-test + Wilcoxon
- [ ] `compare/pareto.py` — Pareto front
- [ ] `compare/shap_compare.py` — SHAP across top-N models
- [ ] `report/builder.py` — ReportBuilder (assembles AnalysisReport)
- [ ] `report/narrative.py` — template-driven summaries
- [ ] `report/html_renderer.py` — Jinja2 -> self-contained HTML
- [ ] `report/templates/` — all Jinja2 templates + CSS + JS
- [ ] Wire up `iml.analyze(df, target)` in `__init__.py`
- [ ] `report.export("report.html")` working end-to-end
- [ ] Integration tests

### Phase 5: Scale & Polish — v0.5 (future)

- [ ] Polars backend in `_compat.py`
- [ ] PDF export (`report/pdf_renderer.py`)
- [ ] Plugin system (entry-points based)
- [ ] Built-in demo datasets (`datasets/`)
- [ ] Example Jupyter notebooks
- [ ] MkDocs Material documentation site
- [ ] CI/CD (GitHub Actions)
- [ ] Target: publish to PyPI

---

## 17. Testing Strategy

### Fixtures (`tests/conftest.py`)

```python
@pytest.fixture
def regression_df():
    """50 rows, 5 numeric + 2 categorical features, 1 continuous target."""

@pytest.fixture
def classification_df():
    """100 rows, binary classification, 70/30 imbalance, mixed types."""

@pytest.fixture
def multiclass_df():
    """150 rows, 3-class classification, balanced."""

@pytest.fixture
def missing_heavy_df():
    """100 rows, ~30% missing in known MCAR/MAR patterns."""

@pytest.fixture
def edge_case_df():
    """Constant columns, all-null columns, single-row, huge cardinality."""
```

### Test Categories

- **Unit** (no marker): fast, isolated, one function each. Every public method.
- **`@pytest.mark.slow`**: >10s (model training, large data)
- **`@pytest.mark.integration`**: full pipeline end-to-end
- **`@pytest.mark.optional`**: requires optional deps; auto-skip via `pytest.importorskip()`

### Verification Checklist

- [ ] `pip install -e .` works in clean `.venv`
- [ ] `import dissectml as iml` works
- [ ] `iml.explore(df)` returns `EDAResult` with all sub-modules accessible
- [ ] `iml.explore(df).show()` renders in Jupyter
- [ ] `iml.explore(df).correlations.heatmap()` returns interactive Plotly figure
- [ ] `iml.battle(df, target="y")` trains 15+ models, returns sorted leaderboard
- [ ] `iml.analyze(df, target="y").export("report.html")` produces valid HTML
- [ ] Edge cases: all-null columns don't crash, constant features are handled
- [ ] Large dataset (100K+ rows): auto-sampling kicks in, reasonable runtime
- [ ] Optional deps missing: graceful skip with clear error message

---

## 18. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| SVC hangs on datasets >10K rows | Per-model timeout (300s); auto-exclude SVC when n>10K unless user forces |
| MultinomialNB requires non-negative features | Check at runtime after preprocessing; skip and log if negative |
| CatBoost console spam | `CatBoostClassifier(verbose=0)` in default_params |
| SHAP KernelExplainer extremely slow | Subsample to 100-500 rows; only top-3 models; make SHAP optional |
| weasyprint broken on Windows (missing GTK) | PDF export fully optional with clear error message |
| Plotly.js is ~3.5MB inlined | Default to CDN; `inline_plotly=True` for offline |
| Models without predict_proba (RidgeClassifier) | Skip probability-dependent metrics; use `decision_function` where possible |
| NumPy 2.x compatibility | Pin `numpy<2.0` in deps until ecosystem catches up |
| Scope creep | Strict phase boundaries; ship v0.1 EDA-only |
