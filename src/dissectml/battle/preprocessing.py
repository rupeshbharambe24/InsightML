"""EDA-informed ColumnTransformer builder for the battle pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


@dataclass
class PreprocessingPlan:
    """Describes the preprocessing choices made for a dataset.

    Attributes:
        numeric_cols: Columns treated as numeric.
        categorical_cols: Low-cardinality categorical columns (OneHotEncoded).
        high_card_cols: High-cardinality categorical columns (OrdinalEncoded).
        imputer: "simple" or "knn" for numeric missing values.
        scaler: "robust" or "standard".
        reasons: Human-readable explanation of each choice.
    """

    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    high_card_cols: list[str] = field(default_factory=list)
    imputer: str = "simple"
    scaler: str = "standard"
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Numeric ({len(self.numeric_cols)}): {self.numeric_cols[:5]}{'...' if len(self.numeric_cols) > 5 else ''}",
            f"Categorical ({len(self.categorical_cols)}): {self.categorical_cols[:5]}{'...' if len(self.categorical_cols) > 5 else ''}",
            f"High-cardinality ({len(self.high_card_cols)}): {self.high_card_cols[:5]}{'...' if len(self.high_card_cols) > 5 else ''}",
            f"Imputer: {self.imputer}",
            f"Scaler: {self.scaler}",
        ]
        if self.reasons:
            lines.append("Reasons:")
            lines.extend(f"  • {r}" for r in self.reasons)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------

_HIGH_CARDINALITY_THRESHOLD = 15  # OneHot vs OrdinalEncoder boundary
_OUTLIER_SEVERITY_THRESHOLD = 0.05  # fraction of outliers -> RobustScaler
_MNAR_IMPUTER = "knn"  # KNN for MNAR / MAR patterns


def build_preprocessing_plan(
    df: pd.DataFrame,
    target: str | None = None,
    eda_result: Any = None,
    *,
    high_cardinality_threshold: int = _HIGH_CARDINALITY_THRESHOLD,
) -> PreprocessingPlan:
    """Derive a :class:`PreprocessingPlan` from a DataFrame (and optional EDA findings).

    When *eda_result* is provided, the plan incorporates:
    - Outlier severity -> RobustScaler vs StandardScaler
    - Missing-data mechanism (MNAR/MAR) -> KNNImputer vs SimpleImputer

    Otherwise heuristics on the raw DataFrame are used.

    Args:
        df: Input DataFrame (features only, target already separated if needed).
        target: Target column name to exclude from feature columns.
        eda_result: Optional ``EDAResult`` from ``dml.explore()``.
        high_cardinality_threshold: nunique >= this -> OrdinalEncoder.

    Returns:
        PreprocessingPlan ready to pass to :func:`build_transformer`.
    """
    reasons: list[str] = []

    feature_cols = [c for c in df.columns if c != target]
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    high_card_cols: list[str] = []

    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            n_unique = series.nunique(dropna=True)
            if n_unique >= high_cardinality_threshold:
                high_card_cols.append(col)
            else:
                categorical_cols.append(col)

    # --- Scaler choice ---
    scaler = "standard"
    if eda_result is not None:
        try:
            outlier_summary = eda_result.outliers.consensus()
            if isinstance(outlier_summary, pd.DataFrame) and not outlier_summary.empty:
                if "fraction" in outlier_summary.columns:
                    avg_frac = float(outlier_summary["fraction"].mean())
                else:
                    avg_frac = 0.0
                if avg_frac > _OUTLIER_SEVERITY_THRESHOLD:
                    scaler = "robust"
                    reasons.append(
                        f"RobustScaler: avg outlier fraction {avg_frac:.2%} > "
                        f"{_OUTLIER_SEVERITY_THRESHOLD:.0%} threshold"
                    )
        except Exception:
            pass

    if scaler == "standard" and eda_result is None:
        # Heuristic: check skewness of numeric columns
        try:
            skews = [abs(float(df[c].skew())) for c in numeric_cols if df[c].notna().sum() > 3]
            if skews and (sum(s > 2 for s in skews) / len(skews)) > 0.3:
                scaler = "robust"
                reasons.append("RobustScaler: >30% of numeric columns have |skew|>2")
        except Exception:
            pass

    # --- Imputer choice ---
    imputer = "simple"
    if eda_result is not None:
        try:
            missingness = eda_result.missing.summary()
            # If any column is classified as MAR or MNAR, use KNN
            if isinstance(missingness, dict):
                mechanisms = [
                    v.get("mechanism", "")
                    for v in missingness.values()
                    if isinstance(v, dict)
                ]
            elif isinstance(missingness, pd.DataFrame) and "mechanism" in missingness.columns:
                mechanisms = list(missingness["mechanism"].dropna())
            else:
                mechanisms = []

            if any(m in ("MAR", "MNAR") for m in mechanisms):
                imputer = "knn"
                reasons.append(
                    "KNNImputer: MAR/MNAR missingness detected in one or more columns"
                )
        except Exception:
            pass

    if imputer == "simple" and eda_result is None:
        # Heuristic: if any numeric column has >20% missing, use KNN
        try:
            high_missing = [
                c for c in numeric_cols
                if df[c].isna().mean() > 0.20
            ]
            if high_missing:
                imputer = "knn"
                reasons.append(
                    f"KNNImputer: {len(high_missing)} numeric col(s) have >20% missing"
                )
        except Exception:
            pass

    if not reasons:
        reasons.append(
            f"StandardScaler + {'KNNImputer' if imputer == 'knn' else 'SimpleImputer(median)'}: "
            "default heuristics (no outlier/missingness flags)"
        )

    return PreprocessingPlan(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        high_card_cols=high_card_cols,
        imputer=imputer,
        scaler=scaler,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Transformer builder
# ---------------------------------------------------------------------------

def build_transformer(
    plan: PreprocessingPlan,
    tree_based: bool = False,
) -> ColumnTransformer:
    """Build a fitted-ready :class:`~sklearn.compose.ColumnTransformer` from a plan.

    Tree-based models skip scaling entirely (pass ``tree_based=True``).

    Args:
        plan: Output of :func:`build_preprocessing_plan`.
        tree_based: If True, omit scaler from the numeric pipeline.

    Returns:
        An unfitted ``ColumnTransformer``.
    """
    transformers: list[tuple] = []

    # Numeric pipeline
    if plan.numeric_cols:
        if plan.imputer == "knn":
            num_imputer = KNNImputer(n_neighbors=5)
        else:
            num_imputer = SimpleImputer(strategy="median")

        if tree_based:
            num_pipeline = Pipeline([("imputer", num_imputer)])
        else:
            scaler = RobustScaler() if plan.scaler == "robust" else StandardScaler()
            num_pipeline = Pipeline([("imputer", num_imputer), ("scaler", scaler)])

        transformers.append(("numerical", num_pipeline, plan.numeric_cols))

    # Categorical pipeline (low cardinality -> OneHotEncoder)
    if plan.categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", cat_pipeline, plan.categorical_cols))

    # High-cardinality pipeline (OrdinalEncoder)
    if plan.high_card_cols:
        high_card_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
        transformers.append(("high_cardinality", high_card_pipeline, plan.high_card_cols))

    if not transformers:
        # Edge case: no feature columns — return passthrough
        return ColumnTransformer(transformers=[], remainder="passthrough")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_full_pipeline(
    estimator: Any,
    plan: PreprocessingPlan,
    tree_based: bool = False,
) -> Pipeline:
    """Build a complete sklearn Pipeline: preprocessor -> estimator.

    Args:
        estimator: A fitted/unfitted sklearn-compatible estimator.
        plan: Output of :func:`build_preprocessing_plan`.
        tree_based: Passed to :func:`build_transformer`.

    Returns:
        An unfitted ``Pipeline`` with steps ``[("preprocessor", ...), ("model", estimator)]``.
    """
    preprocessor = build_transformer(plan, tree_based=tree_based)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])
