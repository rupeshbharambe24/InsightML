"""Pre-model feature importance — composite ranking from multiple methods."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_feature_importance(
    df: pd.DataFrame,
    target: str,
    task: str | None = None,
) -> pd.DataFrame:
    """Rank features using up to 4 methods, returning a composite ranking.

    Methods (applied where feasible):
    1. **Mutual information** — ``mutual_info_classif`` / ``mutual_info_regression``
    2. **Absolute correlation** — |Pearson| with (encoded) target
    3. **F-statistic** — ``f_classif`` / ``f_regression`` (numeric features only)
    4. **Chi-square** — ``chi2`` (non-negative features + classification only)

    Final rank = average rank across all applicable methods (lower = more important).

    Args:
        df: DataFrame with features + target.
        target: Target column name.
        task: ``"classification"`` or ``"regression"``. Inferred if None.

    Returns:
        DataFrame with columns: feature, mi, abs_corr, f_stat, chi2,
        composite_rank, sorted by composite_rank ascending.
    """
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in DataFrame.")

    y = df[target]
    if task is None:
        task = _infer_task(y)

    feature_cols = [c for c in df.columns if c != target]
    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() >= 5
    ]

    if not numeric_cols:
        return pd.DataFrame(columns=["feature", "composite_rank"])

    # Prepare X matrix (median-impute) and y
    X = df[numeric_cols].copy()
    for col in numeric_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    y_enc = _encode_target(y)
    if y_enc is None:
        return pd.DataFrame(columns=["feature", "composite_rank"])

    mask_y = y_enc.notna()
    X_clean = X[mask_y].reset_index(drop=True)
    y_clean = y_enc[mask_y].reset_index(drop=True)

    scores: dict[str, dict[str, float]] = {col: {} for col in numeric_cols}

    # --- 1. Mutual information ---
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        fn = mutual_info_classif if task == "classification" else mutual_info_regression
        mi = fn(X_clean, y_clean, random_state=42)
        for col, val in zip(numeric_cols, mi):
            scores[col]["mi"] = round(float(val), 6)
    except Exception:
        pass

    # --- 2. Absolute Pearson correlation ---
    try:
        for col in numeric_cols:
            corr = abs(float(X_clean[col].corr(y_clean)))
            if not np.isnan(corr):
                scores[col]["abs_corr"] = round(corr, 6)
    except Exception:
        pass

    # --- 3. F-statistic ---
    try:
        from sklearn.feature_selection import f_classif, f_regression
        fn_f = f_classif if task == "classification" else f_regression
        f_vals, _ = fn_f(X_clean, y_clean)
        for col, val in zip(numeric_cols, f_vals):
            if not np.isnan(val):
                scores[col]["f_stat"] = round(float(val), 6)
    except Exception:
        pass

    # --- 4. Chi-square (classification + non-negative features only) ---
    if task == "classification":
        try:
            from sklearn.feature_selection import chi2
            nonneg_cols = [c for c in numeric_cols if float(X_clean[c].min()) >= 0]
            if nonneg_cols:
                chi2_vals, _ = chi2(X_clean[nonneg_cols], y_clean)
                for col, val in zip(nonneg_cols, chi2_vals):
                    if not np.isnan(val):
                        scores[col]["chi2"] = round(float(val), 6)
        except Exception:
            pass

    # --- Build rows ---
    rows = []
    for col in numeric_cols:
        row: dict[str, Any] = {"feature": col}
        row.update(scores[col])
        rows.append(row)

    result = pd.DataFrame(rows)

    # --- Composite rank = mean rank across available score columns ---
    score_cols = [c for c in ["mi", "abs_corr", "f_stat", "chi2"] if c in result.columns]
    if not score_cols:
        result["composite_rank"] = range(1, len(result) + 1)
        return result

    rank_cols = []
    for sc in score_cols:
        rank_col = f"_rank_{sc}"
        result[rank_col] = result[sc].rank(ascending=False, na_option="bottom")
        rank_cols.append(rank_col)

    result["composite_rank"] = result[rank_cols].mean(axis=1).round(2)
    result = result.drop(columns=rank_cols)
    result = result.sort_values("composite_rank").reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_task(y: pd.Series) -> str:
    if (
        pd.api.types.is_bool_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(y)
        or str(y.dtype) in ("string", "category")
        or y.nunique() <= 20
    ):
        return "classification"
    return "regression"


def _encode_target(y: pd.Series) -> pd.Series | None:
    if pd.api.types.is_numeric_dtype(y):
        return y
    try:
        encoded = pd.Categorical(y).codes.astype(float)
        s = pd.Series(encoded, index=y.index)
        s[y.isna()] = np.nan
        return s
    except Exception:
        return None
