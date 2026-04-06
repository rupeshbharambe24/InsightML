"""Target leakage detection — 4-pronged scan for data leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression

from dissectml._types import LeakageWarning

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_CORR_THRESHOLD = 0.95       # absolute correlation -> high-corr leakage
_MI_THRESHOLD = 1.0          # mutual information -> MI leakage
_MI_TOP_PCT = 0.01           # top 1% of MI scores also flagged
_TEMPORAL_DIFF_THRESHOLD = 0.30  # |future_corr - past_corr| -> temporal leakage
_OLS_R2_THRESHOLD = 0.98     # single-feature OLS R² -> derived-feature leakage


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_leakage(
    df: pd.DataFrame,
    target: str,
    datetime_col: str | None = None,
    significance_level: float = 0.05,
) -> list[LeakageWarning]:
    """Run the 4-pronged leakage scan.

    Prongs:
    1. **High correlation** — |Pearson/Cramer/point-biserial| > 0.95
    2. **Mutual information** — MI > 1.0 or top 1%
    3. **Temporal leakage** — if datetime present, future vs past correlation diff > 0.30
    4. **Derived feature** — single-feature OLS R² > 0.98

    Args:
        df: Input DataFrame (features + target).
        target: Target column name.
        datetime_col: Optional datetime column for temporal leakage check.
        significance_level: Not used directly but kept for API consistency.

    Returns:
        List of :class:`~dissectml._types.LeakageWarning` dicts, one per flagged column.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    feature_cols = [c for c in df.columns if c != target and c != datetime_col]
    y = df[target]

    seen: dict[str, LeakageWarning] = {}  # col -> highest-severity warning

    # --- Prong 1: High correlation ---
    for w in _high_correlation_scan(df, feature_cols, y):
        _merge_warning(seen, w)

    # --- Prong 2: Mutual information ---
    for w in _mutual_information_scan(df, feature_cols, y):
        _merge_warning(seen, w)

    # --- Prong 3: Temporal leakage ---
    if datetime_col and datetime_col in df.columns:
        for w in _temporal_leakage_scan(df, feature_cols, y, datetime_col):
            _merge_warning(seen, w)

    # --- Prong 4: Derived feature ---
    for w in _derived_feature_scan(df, feature_cols, y):
        _merge_warning(seen, w)

    return sorted(seen.values(), key=lambda w: w["score"], reverse=True)


# ---------------------------------------------------------------------------
# Prong implementations
# ---------------------------------------------------------------------------


def _high_correlation_scan(
    df: pd.DataFrame, feature_cols: list[str], y: pd.Series
) -> list[LeakageWarning]:
    results = []
    y_numeric = _encode_target(y)
    if y_numeric is None:
        return results

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        mask = df[col].notna() & y_numeric.notna()
        if mask.sum() < 10:
            continue
        try:
            corr = float(df.loc[mask, col].corr(y_numeric[mask]))
            abs_corr = abs(corr)
            if abs_corr >= _CORR_THRESHOLD:
                results.append(LeakageWarning(
                    column=col,
                    score=round(abs_corr, 4),
                    method="high_correlation",
                    severity=_severity(abs_corr, [0.95, 0.98, 0.999]),
                    explanation=(
                        f"|Pearson r|={abs_corr:.4f} with target — "
                        f"almost perfectly correlated, likely leakage."
                    ),
                ))
        except Exception:
            pass
    return results


def _mutual_information_scan(
    df: pd.DataFrame, feature_cols: list[str], y: pd.Series
) -> list[LeakageWarning]:
    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() >= 10
    ]
    if not numeric_cols:
        return []

    X = df[numeric_cols].copy()
    # Simple median imputation for MI computation
    for col in numeric_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    y_clean = y.copy()
    mask = y_clean.notna()
    X = X[mask].reset_index(drop=True)
    y_arr = y_clean[mask].reset_index(drop=True)

    try:
        task = _infer_task(y_arr)
        fn = mutual_info_classif if task == "classification" else mutual_info_regression
        mi_scores = fn(X, y_arr, random_state=42)
    except Exception:
        return []

    threshold_top_pct = np.percentile(mi_scores[mi_scores > 0], 99) if any(mi_scores > 0) else np.inf

    results = []
    for col, mi in zip(numeric_cols, mi_scores):
        if mi >= _MI_THRESHOLD or mi >= threshold_top_pct:
            results.append(LeakageWarning(
                column=col,
                score=round(float(mi), 4),
                method="mutual_information",
                severity=_severity(mi, [1.0, 2.0, 4.0]),
                explanation=(
                    f"Mutual information={mi:.4f} with target — "
                    f"extremely high information overlap."
                ),
            ))
    return results


def _temporal_leakage_scan(
    df: pd.DataFrame,
    feature_cols: list[str],
    y: pd.Series,
    datetime_col: str,
) -> list[LeakageWarning]:
    results = []
    try:
        dt = pd.to_datetime(df[datetime_col], errors="coerce")
        median_dt = dt.dropna().median()
        past_mask = (dt <= median_dt) & dt.notna()
        future_mask = (dt > median_dt) & dt.notna()
    except Exception:
        return results

    if past_mask.sum() < 5 or future_mask.sum() < 5:
        return results

    y_numeric = _encode_target(y)
    if y_numeric is None:
        return results

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            past_corr = abs(float(
                df.loc[past_mask, col].corr(y_numeric[past_mask])
            ))
            future_corr = abs(float(
                df.loc[future_mask, col].corr(y_numeric[future_mask])
            ))
            diff = abs(future_corr - past_corr)
            if diff >= _TEMPORAL_DIFF_THRESHOLD and future_corr > past_corr:
                results.append(LeakageWarning(
                    column=col,
                    score=round(diff, 4),
                    method="temporal_leakage",
                    severity=_severity(diff, [0.30, 0.50, 0.70]),
                    explanation=(
                        f"Future correlation ({future_corr:.3f}) >> "
                        f"past correlation ({past_corr:.3f}); "
                        f"diff={diff:.3f} — possible look-ahead bias."
                    ),
                ))
        except Exception:
            pass
    return results


def _derived_feature_scan(
    df: pd.DataFrame, feature_cols: list[str], y: pd.Series
) -> list[LeakageWarning]:
    results = []
    y_numeric = _encode_target(y)
    if y_numeric is None:
        return results

    mask_y = y_numeric.notna()

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        mask = df[col].notna() & mask_y
        if mask.sum() < 10:
            continue
        try:
            X_col = df.loc[mask, col].values.reshape(-1, 1)
            y_col = y_numeric[mask].values
            reg = LinearRegression()
            reg.fit(X_col, y_col)
            ss_res = float(np.sum((y_col - reg.predict(X_col)) ** 2))
            ss_tot = float(np.sum((y_col - y_col.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if r2 >= _OLS_R2_THRESHOLD:
                results.append(LeakageWarning(
                    column=col,
                    score=round(r2, 4),
                    method="derived_feature",
                    severity=_severity(r2, [0.98, 0.99, 0.999]),
                    explanation=(
                        f"Single-feature OLS R²={r2:.4f} — "
                        f"feature may be derived from / identical to target."
                    ),
                ))
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_target(y: pd.Series) -> pd.Series | None:
    """Return numeric version of y, or None if not encodable."""
    if pd.api.types.is_numeric_dtype(y):
        return y
    try:
        encoded = pd.Categorical(y).codes.astype(float)
        s = pd.Series(encoded, index=y.index)
        s[y.isna()] = np.nan
        return s
    except Exception:
        return None


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


def _severity(score: float, thresholds: list[float]) -> str:
    """Map score to low/moderate/high/critical via thresholds list (3 values)."""
    if score >= thresholds[2]:
        return "critical"
    if score >= thresholds[1]:
        return "high"
    if score >= thresholds[0]:
        return "moderate"
    return "low"


def _merge_warning(seen: dict[str, LeakageWarning], w: LeakageWarning) -> None:
    """Keep highest-score warning per column, appending methods."""
    col = w["column"]
    if col not in seen or w["score"] > seen[col]["score"]:
        seen[col] = w
    else:
        # Append method name if same column already seen with different method
        existing = seen[col]
        if w["method"] not in existing["method"]:
            seen[col] = LeakageWarning(
                column=existing["column"],
                score=existing["score"],
                method=f"{existing['method']}+{w['method']}",
                severity=existing["severity"],
                explanation=existing["explanation"],
            )
