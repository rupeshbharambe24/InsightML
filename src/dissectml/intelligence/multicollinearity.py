"""Multicollinearity detection — VIF, condition number, eigenvalue analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_VIF_MODERATE = 5.0
_VIF_HIGH = 10.0
_CONDITION_MODERATE = 30.0
_CONDITION_HIGH = 100.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_vif(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute Variance Inflation Factors for each numeric column.

    VIF for feature X_i = 1 / (1 - R²_i), where R²_i is the R² from
    regressing X_i on all other features.

    Args:
        df: Feature DataFrame (no target column).
        numeric_cols: Columns to include. Defaults to all numeric columns.

    Returns:
        DataFrame with columns: feature, vif, severity
        (sorted by vif descending).
    """
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Drop columns with zero variance or all-null
    valid_cols = [
        c for c in numeric_cols
        if df[c].notna().sum() >= 5 and df[c].std() > 1e-10
    ]

    if len(valid_cols) < 2:
        return pd.DataFrame(columns=["feature", "vif", "severity"])

    # Build numeric matrix (median-impute missing)
    X = df[valid_cols].copy()
    for col in valid_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    X_arr = X.values.astype(float)
    n, p = X_arr.shape

    rows = []
    for i, col in enumerate(valid_cols):
        y_i = X_arr[:, i]
        X_others = np.delete(X_arr, i, axis=1)
        # Add intercept
        X_design = np.column_stack([np.ones(n), X_others])
        try:
            # OLS via normal equations with pseudo-inverse for stability
            coef, _, _, _ = np.linalg.lstsq(X_design, y_i, rcond=None)
            y_hat = X_design @ coef
            ss_res = float(np.sum((y_i - y_hat) ** 2))
            ss_tot = float(np.sum((y_i - y_i.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            r2 = max(0.0, min(r2, 1.0 - 1e-9))  # clamp to avoid div-by-zero
            vif = 1.0 / (1.0 - r2)
        except Exception:
            vif = float("inf")

        rows.append({
            "feature": col,
            "vif": round(vif, 3),
            "severity": _vif_severity(vif),
        })

    result = pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)
    return result


def compute_condition_number(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Compute the condition number of the feature matrix.

    A high condition number indicates multicollinearity.
    - < 30: low
    - 30–100: moderate
    - > 100: severe

    Args:
        df: Feature DataFrame (no target column).
        numeric_cols: Columns to include. Defaults to all numeric columns.

    Returns:
        Dict with keys: condition_number, severity, eigenvalues, n_near_zero.
    """
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    valid_cols = [
        c for c in numeric_cols
        if df[c].notna().sum() >= 5 and df[c].std() > 1e-10
    ]

    if len(valid_cols) < 2:
        return {"condition_number": None, "severity": "unknown", "eigenvalues": [], "n_near_zero": 0}

    X = df[valid_cols].copy()
    for col in valid_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Standardise to unit variance before computing eigenvalues
    X_arr = X.values.astype(float)
    X_std = (X_arr - X_arr.mean(axis=0)) / (X_arr.std(axis=0) + 1e-10)

    try:
        eigenvalues = np.linalg.eigvalsh(X_std.T @ X_std)
        eigenvalues = np.sort(eigenvalues)[::-1]
        max_ev = float(eigenvalues[0])
        min_ev = float(np.abs(eigenvalues[-1]))
        condition_number = float(np.sqrt(max_ev / max(min_ev, 1e-10)))
        n_near_zero = int(np.sum(eigenvalues < 1e-6))
    except Exception:
        return {"condition_number": None, "severity": "unknown", "eigenvalues": [], "n_near_zero": 0}

    return {
        "condition_number": round(condition_number, 3),
        "severity": _condition_severity(condition_number),
        "eigenvalues": [round(float(e), 6) for e in eigenvalues],
        "n_near_zero": n_near_zero,
    }


def removal_recommendations(
    vif_df: pd.DataFrame,
    df: pd.DataFrame,
    target: str | None = None,
    vif_threshold: float = _VIF_HIGH,
) -> list[dict[str, Any]]:
    """Recommend which collinear features to remove.

    For each pair of high-VIF features, recommends removing the one with
    lower absolute correlation with the target (when target is provided),
    otherwise the one with higher VIF.

    Args:
        vif_df: Output of :func:`compute_vif`.
        df: Full DataFrame including target.
        target: Target column name.
        vif_threshold: VIF >= this triggers a recommendation.

    Returns:
        List of dicts: {feature, vif, recommendation, reason}
    """
    high_vif = vif_df[vif_df["vif"] >= vif_threshold]["feature"].tolist()
    if not high_vif:
        return []

    recommendations = []
    for col in high_vif:
        rec: dict[str, Any] = {
            "feature": col,
            "vif": float(vif_df.loc[vif_df["feature"] == col, "vif"].iloc[0]),
        }

        if target and target in df.columns:
            try:
                y_enc = _encode_target(df[target])
                if y_enc is not None:
                    mask = df[col].notna() & y_enc.notna()
                    corr_with_target = abs(float(df.loc[mask, col].corr(y_enc[mask])))
                    rec["corr_with_target"] = round(corr_with_target, 4)
                    rec["recommendation"] = "consider_removing" if corr_with_target < 0.1 else "keep_if_needed"
                    rec["reason"] = (
                        f"VIF={rec['vif']:.1f}; corr_with_target={corr_with_target:.3f}. "
                        + ("Low target correlation — likely redundant." if corr_with_target < 0.1
                           else "Has some target correlation — use domain knowledge.")
                    )
                else:
                    rec["recommendation"] = "consider_removing"
                    rec["reason"] = f"VIF={rec['vif']:.1f} — highly collinear with other features."
            except Exception:
                rec["recommendation"] = "consider_removing"
                rec["reason"] = f"VIF={rec['vif']:.1f}"
        else:
            rec["recommendation"] = "consider_removing"
            rec["reason"] = f"VIF={rec['vif']:.1f} — highly collinear."

        recommendations.append(rec)

    return recommendations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vif_severity(vif: float) -> str:
    if vif >= _VIF_HIGH:
        return "high"
    if vif >= _VIF_MODERATE:
        return "moderate"
    return "low"


def _condition_severity(cn: float) -> str:
    if cn >= _CONDITION_HIGH:
        return "severe"
    if cn >= _CONDITION_MODERATE:
        return "moderate"
    return "low"


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
