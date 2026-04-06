"""Smart sampling strategies for large datasets."""

from __future__ import annotations

import pandas as pd

from dissectml._config import InsightMLConfig


def smart_sample(
    df: pd.DataFrame,
    target: str | None = None,
    config: InsightMLConfig | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Return a representative subsample if the DataFrame exceeds the size threshold.

    Sampling strategy (in priority order):
    1. Stratified — when target is provided (preserves class distribution)
    2. Temporal — when a datetime column is detected (preserves time ordering)
    3. Random — fallback with fixed random_state for reproducibility

    Args:
        df: Input DataFrame.
        target: Target column name (enables stratified sampling if given).
        config: InsightML configuration. Uses global config if None.
        force: If True, always sample even below threshold.

    Returns:
        The original DataFrame if small enough, otherwise a subsample.
    """
    if config is None:
        from dissectml._config import get_config
        config = get_config()

    n = len(df)
    if not force and n <= config.large_dataset_threshold:
        return df

    sample_size = min(config.sample_size, n)

    # --- Stratified sampling (classification target present) ---
    if target is not None and target in df.columns:
        target_col = df[target]
        if target_col.dtype in ("object", "category", "bool") or (
            target_col.nunique() <= 50
        ):
            try:
                return df.groupby(target, group_keys=False).apply(
                    lambda g: g.sample(
                        frac=sample_size / n,
                        random_state=config.random_state,
                    )
                ).reset_index(drop=True)
            except Exception:
                pass  # Fall through to temporal or random

    # --- Temporal sampling (datetime column present) ---
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    if datetime_cols:
        dt_col = datetime_cols[0]
        sorted_df = df.sort_values(dt_col)
        # Sample evenly spaced indices to preserve time distribution
        step = max(1, n // sample_size)
        indices = list(range(0, n, step))[:sample_size]
        return sorted_df.iloc[indices].reset_index(drop=True)

    # --- Random sampling (fallback) ---
    return df.sample(n=sample_size, random_state=config.random_state).reset_index(
        drop=True
    )
