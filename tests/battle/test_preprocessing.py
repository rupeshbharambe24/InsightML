"""Tests for battle/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from insightml.battle.preprocessing import (
    PreprocessingPlan,
    build_full_pipeline,
    build_preprocessing_plan,
    build_transformer,
)


@pytest.fixture
def simple_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.integers(18, 80, 100).astype(float),
        "income": rng.normal(50000, 15000, 100),
        "city": rng.choice(["NYC", "LA", "CHI"], 100),
        "score": rng.uniform(0, 1, 100),
    })


@pytest.fixture
def df_with_missing(simple_df):
    df = simple_df.copy()
    df.loc[_rng_idx := np.random.default_rng(1).integers(0, 100, 20), "age"] = np.nan
    return df


class TestBuildPreprocessingPlan:
    def test_basic_column_detection(self, simple_df):
        plan = build_preprocessing_plan(simple_df)
        assert "age" in plan.numeric_cols
        assert "income" in plan.numeric_cols
        assert "score" in plan.numeric_cols
        # "city" has 3 unique values -> low cardinality -> categorical
        assert "city" in plan.categorical_cols
        assert len(plan.high_card_cols) == 0

    def test_excludes_target(self, simple_df):
        simple_df["target"] = 0
        plan = build_preprocessing_plan(simple_df, target="target")
        assert "target" not in plan.numeric_cols
        assert "target" not in plan.categorical_cols

    def test_default_scaler_standard(self, simple_df):
        plan = build_preprocessing_plan(simple_df)
        assert plan.scaler == "standard"

    def test_high_cardinality_detection(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({
            "num": rng.normal(0, 1, 200),
            "low_cat": rng.choice(["a", "b", "c"], 200),
            "high_cat": [f"cat_{i}" for i in rng.integers(0, 100, 200)],
        })
        plan = build_preprocessing_plan(df, high_cardinality_threshold=10)
        assert "low_cat" in plan.categorical_cols
        assert "high_cat" in plan.high_card_cols

    def test_knn_imputer_high_missing(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 100),
            "b": rng.normal(0, 1, 100),
        })
        # >20% missing in column b
        df.loc[:25, "b"] = np.nan
        plan = build_preprocessing_plan(df)
        assert plan.imputer == "knn"


class TestBuildTransformer:
    def test_returns_column_transformer(self, simple_df):
        plan = build_preprocessing_plan(simple_df)
        ct = build_transformer(plan)
        from sklearn.compose import ColumnTransformer
        assert isinstance(ct, ColumnTransformer)

    def test_fit_transform_works(self, simple_df):
        plan = build_preprocessing_plan(simple_df)
        ct = build_transformer(plan)
        X_out = ct.fit_transform(simple_df)
        assert X_out.shape[0] == len(simple_df)
        assert X_out.shape[1] > 0

    def test_tree_based_no_scaler(self, simple_df):
        plan = build_preprocessing_plan(simple_df)
        ct = build_transformer(plan, tree_based=True)
        # Fit and transform should still work
        X_out = ct.fit_transform(simple_df)
        assert X_out.shape[0] == len(simple_df)

    def test_empty_df_passthrough(self):
        plan = PreprocessingPlan()  # all empty lists
        ct = build_transformer(plan)
        from sklearn.compose import ColumnTransformer
        assert isinstance(ct, ColumnTransformer)


class TestBuildFullPipeline:
    def test_returns_pipeline(self, simple_df):
        from sklearn.ensemble import RandomForestClassifier
        plan = build_preprocessing_plan(simple_df)
        pipeline = build_full_pipeline(RandomForestClassifier(), plan)
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps.keys()) == ["preprocessor", "model"]

    def test_pipeline_fits(self, simple_df):
        from sklearn.linear_model import LogisticRegression
        plan = build_preprocessing_plan(simple_df)
        y = pd.Series(np.random.default_rng(0).integers(0, 2, len(simple_df)))
        pipeline = build_full_pipeline(LogisticRegression(max_iter=500), plan)
        pipeline.fit(simple_df, y)
        preds = pipeline.predict(simple_df)
        assert len(preds) == len(simple_df)
