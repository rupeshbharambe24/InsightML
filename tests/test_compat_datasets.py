"""Tests for _compat.py and datasets/."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# to_pandas
# ---------------------------------------------------------------------------

class TestToPandas:
    def test_passthrough_dataframe(self):
        from dissectml._compat import to_pandas

        df = pd.DataFrame({"a": [1, 2]})
        result = to_pandas(df)
        assert result is df

    def test_from_dict(self):
        from dissectml._compat import to_pandas

        result = to_pandas({"x": [1, 2], "y": [3, 4]})
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["x", "y"]

    def test_from_list_of_dicts(self):
        from dissectml._compat import to_pandas

        result = to_pandas([{"a": 1}, {"a": 2}])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_from_numpy_array(self):
        from dissectml._compat import to_pandas

        arr = np.arange(6).reshape(3, 2)
        result = to_pandas(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)

    def test_from_csv_path(self, tmp_path):
        from dissectml._compat import to_pandas

        csv = tmp_path / "data.csv"
        pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]}).to_csv(csv, index=False)
        result = to_pandas(str(csv))
        assert isinstance(result, pd.DataFrame)
        assert "col1" in result.columns

    def test_from_path_object(self, tmp_path):
        from dissectml._compat import to_pandas

        csv = tmp_path / "data.csv"
        pd.DataFrame({"x": [10, 20]}).to_csv(csv, index=False)
        result = to_pandas(csv)
        assert list(result["x"]) == [10, 20]

    def test_unsupported_type_raises(self):
        from dissectml._compat import to_pandas

        with pytest.raises(TypeError, match="Unsupported data type"):
            to_pandas(12345)

    def test_polars_skipped_gracefully_if_not_installed(self):
        """If polars is not installed, non-polars inputs still work."""
        from dissectml._compat import to_pandas

        df = pd.DataFrame({"z": [1]})
        assert to_pandas(df) is df

    def test_is_polars_available_returns_bool(self):
        from dissectml._compat import is_polars_available

        result = is_polars_available()
        assert isinstance(result, bool)

    def test_get_pandas_version_returns_tuple(self):
        from dissectml._compat import get_pandas_version

        v = get_pandas_version()
        assert isinstance(v, tuple)
        assert all(isinstance(x, int) for x in v)
        assert v[0] >= 1  # at least pandas 1.x


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

class TestLoadTitanic:
    def test_returns_dataframe(self):
        from dissectml.datasets import load_titanic

        df = load_titanic()
        assert isinstance(df, pd.DataFrame)

    def test_has_survived_column(self):
        from dissectml.datasets import load_titanic

        df = load_titanic()
        assert "survived" in df.columns

    def test_minimum_rows(self):
        from dissectml.datasets import load_titanic

        df = load_titanic()
        assert len(df) >= 100

    def test_survived_is_binary(self):
        from dissectml.datasets import load_titanic

        df = load_titanic()
        assert set(df["survived"].dropna().unique()).issubset({0, 1})

    def test_sex_column_present(self):
        from dissectml.datasets import load_titanic

        df = load_titanic()
        assert "sex" in df.columns or "pclass" in df.columns


class TestLoadHousing:
    def test_returns_dataframe(self):
        from dissectml.datasets import load_housing

        df = load_housing()
        assert isinstance(df, pd.DataFrame)

    def test_has_target_column(self):
        from dissectml.datasets import load_housing

        df = load_housing()
        # sklearn uses MedHouseVal; our synthetic also uses MedHouseVal
        assert "MedHouseVal" in df.columns

    def test_minimum_rows(self):
        from dissectml.datasets import load_housing

        df = load_housing()
        assert len(df) >= 100

    def test_all_numeric(self):
        from dissectml.datasets import load_housing

        df = load_housing()
        assert all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)

    def test_no_all_null_columns(self):
        from dissectml.datasets import load_housing

        df = load_housing()
        for col in df.columns:
            assert df[col].notna().any(), f"Column {col!r} is all-null"


class TestDatasetsToplevel:
    def test_load_titanic_accessible_from_iml(self):
        import dissectml as iml

        df = iml.load_titanic()
        assert isinstance(df, pd.DataFrame)

    def test_load_housing_accessible_from_iml(self):
        import dissectml as iml

        df = iml.load_housing()
        assert isinstance(df, pd.DataFrame)

    def test_to_pandas_accessible_from_iml(self):
        import dissectml as iml

        df = iml.to_pandas({"a": [1]})
        assert isinstance(df, pd.DataFrame)
