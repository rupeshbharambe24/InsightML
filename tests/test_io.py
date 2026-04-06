"""Tests for dissectml._io — file I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dissectml._io import SUPPORTED_EXTENSIONS, read_data
from dissectml.exceptions import UnsupportedFormatError

# ---------------------------------------------------------------------------
# read_data() — CSV
# ---------------------------------------------------------------------------

class TestReadCSV:
    """Tests for read_data with CSV files."""

    def test_csv_returns_dataframe(self, tmp_path: Path):
        """read_data() with a CSV file returns a DataFrame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
        df = read_data(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "c"]

    def test_csv_with_kwargs(self, tmp_path: Path):
        """read_data() forwards extra kwargs to the pandas reader."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a;b;c\n1;2;3\n4;5;6\n")
        df = read_data(csv_file, sep=";")
        assert list(df.columns) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# read_data() — TSV
# ---------------------------------------------------------------------------

class TestReadTSV:
    """Tests for read_data with TSV files."""

    def test_tsv_returns_dataframe(self, tmp_path: Path):
        """read_data() with a TSV file returns a DataFrame."""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")
        df = read_data(tsv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# read_data() — JSON
# ---------------------------------------------------------------------------

class TestReadJSON:
    """Tests for read_data with JSON files."""

    def test_json_returns_dataframe(self, tmp_path: Path):
        """read_data() with a JSON file returns a DataFrame."""
        json_file = tmp_path / "data.json"
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        json_file.write_text(json.dumps(data))
        df = read_data(json_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "a" in df.columns
        assert "b" in df.columns


# ---------------------------------------------------------------------------
# read_data() — error paths
# ---------------------------------------------------------------------------

class TestReadDataErrors:
    """Tests for read_data error handling."""

    def test_unsupported_extension_raises(self, tmp_path: Path):
        """read_data() raises UnsupportedFormatError for unknown extensions."""
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("hello")
        with pytest.raises(UnsupportedFormatError, match="Unsupported file format"):
            read_data(bad_file)

    def test_nonexistent_file_raises(self):
        """read_data() raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="not found"):
            read_data("/tmp/does_not_exist_12345.csv")

    def test_string_path_also_works(self, tmp_path: Path):
        """read_data() accepts a string path as well as a Path object."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y\n1,2\n")
        df = read_data(str(csv_file))
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS
# ---------------------------------------------------------------------------

class TestSupportedExtensions:
    """Tests for the SUPPORTED_EXTENSIONS constant."""

    @pytest.mark.parametrize("ext", [".csv", ".tsv", ".xlsx", ".parquet", ".json"])
    def test_contains_expected(self, ext: str):
        """SUPPORTED_EXTENSIONS contains the expected file extension."""
        assert ext in SUPPORTED_EXTENSIONS

    def test_is_list(self):
        """SUPPORTED_EXTENSIONS is a list."""
        assert isinstance(SUPPORTED_EXTENSIONS, list)
