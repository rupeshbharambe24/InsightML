"""Tests for dissectml._lazy — optional dependency guard."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

from dissectml._lazy import is_available, require
from dissectml.exceptions import OptionalDependencyError


def _import_raising(name: str):
    """Simulate ImportError for a specific package name."""
    raise ImportError(f"No module named '{name}'")


# ---------------------------------------------------------------------------
# require()
# ---------------------------------------------------------------------------

class TestRequire:
    """Tests for the require() function."""

    def test_returns_module_for_installed_stdlib(self):
        """require() returns a module object for an installed stdlib package."""
        mod = require("json")
        assert hasattr(mod, "dumps")

    def test_returns_module_for_os(self):
        """require() returns the os module successfully."""
        mod = require("os")
        assert hasattr(mod, "path")

    def test_raises_for_nonexistent_package(self):
        """require() raises OptionalDependencyError for a missing package."""
        with pytest.raises(OptionalDependencyError):
            require("nonexistent_package_xyz_12345")

    def test_error_message_includes_install_hint_from_extra_map(self):
        """Error message includes the pip install hint from _EXTRA_MAP."""
        # xgboost may be installed, so we mock import_module to raise
        with patch.object(importlib, "import_module", side_effect=ImportError("no xgboost")):
            with pytest.raises(OptionalDependencyError, match=r"pip install dissectml\[boost\]"):
                require("xgboost")

    def test_error_message_for_unmapped_package(self):
        """Unmapped packages use the package name as the install hint."""
        with pytest.raises(OptionalDependencyError, match=r"pip install dissectml\[nonexistent_pkg\]"):
            require("nonexistent_pkg")

    def test_explicit_extra_overrides_map(self):
        """Providing extra= explicitly overrides the _EXTRA_MAP lookup."""
        with pytest.raises(OptionalDependencyError, match=r"pip install dissectml\[mygroup\]"):
            require("nonexistent_package_xyz_12345", extra="mygroup")

    def test_explicit_extra_with_pip_install_prefix(self):
        """If extra starts with 'pip install', it is used verbatim."""
        with pytest.raises(OptionalDependencyError, match=r"pip install openpyxl"):
            require("openpyxl_fake_not_installed", extra="pip install openpyxl")

    def test_extra_map_entry_with_pip_install_prefix(self):
        """_EXTRA_MAP entries with 'pip install' prefix are used verbatim."""
        # openpyxl maps to ("openpyxl", "pip install openpyxl") in _EXTRA_MAP
        with patch.object(importlib, "import_module", side_effect=ImportError("no openpyxl")):
            with pytest.raises(OptionalDependencyError, match=r"pip install openpyxl"):
                require("openpyxl")


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------

class TestIsAvailable:
    """Tests for the is_available() function."""

    def test_returns_true_for_json(self):
        """is_available() returns True for the json stdlib module."""
        assert is_available("json") is True

    def test_returns_true_for_os(self):
        """is_available() returns True for the os stdlib module."""
        assert is_available("os") is True

    def test_returns_false_for_nonexistent(self):
        """is_available() returns False for a package that does not exist."""
        assert is_available("nonexistent_package_xyz") is False


# ---------------------------------------------------------------------------
# OptionalDependencyError import
# ---------------------------------------------------------------------------

class TestOptionalDependencyError:
    """Verify the exception can be imported from dissectml.exceptions."""

    def test_is_exception_subclass(self):
        assert issubclass(OptionalDependencyError, Exception)

    def test_inherits_from_dependency_error(self):
        from dissectml.exceptions import DependencyError
        assert issubclass(OptionalDependencyError, DependencyError)
