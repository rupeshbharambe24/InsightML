"""Optional dependency guard for InsightML extras."""

from __future__ import annotations

import importlib
from types import ModuleType

from dissectml.exceptions import OptionalDependencyError

# Maps package import name -> (install name, extra group)
_EXTRA_MAP: dict[str, tuple[str, str]] = {
    "xgboost": ("xgboost", "boost"),
    "lightgbm": ("lightgbm", "boost"),
    "catboost": ("catboost", "boost"),
    "shap": ("shap", "explain"),
    "weasyprint": ("weasyprint", "report"),
    "kaleido": ("kaleido", "report"),
    "polars": ("polars", "scale"),
    "optuna": ("optuna", "scale"),
    "openpyxl": ("openpyxl", "pip install openpyxl"),
    "pyarrow": ("pyarrow", "pip install pyarrow"),
}


def require(package_name: str, extra: str | None = None) -> ModuleType:
    """Import an optional package, raising a helpful error if missing.

    Args:
        package_name: The import name of the package (e.g., "xgboost").
        extra: The dissectml extra group to suggest (e.g., "boost").
               If None, inferred from _EXTRA_MAP or shown as bare install.

    Returns:
        The imported module.

    Raises:
        OptionalDependencyError: If the package is not installed.
    """
    try:
        return importlib.import_module(package_name)
    except ImportError:
        if extra is None:
            entry = _EXTRA_MAP.get(package_name)
            extra = entry[1] if entry else package_name
        if extra.startswith("pip install"):
            install_hint = extra
        else:
            install_hint = f"pip install dissectml[{extra}]"
        raise OptionalDependencyError(
            f"Optional dependency '{package_name}' is not installed. "
            f"Install it with: {install_hint}"
        ) from None


def is_available(package_name: str) -> bool:
    """Check if an optional package is importable (no error raised)."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False
