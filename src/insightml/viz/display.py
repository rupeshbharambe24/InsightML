"""Environment detection and display utilities for InsightML."""

from __future__ import annotations

import os
import tempfile
import webbrowser
from pathlib import Path


def detect_environment() -> str:
    """Detect the current Python execution environment.

    Returns:
        One of: 'jupyter', 'colab', 'vscode', 'terminal'
    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        if shell == "ZMQInteractiveShell":
            # Check Colab first (Colab also uses ZMQ)
            try:
                import google.colab  # type: ignore[import]  # noqa: F401
                return "colab"
            except ImportError:
                pass
            return "jupyter"
        if shell == "TerminalInteractiveShell":
            return "terminal"
    except NameError:
        pass

    if "VSCODE_PID" in os.environ or "TERM_PROGRAM" in os.environ and os.environ.get("TERM_PROGRAM") == "vscode":
        return "vscode"

    return "terminal"


def display_html(html: str) -> None:
    """Display HTML in the current environment.

    - Jupyter/Colab: uses IPython.display.HTML (inline rendering)
    - VS Code: uses IPython.display.HTML if available, else browser
    - Terminal: opens a temporary file in the default browser
    """
    env = detect_environment()

    if env in ("jupyter", "colab", "vscode"):
        try:
            from IPython.display import HTML, display
            display(HTML(html))
            return
        except ImportError:
            pass

    # Terminal fallback: open in browser
    show_in_browser(html)


def show_in_browser(html: str, title: str = "InsightML") -> None:
    """Write html to a temp file and open it in the default web browser."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(f"<!DOCTYPE html><html><head><title>{title}</title></head><body>{html}</body></html>")
        tmp_path = f.name
    webbrowser.open(f"file://{Path(tmp_path).as_posix()}")


class HTMLReprMixin:
    """Mixin that provides ``_repr_html_()`` and ``show()`` for result classes."""

    def _repr_html_(self) -> str:
        """Return HTML string for Jupyter rich display. Override in subclasses."""
        return f"<pre>{self!r}</pre>"

    def show(self) -> None:
        """Display this object — Jupyter inline or browser fallback."""
        display_html(self._repr_html_())
