"""Tests for viz.display — environment detection and display utilities."""
from __future__ import annotations

from dissectml.viz.display import (
    HTMLReprMixin,
    detect_environment,
    display_html,
    show_in_browser,
)

# ---------------------------------------------------------------------------
# detect_environment()
# ---------------------------------------------------------------------------

class TestDetectEnvironment:
    def test_returns_string(self):
        result = detect_environment()
        assert isinstance(result, str)

    def test_returns_expected_value(self):
        """In a test runner (not Jupyter), should return 'terminal' or 'vscode'."""
        result = detect_environment()
        assert result in ("jupyter", "colab", "vscode", "terminal")

    def test_terminal_when_no_ipython(self, monkeypatch):
        """Without IPython and without VSCODE_PID, should return 'terminal'."""
        monkeypatch.delenv("VSCODE_PID", raising=False)
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        result = detect_environment()
        assert result == "terminal"


# ---------------------------------------------------------------------------
# show_in_browser()
# ---------------------------------------------------------------------------

class TestShowInBrowser:
    def test_calls_webbrowser_open(self, monkeypatch, tmp_path):
        """show_in_browser should write a temp file and call webbrowser.open."""
        opened_urls = []
        monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))
        show_in_browser("<h1>Hello</h1>", title="Test")
        assert len(opened_urls) == 1
        assert opened_urls[0].startswith("file://")

    def test_html_content_written(self, monkeypatch, tmp_path):
        """The temp file should contain the HTML content."""
        opened_urls = []
        monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))
        show_in_browser("<p>Test content</p>", title="TestTitle")
        assert len(opened_urls) == 1
        # Extract file path from URL and read it
        url = opened_urls[0]
        # URL is file:///path/to/file.html
        file_path = url.replace("file://", "")
        # On Windows the path may start with / before the drive letter
        if file_path.startswith("/") and len(file_path) > 2 and file_path[2] == ":":
            file_path = file_path[1:]
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        assert "Test content" in content
        assert "TestTitle" in content


# ---------------------------------------------------------------------------
# HTMLReprMixin
# ---------------------------------------------------------------------------

class TestHTMLReprMixin:
    def test_repr_html_returns_string(self):
        obj = HTMLReprMixin()
        result = obj._repr_html_()
        assert isinstance(result, str)

    def test_repr_html_contains_pre_tag(self):
        obj = HTMLReprMixin()
        result = obj._repr_html_()
        assert "<pre>" in result

    def test_show_does_not_crash(self, monkeypatch):
        """show() calls display_html; monkeypatch to avoid browser open."""
        monkeypatch.setattr(
            "dissectml.viz.display.display_html", lambda html: None
        )
        obj = HTMLReprMixin()
        obj.show()


# ---------------------------------------------------------------------------
# display_html()
# ---------------------------------------------------------------------------

class TestDisplayHTML:
    def test_terminal_mode_calls_show_in_browser(self, monkeypatch):
        """In terminal mode, display_html should fall through to show_in_browser."""
        monkeypatch.setattr(
            "dissectml.viz.display.detect_environment", lambda: "terminal"
        )
        opened = []
        monkeypatch.setattr(
            "dissectml.viz.display.show_in_browser",
            lambda html, **kw: opened.append(html),
        )
        display_html("<div>Test</div>")
        assert len(opened) == 1
        assert "<div>Test</div>" in opened[0]

    def test_does_not_crash_in_terminal(self, monkeypatch):
        """display_html should not raise in terminal mode."""
        monkeypatch.setattr(
            "dissectml.viz.display.detect_environment", lambda: "terminal"
        )
        monkeypatch.setattr("webbrowser.open", lambda url: None)
        display_html("<p>Hello</p>")
