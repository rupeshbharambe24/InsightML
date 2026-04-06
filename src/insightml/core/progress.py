"""Progress tracking — auto-detects notebook vs terminal environment."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")


class ProgressTracker:
    """Wraps tqdm or rich.progress depending on environment.

    - Jupyter/Colab: uses tqdm.notebook for inline progress bars
    - Terminal: uses rich.progress for styled console output
    - Silent (verbosity=0): no output
    """

    def __init__(self, verbosity: int = 1) -> None:
        self.verbosity = verbosity
        self._env = self._detect_env()

    @staticmethod
    def _detect_env() -> str:
        from insightml.viz.display import detect_environment
        return detect_environment()

    def track(
        self,
        iterable: Iterable[T],
        description: str = "Processing",
        total: int | None = None,
    ) -> Iterator[T]:
        """Wrap an iterable with a progress bar.

        Args:
            iterable: Items to iterate over.
            description: Label shown next to the progress bar.
            total: Total count (for accurate percentage display).

        Yields:
            Items from the iterable.
        """
        if self.verbosity == 0:
            yield from iterable
            return

        try:
            if self._env in ("jupyter", "colab"):
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
            yield from tqdm(iterable, desc=description, total=total, leave=False)
        except ImportError:
            # tqdm not available — iterate silently
            yield from iterable

    def log(self, message: str, level: int = 1) -> None:
        """Print a message if verbosity >= level.

        Args:
            message: Text to print.
            level: Minimum verbosity level to show this message.
        """
        if self.verbosity >= level:
            try:
                from rich import print as rprint
                rprint(f"[dim]{message}[/dim]")
            except ImportError:
                print(message)

    @contextmanager
    def task(self, description: str) -> Generator[None, None, None]:
        """Context manager that logs start/end of a named task."""
        self.log(f"[→] {description}", level=1)
        yield
        self.log(f"[✓] {description}", level=2)
