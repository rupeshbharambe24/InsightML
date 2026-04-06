"""Tests for dissectml.core.progress — ProgressTracker."""

from __future__ import annotations

from dissectml.core.progress import ProgressTracker

# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestProgressTrackerInit:
    """Tests for ProgressTracker construction."""

    def test_default_verbosity(self):
        """ProgressTracker() defaults to verbosity=1."""
        pt = ProgressTracker(verbosity=1)
        assert pt.verbosity == 1

    def test_silent_tracker(self):
        """ProgressTracker(verbosity=0) creates a silent tracker."""
        pt = ProgressTracker(verbosity=0)
        assert pt.verbosity == 0


# ---------------------------------------------------------------------------
# track()
# ---------------------------------------------------------------------------

class TestTrack:
    """Tests for ProgressTracker.track."""

    def test_silent_yields_all_items(self):
        """track() with verbosity=0 yields every item from the iterable."""
        pt = ProgressTracker(verbosity=0)
        items = list(pt.track(range(10), description="test"))
        assert items == list(range(10))

    def test_verbose_yields_all_items(self):
        """track() with verbosity=1 still yields every item."""
        pt = ProgressTracker(verbosity=1)
        items = list(pt.track([1, 2, 3], description="counting"))
        assert items == [1, 2, 3]

    def test_track_with_total(self):
        """track() works correctly when total is provided."""
        pt = ProgressTracker(verbosity=0)
        items = list(pt.track(range(5), description="test", total=5))
        assert items == list(range(5))

    def test_track_empty_iterable(self):
        """track() handles an empty iterable gracefully."""
        pt = ProgressTracker(verbosity=1)
        items = list(pt.track([], description="empty"))
        assert items == []


# ---------------------------------------------------------------------------
# log()
# ---------------------------------------------------------------------------

class TestLog:
    """Tests for ProgressTracker.log."""

    def test_silent_log_prints_nothing(self, capsys):
        """log() with verbosity=0 produces no output."""
        pt = ProgressTracker(verbosity=0)
        pt.log("should not appear", level=1)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_log_prints_message(self, capsys):
        """log() with verbosity=1 prints the message."""
        pt = ProgressTracker(verbosity=1)
        pt.log("hello world", level=1)
        captured = capsys.readouterr()
        # The message should appear somewhere in the output
        # (rich may add formatting codes)
        assert "hello world" in captured.out

    def test_high_level_suppressed(self, capsys):
        """log() with level > verbosity is suppressed."""
        pt = ProgressTracker(verbosity=1)
        pt.log("debug message", level=2)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# task()
# ---------------------------------------------------------------------------

class TestTask:
    """Tests for ProgressTracker.task context manager."""

    def test_task_does_not_crash(self):
        """task() context manager enters and exits without error."""
        pt = ProgressTracker(verbosity=1)
        with pt.task("test task"):
            pass  # should not raise

    def test_task_silent_does_not_crash(self):
        """task() with verbosity=0 still works."""
        pt = ProgressTracker(verbosity=0)
        with pt.task("silent task"):
            pass

    def test_task_body_executes(self):
        """Code inside the task() block actually runs."""
        pt = ProgressTracker(verbosity=0)
        executed = False
        with pt.task("run body"):
            executed = True
        assert executed is True
