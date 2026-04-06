"""InsightML exception hierarchy."""

from __future__ import annotations


class InsightMLError(Exception):
    """Base exception for all InsightML errors."""


# --- Input / Validation errors ---

class ValidationError(InsightMLError):
    """Input data or parameter failed validation."""


class EmptyDataFrameError(ValidationError):
    """DataFrame has no rows or no columns."""


class TargetNotFoundError(ValidationError):
    """Specified target column does not exist in the DataFrame."""


class UnsupportedFormatError(ValidationError):
    """File format is not supported."""


class InvalidTaskError(ValidationError):
    """Task type is not 'classification', 'regression', or 'auto'."""


# --- Dependency errors ---

class DependencyError(InsightMLError):
    """A required dependency is missing."""


class OptionalDependencyError(DependencyError):
    """An optional extra dependency is missing. Install the relevant extra."""


# --- Computation errors ---

class ComputationError(InsightMLError):
    """An error occurred during computation."""


class ModelTrainingError(ComputationError):
    """A model failed during training."""


class StageTimeoutError(ComputationError):
    """A pipeline stage or model exceeded its timeout."""


# --- Report errors ---

class ReportError(InsightMLError):
    """An error occurred during report generation."""


class TemplateError(ReportError):
    """A Jinja2 template could not be rendered."""


class ExportError(ReportError):
    """Report could not be exported to the requested format."""
