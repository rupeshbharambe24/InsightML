"""Core infrastructure: base classes, pipeline, data container."""

from dissectml.core.base import BaseStage, PipelineContext, StageResult
from dissectml.core.data_container import DataContainer

__all__ = ["BaseStage", "PipelineContext", "StageResult", "DataContainer"]
