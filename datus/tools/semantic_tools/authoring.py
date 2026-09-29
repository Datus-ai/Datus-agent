# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Result types for Explorer's OSI YAML metric editor."""

from typing import List, Optional

from pydantic import BaseModel, Field

from .exceptions import SemanticCoreException


class AuthoringNotSupportedError(SemanticCoreException):
    """Retained for Explorer callers handling a missing editor capability."""


class MetricSource(BaseModel):
    """The source-of-truth YAML for a single metric.

    ``text`` is round-trippable through ``DosiRuntime.write_metric_source``.
    """

    name: str = Field(..., description="Metric name")
    format: str = Field(..., description="Source format ('osi')")
    text: str = Field(..., description="YAML text of the metric definition")
    semantic_model: Optional[str] = Field(None, description="Owning OSI semantic model name")
    file_path: Optional[str] = Field(None, description="Absolute path of the source file")


class MetricMutationResult(BaseModel):
    """Outcome of a write/delete so callers can re-sync only what changed."""

    name: str = Field(..., description="Metric name")
    format: str = Field(..., description="Source format ('osi')")
    file_path: str = Field(..., description="File that was written/removed from")
    semantic_model: Optional[str] = Field(None, description="Owning semantic model name (OSI)")
    created: bool = Field(False, description="True if the metric was newly created")
    deleted: bool = Field(False, description="True if the metric was removed")
    affected_paths: List[str] = Field(
        default_factory=list,
        description="Source files that changed; re-index these into the KB",
    )
