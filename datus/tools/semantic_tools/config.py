# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Base configuration for semantic adapters.

Specific adapter configurations (MetricFlowConfig, DbtConfig, CubeConfig, etc.)
should be defined in their respective adapter packages.
"""

from typing import Optional

from pydantic import BaseModel, Field


class SemanticAdapterConfig(BaseModel):
    """Base configuration for semantic adapters."""

    namespace: Optional[str] = Field(default=None, description="Datus namespace for configuration")
    timeout_seconds: int = Field(default=30, description="Operation timeout in seconds")

    class Config:
        extra = "allow"  # Allow additional fields for adapter-specific config
