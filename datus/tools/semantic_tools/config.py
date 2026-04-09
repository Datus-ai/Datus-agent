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
    api_base_url: Optional[str] = Field(default=None, description="API base URL")
    auth_token: Optional[str] = Field(default=None, description="Auth token (JWT, API key, service token)")
    username: Optional[str] = Field(default=None, description="Username for basic auth or token exchange")
    password: Optional[str] = Field(
        default=None,
        description="Password for basic auth or token exchange",
        json_schema_extra={"input_type": "password"},
    )

    class Config:
        extra = "allow"  # Allow additional fields for adapter-specific config
