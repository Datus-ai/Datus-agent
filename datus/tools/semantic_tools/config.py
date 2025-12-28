# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Optional

from pydantic import BaseModel, Field


class SemanticAdapterConfig(BaseModel):
    """Base configuration for semantic adapters."""

    namespace: Optional[str] = Field(default=None, description="Datus namespace for configuration")
    timeout_seconds: int = Field(default=30, description="Operation timeout in seconds")

    class Config:
        extra = "allow"  # Allow additional fields for adapter-specific config


class MetricFlowConfig(SemanticAdapterConfig):
    """MetricFlow-specific configuration."""

    project_root: Optional[str] = Field(default=None, description="MetricFlow project root directory")
    cli_path: str = Field(default="mf", description="Path to MetricFlow CLI command")


class DbtConfig(SemanticAdapterConfig):
    """dbt-specific configuration."""

    project_dir: str = Field(..., description="dbt project directory")
    profiles_dir: Optional[str] = Field(default=None, description="dbt profiles directory")
    target: Optional[str] = Field(default=None, description="dbt target name")


class CubeConfig(SemanticAdapterConfig):
    """Cube.js-specific configuration."""

    api_url: str = Field(..., description="Cube API URL")
    api_token: Optional[str] = Field(default=None, description="Cube API authentication token")
