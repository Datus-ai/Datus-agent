"""Configuration management for MCP MetricFlow Server."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class MetricFlowConfig(BaseModel):
    """Configuration for MetricFlow CLI."""

    # MetricFlow CLI path - updated to use the specific path
    mf_path: str = Field(default="/path/to/mf", description="Path to the mf CLI executable")

    # Project directory where MetricFlow project is located
    project_dir: Optional[Path] = Field(default=None, description="Path to the MetricFlow project directory")

    # Additional CLI flags
    verbose: bool = Field(default=False, description="Enable verbose output")

    @classmethod
    def from_env(cls) -> "MetricFlowConfig":
        """Create configuration from environment variables."""
        return cls(
            mf_path=os.getenv("MF_PATH", "/path/to/mf"),
            project_dir=Path(os.getenv("MF_PROJECT_DIR", ".")) if os.getenv("MF_PROJECT_DIR") else None,
            verbose=os.getenv("MF_VERBOSE", "false").lower() == "true",
        )
