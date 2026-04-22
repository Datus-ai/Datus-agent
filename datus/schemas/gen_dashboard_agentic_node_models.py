# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenDashboard Agentic Node Models

Input and output models for the GenDashboardAgenticNode,
which manages BI dashboard operations (Superset, Grafana).
"""

from typing import Any, Dict, Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class GenDashboardNodeInput(BaseInput):
    """Input model for GenDashboardAgenticNode."""

    user_message: str = Field(..., description="User's dashboard request (required)")
    database: Optional[str] = Field(None, description="Source datasource")
    prompt_version: Optional[str] = Field(None, description="Prompt template version")


class GenDashboardNodeResult(BaseResult):
    """Result model for GenDashboardAgenticNode."""

    response: str = Field(default="", description="Natural language response/summary")
    dashboard_result: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured dashboard result data (dashboard_id, charts created, etc.)",
    )
    tokens_used: int = Field(default=0, description="Total tokens used")
