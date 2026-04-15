# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Scheduler Agentic Node Models

Input and output models for the SchedulerAgenticNode,
which manages job scheduling operations (Airflow).
"""

from typing import Any, Dict, Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class SchedulerNodeInput(BaseInput):
    """Input model for SchedulerAgenticNode."""

    user_message: str = Field(..., description="User's scheduler request (required)")
    database: Optional[str] = Field(None, description="Source database namespace")
    prompt_version: Optional[str] = Field(None, description="Prompt template version")


class SchedulerNodeResult(BaseResult):
    """Result model for SchedulerAgenticNode."""

    response: str = Field(default="", description="Natural language response/summary")
    scheduler_result: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured scheduler result data (job_id, status, etc.)",
    )
    tokens_used: int = Field(default=0, description="Total tokens used")
