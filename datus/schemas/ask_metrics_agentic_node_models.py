# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""AskMetrics Agentic Node models."""

from typing import Any, Dict, Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class AskMetricsNodeInput(BaseInput):
    """Input model for AskMetricsAgenticNode."""

    user_message: str = Field(..., description="User's metric question")
    catalog: Optional[str] = Field(None, description="Database catalog")
    database: Optional[str] = Field(None, description="Database name")
    db_schema: Optional[str] = Field(None, description="Database schema")
    max_turns: int = Field(default=12, description="Maximum turns for quick metric answering")
    prompt_version: Optional[str] = Field(None, description="Prompt template version override")


class AskMetricsNodeResult(BaseResult):
    """Result model for AskMetricsAgenticNode."""

    response: str = Field(default="", description="Markdown metric answer")
    markdown_report: str = Field(default="", description="Final Markdown report")
    report_result: Optional[Dict[str, Any]] = Field(default=None, description="Optional structured metadata")
    tokens_used: int = Field(default=0, description="Total tokens used")
