# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenReport Agentic Node Models

This module defines the input and output models for the GenReportAgenticNode,
which provides a generic base for report generation nodes with semantic and database tools.
"""

from typing import Any, Dict, Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class GenReportNodeInput(BaseInput):
    """
    Input model for GenReportAgenticNode.

    Provides common fields for report generation nodes. Subclasses can extend
    this with additional fields specific to their report type.
    """

    user_message: str = Field(..., description="User's analysis question (required)")

    # Database context
    catalog: Optional[str] = Field(None, description="Database catalog")
    database: Optional[str] = Field(None, description="Database name")
    db_schema: Optional[str] = Field(None, description="Database schema")

    # Prompt configuration
    prompt_version: Optional[str] = Field(None, description="Prompt template version")


class GenReportNodeResult(BaseResult):
    """
    Result model for GenReportAgenticNode.

    Contains the response text, optional structured report result, and execution metadata.
    """

    response: str = Field(default="", description="Natural language response/summary")
    report_result: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured report result data (specific to report type)",
    )
    tokens_used: int = Field(default=0, description="Total tokens used in the analysis")
