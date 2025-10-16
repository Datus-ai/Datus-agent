# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Input and output models for SQL Summary generation agentic node.

This module defines the data models used for SQL summary generation workflow,
including input parameters and result structures.
"""

from typing import Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class SqlSummaryNodeInput(BaseInput):
    """Input model for SQL summary generation node."""

    user_message: str = Field(..., description="User's input message or request")
    sql_query: Optional[str] = Field(default=None, description="SQL query to summarize")
    comment: Optional[str] = Field(default=None, description="Existing comment or description for the SQL")
    catalog: Optional[str] = Field(default=None, description="Database catalog for context")
    database: Optional[str] = Field(default=None, description="Database name for context")
    db_schema: Optional[str] = Field(default=None, description="Database schema for context")
    prompt_version: Optional[str] = Field(default=None, description="Version for prompt template")
    prompt_language: Optional[str] = Field(default="en", description="Language for prompts (en/zh)")
    agent_description: Optional[str] = Field(default=None, description="Custom agent description")


class SqlSummaryNodeResult(BaseResult):
    """Result model for SQL summary generation node."""

    response: str = Field(..., description="AI assistant's response")
    sql_summary_file: Optional[str] = Field(default=None, description="Path to generated SQL summary YAML file")
    tokens_used: int = Field(default=0, description="Total tokens used in generation")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")
