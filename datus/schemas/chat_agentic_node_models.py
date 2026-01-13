# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Schema models for Chat Agentic Node.

This module defines the input and output models for the ChatAgenticNode,
providing structured validation for chat interactions with streaming support.
"""

from typing import Optional

from pydantic import AliasChoices, Field

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import Metric, ReferenceSql, TableSchema


class ChatNodeInput(BaseInput):
    """
    Input model for ChatAgenticNode interactions.
    """

    user_message: str = Field(..., description="User's chat message input")
    catalog: Optional[str] = Field(default=None, description="Database catalog for context")
    database: Optional[str] = Field(default=None, description="Database name for context")
    db_schema: Optional[str] = Field(default=None, description="Database schema for context")
    max_turns: int = Field(default=30, description="Maximum conversation turns per interaction")
    external_knowledge: Optional[str] = Field(default="", description="External knowledge")
    workspace_root: Optional[str] = Field(default=None, description="Root directory path for filesystem MCP server")
    prompt_version: Optional[str] = Field(default=None, description="Version for prompt template")
    schemas: Optional[list[TableSchema]] = Field(default=None, description="Schemas to use")
    metrics: Optional[list[Metric]] = Field(default=None, description="Metrics to use")
    reference_sql: Optional[list[ReferenceSql]] = Field(
        default=None,
        description="Reference SQL snippets to reuse/adjust",
        validation_alias=AliasChoices("reference_sql", "historical_sql"),
    )
    plan_mode: bool = Field(default=False, description="Whether this is a plan mode interaction")
    auto_execute_plan: bool = Field(
        default=False, description="Whether to auto-execute plan without user confirmation (for workflow/benchmark)"
    )

    class Config:
        populate_by_name = True


class ChatNodeResult(BaseResult):
    """
    Result model for ChatAgenticNode interactions.
    """

    response: str = Field(..., description="AI assistant's response")
    sql: Optional[str] = Field(default=None, description="SQL query generated or referenced in response")
    tokens_used: int = Field(default=0, description="Total tokens used in this interaction")
