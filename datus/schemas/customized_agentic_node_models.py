"""
Schema models for Customized Agentic Node.

This module defines the input and output models for the CustomizedAgenticNode,
providing structured validation for flexible, configuration-based interactions
with streaming support.
"""

from typing import Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import HistoricalSql, Metric, TableSchema


class CustomizedNodeInput(BaseInput):
    """
    Input model for CustomizedAgenticNode interactions.
    """

    user_message: str = Field(..., description="User's input message")
    catalog: Optional[str] = Field(default=None, description="Database catalog for context")
    database: Optional[str] = Field(default=None, description="Database name for context")
    db_schema: Optional[str] = Field(default=None, description="Database schema for context")
    max_turns: int = Field(default=30, description="Maximum conversation turns per interaction")
    workspace_root: Optional[str] = Field(default=None, description="Root directory path for filesystem MCP server")
    prompt_version: Optional[str] = Field(default=None, description="Version for prompt template")
    prompt_language: Optional[str] = Field(default="en", description="Language for prompt template")
    schemas: Optional[list[TableSchema]] = Field(default=None, description="Table schemas to use")
    metrics: Optional[list[Metric]] = Field(default=None, description="Metrics to use")
    historical_sql: Optional[list[HistoricalSql]] = Field(default=None, description="Historical SQL to reference")
    agent_description: Optional[str] = Field(default=None, description="Custom agent description override")
    custom_rules: Optional[list[str]] = Field(default=None, description="Additional custom rules for this interaction")

    class Config:
        populate_by_name = True


class CustomizedNodeResult(BaseResult):
    """
    Result model for CustomizedAgenticNode interactions.
    """

    response: str = Field(..., description="AI assistant's response")
    sql: Optional[str] = Field(default=None, description="SQL query generated or referenced in response")
    tokens_used: int = Field(default=0, description="Total tokens used in this interaction")
    error: Optional[str] = Field(default=None, description="Error message if interaction failed")
