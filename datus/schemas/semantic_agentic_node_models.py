# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Schema models for Semantic Agentic Node.

This module defines the input and output models for the SemanticAgenticNode,
providing structured validation for semantic model generation interactions.
"""

from typing import List, Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class SemanticNodeInput(BaseInput):
    """
    Input model for SemanticAgenticNode interactions.
    """

    user_message: str = Field(..., description="User's input message")
    catalog: Optional[str] = Field(default=None, description="Database catalog for context")
    database: Optional[str] = Field(default=None, description="Database name for context")
    db_schema: Optional[str] = Field(default=None, description="Database schema for context")
    max_turns: int = Field(default=30, description="Maximum conversation turns per interaction")
    workspace_root: Optional[str] = Field(default=None, description="Root directory path for filesystem MCP server")
    prompt_version: Optional[str] = Field(default=None, description="Version for prompt template")
    prompt_language: Optional[str] = Field(default="en", description="Language for prompt template")
    agent_description: Optional[str] = Field(default=None, description="Custom agent description override")
    custom_rules: Optional[list[str]] = Field(default=None, description="Additional custom rules for this interaction")

    # Configuration fields from agent.yml
    system_prompt: Optional[str] = Field(default=None, description="System prompt type identifier")
    tools: Optional[str] = Field(default=None, description="Tools configuration pattern")
    mcp: Optional[str] = Field(default=None, description="MCP server configuration pattern")
    rules: Optional[list[str]] = Field(default=None, description="Configuration rules for the node")

    class Config:
        populate_by_name = True


class SemanticNodeResult(BaseResult):
    """
    Result model for SemanticAgenticNode interactions.
    """

    response: str = Field(..., description="AI assistant's response")
    semantic_models: List[str] = Field(
        default_factory=list, description="List of generated semantic model file paths (single table or multi-table)"
    )
    tokens_used: int = Field(default=0, description="Total tokens used in this interaction")
    error: Optional[str] = Field(default=None, description="Error message if interaction failed")
