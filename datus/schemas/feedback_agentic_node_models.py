# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Input and output models for Feedback agentic node.

This module defines the data models used for the feedback workflow,
which analyzes conversation history and archives knowledge, SQL patterns,
metrics, and skills via sub-agents.
"""

from typing import Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class FeedbackNodeInput(BaseInput):
    """Input model for feedback node."""

    user_message: str = Field(..., description="Feedback instruction (e.g., 'analyze and archive')")
    source_session_id: Optional[str] = Field(default=None, description="Session to copy and analyze")
    database: Optional[str] = Field(default=None, description="Database name for context")


class FeedbackNodeResult(BaseResult):
    """Result model for feedback node."""

    response: str = Field(..., description="AI assistant's feedback summary response")
    items_saved: int = Field(default=0, description="Number of items saved to knowledge base")
    storage_summary: Optional[dict[str, int]] = Field(
        default=None, description="Summary of what was archived by category {knowledge: N, sql_pattern: N, ...}"
    )
    tokens_used: int = Field(default=0, description="Total tokens used in generation")
