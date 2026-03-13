# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Schema models for Explore Agentic Node.

This module defines the input and output models for the ExploreAgenticNode,
providing structured validation for read-only data exploration interactions
that gather context (schema, data samples, metrics, knowledge) before SQL generation.
"""

from typing import Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class ExploreNodeInput(BaseInput):
    """
    Input model for ExploreAgenticNode interactions.
    """

    user_message: str = Field(..., description="Exploration task description")
    database: Optional[str] = Field(default=None, description="Database name for context")


class ExploreNodeResult(BaseResult):
    """
    Result model for ExploreAgenticNode interactions.
    """

    response: str = Field(default="", description="Exploration result summary (plain text)")
    tokens_used: int = Field(default=0, description="Total tokens used in this interaction")
