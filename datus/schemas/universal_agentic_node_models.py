# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Schema models for Universal Agentic Node.

Defines the result model for UniversalAgenticNode, which uses SemanticNodeInput
for input and a dedicated UniversalNodeResult for output.
"""

from pydantic import Field

from datus.schemas.base import BaseResult


class UniversalNodeResult(BaseResult):
    """
    Result model for UniversalAgenticNode interactions.
    """

    output: str = Field(default="", description="AI assistant's output content")
    tokens_used: int = Field(default=0, description="Total tokens used in this interaction")
