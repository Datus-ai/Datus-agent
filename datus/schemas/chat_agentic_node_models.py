"""
Schema models for Chat Agentic Node.

This module defines the input and output models for the ChatAgenticNode,
providing structured validation for chat interactions with streaming support.
"""

from typing import Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class ChatNodeInput(BaseInput):
    """
    Input model for ChatAgenticNode interactions.
    """

    user_message: str = Field(..., description="User's chat message input")
    catalog: Optional[str] = Field(default=None, description="Database catalog for context")
    database: Optional[str] = Field(default=None, description="Database name for context")
    db_schema: Optional[str] = Field(default=None, description="Database schema for context")

    class Config:
        populate_by_name = True


class ChatNodeResult(BaseResult):
    """
    Result model for ChatAgenticNode interactions.
    """

    response: str = Field(..., description="AI assistant's response")
    sql: Optional[str] = Field(default=None, description="SQL query generated or referenced in response")
    tokens_used: int = Field(default=0, description="Total tokens used in this interaction")
