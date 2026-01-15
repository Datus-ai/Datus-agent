# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Input and output models for ExtKnowledge generation agentic node.

This module defines the data models used for external knowledge generation workflow,
including input parameters and result structures.
"""

from typing import Optional

from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult


class ExtKnowledgeNodeInput(BaseInput):
    """Input model for external knowledge generation node."""

    user_message: str = Field(..., description="User's input message or request")

    # Optional fields for workflow mode (passed directly) or agentic mode (parsed via LLM)
    question: Optional[str] = Field(default=None, description="Business question extracted from user_message")
    gold_sql: Optional[str] = Field(
        default=None,
        description="Reference SQL answer. Accessed via get_gold_sql() tool, not exposed in prompt.",
    )

    search_text: Optional[str] = Field(default=None, description="Business search_text/concept to define")
    explanation: Optional[str] = Field(default=None, description="Existing explanation for the knowledge")
    subject_path: Optional[str] = Field(
        default=None, description="Subject path for classification (e.g., 'Finance/Revenue')"
    )
    prompt_version: Optional[str] = Field(default=None, description="Version for prompt template")
    prompt_language: Optional[str] = Field(default="en", description="Language for prompts (en/zh)")
    agent_description: Optional[str] = Field(default=None, description="Custom agent description")


class ExtKnowledgeNodeResult(BaseResult):
    """Result model for external knowledge generation node."""

    response: str = Field(..., description="AI assistant's response")
    ext_knowledge_file: Optional[str] = Field(
        default=None, description="Path to generated external knowledge YAML file"
    )
    tokens_used: int = Field(default=0, description="Total tokens used in generation")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")
