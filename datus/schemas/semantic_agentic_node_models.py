# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Schema models for Semantic Agentic Node.

This module defines the input and output models for the SemanticAgenticNode,
providing structured validation for semantic model generation interactions.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from datus.schemas.at_context import AtContextInput
from datus.schemas.base import BaseResult


class SourceQueryEvidence(BaseModel):
    """Structured success-story SQL carried independently from the LLM prompt."""

    source_sql_name: str = Field(..., description="Stable source-query name, for example sql_1")
    sql: str = Field(..., description="Original SQL from the success-story row")
    question: str = Field(default="", description="Business question associated with the SQL")
    external_knowledge: str = Field(default="", description="Optional row-scoped business evidence")
    source_id: str = Field(default="", description="Optional provenance source identifier")
    source_type: str = Field(default="success_story", description="Optional provenance source type")
    source_context_ids: List[str] = Field(default_factory=list, description="Optional provenance context IDs")
    source_metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional provenance metadata")


class SemanticNodeInput(AtContextInput):
    """
    Input model for SemanticAgenticNode interactions.
    """

    user_message: str = Field(..., description="User's input message")
    catalog: Optional[str] = Field(default=None, description="Database catalog for context")
    database: Optional[str] = Field(default=None, description="Database name for context")
    db_schema: Optional[str] = Field(default=None, description="Database schema for context")
    semantic_model_name: Optional[str] = Field(
        default=None,
        description="Explicit stable semantic model name; takes priority over inferred naming in Ossie mode",
    )
    semantic_model_file: Optional[str] = Field(
        default=None,
        description="Optional semantic model file hint; the agent must verify it before use",
    )
    business_domain: Optional[str] = Field(
        default=None,
        description="Business domain used to name a new Ossie semantic model when no explicit name is supplied",
    )
    fact_tables: Optional[list[str]] = Field(
        default=None,
        description="Fact tables in priority order; the first/core fact table is the stable naming fallback",
    )
    dimension_tables: Optional[list[str]] = Field(
        default=None,
        description="Dimension tables used by the model; recorded for context but excluded from model naming",
    )
    max_turns: Optional[int] = Field(default=None, description="Maximum conversation turns; None uses node config")
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

    model_config = ConfigDict(populate_by_name=True)


class SemanticNodeResult(BaseResult):
    """
    Result model for SemanticAgenticNode interactions.
    """

    response: str = Field(..., description="AI assistant's response")
    semantic_models: List[str] = Field(
        default_factory=list, description="List of generated semantic model file paths (single table or multi-table)"
    )
    tokens_used: int = Field(default=0, description="Total tokens used in this interaction")


class GenMetricsNodeResult(SemanticNodeResult):
    """Metric generation result, including an actionable blocked outcome."""

    status: Optional[Literal["generated", "skipped", "blocked"]] = Field(
        default=None,
        description="Metric generation outcome; None is reserved for execution errors.",
    )
    blocker_code: Optional[
        Literal[
            "semantic_model_required",
            "semantic_model_selection_required",
            "semantic_model_target_invalid",
        ]
    ] = Field(default=None, description="Actionable prerequisite when status is blocked")
    skip_reason: Optional[Literal["not_a_metric"]] = Field(
        default=None,
        description="Why metric generation was skipped; only non-metric requests may skip in OSI mode.",
    )
