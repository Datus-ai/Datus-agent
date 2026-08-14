# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Schema models for Semantic Agentic Node.

This module defines the input and output models for the SemanticAgenticNode,
providing structured validation for semantic model generation interactions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from datus.schemas.at_context import AtContextInput
from datus.schemas.base import BaseResult


class SourceQueryEvidence(BaseModel):
    """Structured success-story SQL carried independently from the LLM prompt."""

    source_sql_name: str = Field(..., description="Stable source-query name, for example sql_1")
    sql: str = Field(..., description="Original SQL from the success-story row")
    question: str = Field(default="", description="Business question associated with the SQL")
    source_id: str = Field(default="", description="Optional provenance source identifier")
    source_type: str = Field(default="success_story", description="Optional provenance source type")
    source_context_ids: List[str] = Field(default_factory=list, description="Optional provenance context IDs")
    source_metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional provenance metadata")


def source_query_from_success_story_row(
    row: Any,
    row_index: int,
    success_story: str,
) -> Optional[SourceQueryEvidence]:
    """Build structured SQL evidence from one success-story row."""
    sql = _clean_tabular_cell(row.get("sql"))
    if not sql:
        return None
    provenance = source_provenance_from_success_story_row(row, row_index, success_story) or {}
    return SourceQueryEvidence(
        source_sql_name=f"sql_{row_index + 1}",
        sql=sql,
        question=_clean_tabular_cell(row.get("question")),
        source_id=provenance.get("source_id")
        or _clean_tabular_cell(row.get("source_id"))
        or f"{Path(success_story).name}:{row_index}",
        source_type=(provenance.get("source_type") or _clean_tabular_cell(row.get("source_type")) or "success_story"),
        source_context_ids=provenance.get("source_context_ids", []),
        source_metadata=(provenance.get("source_metadata") or _parse_source_metadata(row.get("source_metadata"))),
    )


def source_provenance_from_success_story_row(
    row: Any,
    row_index: int,
    success_story: str,
) -> Optional[dict[str, Any]]:
    """Normalize optional provenance from one success-story row."""
    context_ids: list[str] = []
    for column in ("source_context_ids", "source_context_id", "context_ids", "context_id"):
        context_ids.extend(_parse_context_ids(row.get(column)))
    context_ids = list(dict.fromkeys(context_ids))
    if not context_ids:
        return None

    metadata = _parse_source_metadata(row.get("source_metadata"))
    source_id = _clean_tabular_cell(row.get("source_id")) or f"{Path(success_story).name}:{row_index}"
    source_type = _clean_tabular_cell(row.get("source_type")) or "success_story"
    metadata.setdefault("source_id", source_id)
    metadata.setdefault("source_type", source_type)
    metadata.setdefault("row_index", row_index)
    question = _clean_tabular_cell(row.get("question"))
    if question:
        metadata.setdefault("question", question)
    task_id = _clean_tabular_cell(row.get("task_id"))
    if task_id:
        metadata.setdefault("task_id", task_id)
    return {
        "source_id": source_id,
        "source_type": source_type,
        "source_context_ids": context_ids,
        "source_metadata": metadata,
    }


def _clean_tabular_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(value != value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _parse_context_ids(value: Any) -> list[str]:
    text = _clean_tabular_cell(value)
    if not text:
        return []
    return [part for part in (item.strip() for item in text.replace(",", ";").split(";")) if part]


def _parse_source_metadata(value: Any) -> dict[str, Any]:
    text = _clean_tabular_cell(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return parsed if isinstance(parsed, dict) else {"raw": text}


class SemanticNodeInput(AtContextInput):
    """
    Input model for SemanticAgenticNode interactions.
    """

    user_message: str = Field(..., description="User's input message")
    authoring_scope: Literal["datasets", "full"] = Field(
        default="full",
        description=(
            "Semantic-modeling write scope. `datasets` permits only datasets, fields, relationships, and model "
            "metadata; `full` also permits metric changes."
        ),
    )
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
    source_queries: List[SourceQueryEvidence] = Field(
        default_factory=list,
        description="Structured SQL evidence supplied directly by workflow callers.",
    )

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


class SemanticModelingNodeResult(SemanticNodeResult):
    """Outcome of unified semantic dataset and metric authoring."""

    status: Optional[Literal["generated", "skipped", "blocked"]] = Field(
        default=None,
        description="Unified semantic-modeling outcome; None is reserved for execution errors.",
    )
    blocker_code: Optional[
        Literal[
            "semantic_model_required",
            "semantic_model_selection_required",
            "semantic_model_target_invalid",
        ]
    ] = Field(default=None, description="Actionable target prerequisite when status is blocked")
    skip_reason: Optional[Literal["no_semantic_change"]] = Field(
        default=None,
        description="The request did not require a semantic dataset, relationship, or metric change.",
    )
