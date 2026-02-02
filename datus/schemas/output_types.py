# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Structured Output Types for Agent SDK integration.

These Pydantic models define the expected output format for various agentic node operations.
When used with the Agent SDK's output_type parameter, the model will automatically
validate and structure its response according to these schemas.

Benefits of Structured Output:
- No need for JSON format instructions in prompts
- Automatic response validation
- Type-safe output handling
- Better error messages when output doesn't match schema

Example usage:
    from datus.schemas.output_types import GenSqlOutput

    result = await model.generate_with_tools_stream(
        prompt="Generate a query for top 10 customers",
        output_type=GenSqlOutput,
        ...
    )
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SemanticModelGenerationOutput(BaseModel):
    """Output for semantic model generation."""

    semantic_model_files: List[str] = Field(
        description="List of generated semantic model file paths (e.g., ['orders.yml', 'customers.yml'])"
    )
    output: str = Field(description="Summary message describing what was generated")


class MetricGenerationOutput(BaseModel):
    """Output for metric generation."""

    semantic_model_file: Optional[str] = Field(
        default=None, description="Path to the semantic model file (e.g., 'orders.yml')"
    )
    metric_file: str = Field(description="Path to the generated metric file (e.g., 'sales_metrics.yml')")
    output: str = Field(description="Summary message describing what was generated")


class SqlSummaryGenerationOutput(BaseModel):
    """Output for SQL summary generation."""

    sql_summary_file: str = Field(description="Path to the generated SQL summary file (e.g., 'query_001.yaml')")
    output: str = Field(description="Summary message describing the SQL classification and summary")


class GenSqlOutput(BaseModel):
    """Output for SQL generation node."""

    sql: str = Field(description="The generated SQL query")
    tables: List[str] = Field(description="List of table names used in the query")
    explanation: str = Field(description="Explanation of what the SQL does")


class ReportGenerationOutput(BaseModel):
    """Output for report generation."""

    report: str = Field(description="The generated markdown report content")
    data_sources: List[str] = Field(default_factory=list, description="List of data sources used in the report")
    key_findings: List[str] = Field(default_factory=list, description="Key findings from the analysis")


class ExtKnowledgeGenerationOutput(BaseModel):
    """Output for external knowledge generation."""

    ext_knowledge_file: str = Field(description="Path to the generated external knowledge file")
    output: str = Field(description="Summary message describing what was generated")


class CompareOutput(BaseModel):
    """Output for SQL comparison analysis."""

    explanation: str = Field(description="Detailed explanation of the comparison analysis")
    suggest: str = Field(description="Suggestions for improving or fixing the SQL query")
