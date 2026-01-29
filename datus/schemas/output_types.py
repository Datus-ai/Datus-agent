# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Structured Output Types for Agent SDK integration.

These Pydantic models define the expected output format for various agent operations.
When used with the Agent SDK's output_type parameter, the model will automatically
validate and structure its response according to these schemas.

Benefits of Structured Output:
- No need for JSON format instructions in prompts
- Automatic response validation
- Type-safe output handling
- Better error messages when output doesn't match schema

Example usage:
    from datus.schemas.output_types import SQLGenerationResult

    result = await model.generate_with_tools(
        prompt="Generate a query for top 10 customers",
        output_type=SQLGenerationResult,
        ...
    )
    # result["content"] is now a SQLGenerationResult object
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SQLGenerationResult(BaseModel):
    """Result of SQL generation."""

    sql: str = Field(description="The generated SQL query")
    explanation: str = Field(description="Explanation of what the SQL does and why these tables were chosen")
    tables_used: List[str] = Field(description="List of table names used in the query")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score based on query complexity and data coverage"
    )


class SchemaLinkingResult(BaseModel):
    """Result of schema linking analysis."""

    relevant_tables: List[str] = Field(description="Tables relevant to the user's query")
    relevant_columns: List[str] = Field(description="Columns relevant to the user's query (format: table.column)")
    reasoning: str = Field(description="Explanation of why these tables and columns were selected")


class ReflectionResult(BaseModel):
    """Result of SQL reflection/validation."""

    is_correct: bool = Field(description="Whether the SQL is correct and answers the question")
    issues: List[str] = Field(default_factory=list, description="List of issues found in the SQL")
    suggested_fix: Optional[str] = Field(default=None, description="Suggested fix for the SQL if issues found")


class SQLFixResult(BaseModel):
    """Result of SQL fix operation."""

    fixed_sql: str = Field(description="The corrected SQL query")
    changes_made: List[str] = Field(description="List of changes made to fix the SQL")
    explanation: str = Field(description="Explanation of what was wrong and how it was fixed")


class MetricSearchResult(BaseModel):
    """Result of metric search."""

    metric_name: str = Field(description="Name of the found metric")
    metric_definition: str = Field(description="Definition or formula of the metric")
    related_tables: List[str] = Field(description="Tables involved in computing this metric")
    relevance_score: float = Field(ge=0.0, le=1.0, description="How relevant this metric is to the query")


class ChartRecommendation(BaseModel):
    """Recommendation for chart visualization."""

    chart_type: str = Field(description="Recommended chart type (bar, line, pie, scatter, etc.)")
    x_axis: str = Field(description="Column to use for x-axis")
    y_axis: str = Field(description="Column to use for y-axis")
    group_by: Optional[str] = Field(default=None, description="Optional column to group data by")
    reasoning: str = Field(description="Explanation of why this chart type is recommended")


class QueryClassification(BaseModel):
    """Classification of user query intent."""

    intent: str = Field(description="Primary intent (question, command, clarification, etc.)")
    domain: str = Field(description="Business domain (sales, marketing, finance, etc.)")
    complexity: str = Field(description="Query complexity (simple, medium, complex)")
    requires_aggregation: bool = Field(description="Whether the query requires aggregation")
    time_dimension: Optional[str] = Field(default=None, description="Time dimension if applicable (day, week, month)")
