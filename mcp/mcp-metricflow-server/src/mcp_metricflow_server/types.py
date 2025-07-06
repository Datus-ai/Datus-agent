"""Type definitions for MCP MetricFlow Server."""

from typing import Optional

from pydantic import BaseModel, Field


class MetricToolResponse(BaseModel):
    """Response for metric-related tools."""

    name: str = Field(description="Metric name")
    description: Optional[str] = Field(default=None, description="Metric description")
    type: Optional[str] = Field(default=None, description="Metric type")
    expr: Optional[str] = Field(default=None, description="Metric expression")


class DimensionToolResponse(BaseModel):
    """Response for dimension-related tools."""

    name: str = Field(description="Dimension name")
    description: Optional[str] = Field(default=None, description="Dimension description")
    type: Optional[str] = Field(default=None, description="Dimension type")
    expr: Optional[str] = Field(default=None, description="Dimension expression")


class EntityToolResponse(BaseModel):
    """Response for entity-related tools."""

    name: str = Field(description="Entity name")
    description: Optional[str] = Field(default=None, description="Entity description")
    type: Optional[str] = Field(default=None, description="Entity type")
    expr: Optional[str] = Field(default=None, description="Entity expression")


class OrderByParam(BaseModel):
    """Order by parameter for queries."""

    field: str = Field(description="Field to order by")
    direction: str = Field(default="asc", description="Order direction (asc/desc)")


class QueryMetricsSuccess(BaseModel):
    """Successful query metrics response."""

    result: str = Field(description="Query result")
    sql: Optional[str] = Field(default=None, description="Generated SQL")


class QueryMetricsError(BaseModel):
    """Error response for query metrics."""

    error: str = Field(description="Error message")


class MetricFlowCommandResult(BaseModel):
    """Result of a MetricFlow CLI command execution."""

    success: bool = Field(description="Whether the command succeeded")
    output: str = Field(description="Command output")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    return_code: int = Field(description="Command return code")
