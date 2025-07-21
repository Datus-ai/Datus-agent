"""
API models for the Datus Agent FastAPI service.
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Query execution type."""

    SYNC = "sync"
    ASYNC = "async"


class DatabaseType(str, Enum):
    """Supported database types."""

    DUCKDB = "duckdb"
    SNOWFLAKE = "snowflake"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    STARROCKS = "starrocks"


class QueryRequest(BaseModel):
    """Request model for SQL query generation."""

    query: str = Field(..., description="Natural language query description")
    namespace: str = Field(..., description="Database namespace to use")
    database: Optional[str] = Field(None, description="Specific database name")
    schema_name: Optional[str] = Field(None, description="Database schema name")
    domain: Optional[str] = Field(None, description="Business domain context")
    layer1: Optional[str] = Field(None, description="Layer 1 context")
    layer2: Optional[str] = Field(None, description="Layer 2 context")
    external_knowledge: Optional[str] = Field(None, description="Additional context or evidence")
    query_type: QueryType = Field(QueryType.SYNC, description="Query execution type")
    max_steps: Optional[int] = Field(10, description="Maximum workflow steps")
    plan: Optional[str] = Field(
        "fixed", description="Workflow plan type from workflow.yml (fixed, dynamic, reflection, etc.)"
    )


class QueryResponse(BaseModel):
    """Response model for SQL query results."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Query execution status")
    sql: Optional[str] = Field(None, description="Generated SQL query")
    result: Optional[List[Dict[str, Any]]] = Field(None, description="Query execution results")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if any")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")


class StreamResponse(BaseModel):
    """Response model for streaming updates."""

    task_id: str = Field(..., description="Unique task identifier")
    event_type: str = Field(..., description="Type of update event")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: float = Field(..., description="Event timestamp")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Service version")
    database_status: Dict[str, str] = Field(..., description="Database connection status")
    llm_status: str = Field(..., description="LLM service status")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Detailed error information")
    task_id: Optional[str] = Field(None, description="Associated task ID if any")
