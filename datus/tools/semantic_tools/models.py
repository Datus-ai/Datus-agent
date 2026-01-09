# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DimensionInfo(BaseModel):
    """Information about a dimension"""

    name: str = Field(..., description="Dimension name")
    description: Optional[str] = Field(None, description="Dimension description")


class MetricDefinition(BaseModel):
    """Metadata about a specific metric"""

    name: str = Field(..., description="Metric name")
    description: Optional[str] = Field(None, description="Metric description")
    type: Optional[str] = Field(None, description="Metric type (simple, ratio, derived, etc.)")
    dimensions: List[str] = Field(default_factory=list, description="Available dimensions for this metric")
    measures: List[str] = Field(default_factory=list, description="Underlying measures used")
    unit: Optional[str] = Field(None, description="Unit of measurement (e.g., 'USD', 'count', 'percent')")
    format: Optional[str] = Field(None, description="Display format (e.g., ',.2f', '0.00%')")
    path: Optional[List[str]] = Field(
        None, description="Subject tree hierarchy path (e.g., ['domain', 'layer1', 'layer2'])"
    )


class QueryResult(BaseModel):
    """
    Standardized query response for both actual execution and dry-run.

    For actual query (dry_run=False):
        - columns: list of column names
        - data: list of row dictionaries
        - metadata: optional metadata (execution_time, row_count, etc.)

    For dry-run/explain (dry_run=True):
        - columns: typically empty or ["sql", "plan", "error"]
        - data: explain results, e.g. [{"sql": "SELECT...", "plan": "...", "valid": true}]
        - metadata: validation info, warnings, etc.
    """

    columns: List[str] = Field(default_factory=list, description="Column names")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Query result rows")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (execution_time, warnings, etc.)"
    )


class ValidationIssue(BaseModel):
    """A single validation issue"""

    severity: str = Field(..., description="Severity level: error, warning, info")
    message: str = Field(..., description="Issue description")
    location: Optional[str] = Field(None, description="Location in config where issue was found")


class ValidationResult(BaseModel):
    """Result of a semantic configuration validation check"""

    valid: bool = Field(..., description="Whether the configuration is valid")
    issues: List[ValidationIssue] = Field(default_factory=list, description="List of validation issues")


class AnomalyContext(BaseModel):
    """Context information for anomaly detection in attribution analysis"""

    model_config = {"extra": "forbid"}

    rule: Optional[str] = Field(None, description="Anomaly detection rule name (e.g., 'wow_growth_gt_20pct')")
    observed_change_pct: Optional[float] = Field(None, description="Observed percentage change")
