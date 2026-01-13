# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import DimensionInfo, MetricDefinition, QueryResult, ValidationResult


class BaseSemanticAdapter(ABC):
    """
    Base class for all semantic layer adapters.

    This is the minimal interface that backend adapters must implement.
    Adapters translate these standardized calls to backend-specific APIs
    (MetricFlow, dbt Semantic Layer, Cube, etc.).

    Higher-level features (caching, LLM integration, dimension value enumeration)
    are provided by the SemanticLayerService wrapper.
    """

    def __init__(self, config: Any, service_type: str):
        """
        Initialize semantic adapter.

        Args:
            config: Adapter configuration
            service_type: Type of semantic service (e.g., 'metricflow', 'dbt', 'cube')
        """
        self.config = config
        self.service_type = service_type
        self.namespace = getattr(config, "namespace", None)

    # ==================== Semantic Model Interface (Simplified) ====================

    def get_semantic_model(
        self,
        table_name: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Get semantic model for a specific table.

        Returns semantic model with structure:
        {
            "semantic_model_name": str,
            "description": str,
            "dimensions": List[{name, description, expr}],
            "measures": List[{name, description, expr}],
            "identifiers": List[{name, description, expr}],
        }

        This is the ONLY method required for semantic model interface.
        Used by describe_table() to enrich table metadata.

        Default implementation returns None (not all adapters support semantic models).
        """
        return None

    def list_semantic_models(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> List[str]:
        """
        List all available semantic models (optional, for discovery).
        Default implementation returns empty list.

        Returns:
            List of semantic model names
        """
        return []

    # ==================== Metrics Interface (Complete CRUD) ====================

    @abstractmethod
    async def list_metrics(
        self,
        path: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MetricDefinition]:
        """
        List available metrics from the semantic layer.

        Args:
            path: the path level of metrics. for example: ["domainA"], ["domain", "layer1", "layer2"],
                  leave it as None or [] to list all the metrics.
            limit: Maximum number of metrics to return
            offset: Number of metrics to skip (for pagination)

        Backend adapters should map their native metadata to MetricDefinition format.
        """
        raise NotImplementedError()

    @abstractmethod
    async def get_dimensions(
        self,
        metric_name: str,
        path: Optional[List[str]] = None,
    ) -> List[DimensionInfo]:
        """
        Get queryable dimensions for a specific metric.

        Args:
            metric_name: Name of the metric to query dimensions for.
            path: path name of metric e.g. ["domainA"], ["domain", "layer1", "layer2"]

        Returns:
            List of DimensionInfo objects containing dimension name and description.
        """
        raise NotImplementedError()

    @abstractmethod
    async def query_metrics(
        self,
        metrics: List[str],
        dimensions: Optional[List[str]] = None,
        path: Optional[List[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_granularity: Optional[str] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> QueryResult:
        """
        Execute a metric query or explain the execution plan.

        Adapters translate the standardized parameters to backend-specific format:
        - where -> backend's filter syntax
        - time parameters -> backend's time constraint format
        - dry_run=True -> backend's explain/validate API

        Args:
            metrics: List of metric names to query
            dimensions: List of dimensions to group by
            path: metric path for scoping
            time_start: Start time (ISO format like '2024-01-01' or relative like '-7d')
            time_end: End time (ISO format like '2024-01-31' or relative like 'now')
            time_granularity: Time granularity for aggregation ('day', 'week', 'month', 'quarter', 'year')
            where: SQL WHERE clause without the WHERE keyword.
                   Example: "region = 'US' AND revenue > 1000"
            limit: Maximum number of rows to return
            order_by: List of fields to sort by
            dry_run: If True, only validate and return query plan without execution

        Returns:
            QueryResult with:
            - If dry_run=False: columns and data contain actual query results
            - If dry_run=True: data contains explain info like [{"sql": "...", "plan": "...", "valid": true}]
        """
        raise NotImplementedError()

    @abstractmethod
    async def validate_semantic(self) -> ValidationResult:
        """
        Validate the semantic layer configuration files.

        Checks for:
        - Configuration syntax validity (YAML, JSON, etc.)
        - Reference integrity (metrics reference valid measures/dimensions)
        - Semantic correctness (no circular dependencies, etc.)
        - SQL syntax compatibility (if applicable)

        Backend-specific validation:
        - MetricFlow: mf validate-configs
        - dbt: dbt parse / dbt compile
        - Cube: schema validation

        Returns:
            ValidationResult with validation status and list of issues
        """
        raise NotImplementedError()
