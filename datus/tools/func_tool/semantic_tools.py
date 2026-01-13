# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Function Tools

Provides unified interface to semantic layer services through adapters.
Tools delegate to registered semantic adapters while leveraging unified storage for performance.
"""

from typing import List, Optional

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import MetricRAG
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.tools.func_tool.attribution_utils import DimensionAttributionUtil
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.tools.semantic_tools.models import AnomalyContext
from datus.tools.semantic_tools.registry import semantic_adapter_registry
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _run_async(coro):
    """
    Run async coroutine safely, handling both sync and async contexts.

    Delegates to the centralized run_async utility which handles:
    - Deadlock prevention for nested calls
    - Proper event loop management
    - Timeout support
    - Improved error handling

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine
    """
    from datus.utils.async_utils import run_async

    return run_async(coro)


class SemanticTools:
    """Function tool wrapper for semantic layer operations."""

    def __init__(
        self,
        agent_config: AgentConfig,
        sub_agent_name: Optional[str] = None,
        adapter_type: Optional[str] = None,
    ):
        """
        Initialize semantic function tool.

        Args:
            agent_config: Agent configuration
            sub_agent_name: Optional sub-agent name for scoped storage
            adapter_type: Optional adapter type (e.g., "metricflow"). If not provided, tools will use storage only.
        """
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.adapter_type = adapter_type

        # Initialize storage RAG interfaces
        self.semantic_model_rag = SemanticModelRAG(agent_config, sub_agent_name)
        self.metric_rag = MetricRAG(agent_config, sub_agent_name)

        # Lazy load adapter and attribution tool
        self._adapter: Optional[BaseSemanticAdapter] = None
        self._attribution_tool: Optional[DimensionAttributionUtil] = None

    @property
    def adapter(self) -> Optional[BaseSemanticAdapter]:
        """Lazy load semantic adapter if configured."""
        if self._adapter is None and self.adapter_type:
            try:
                # Try to get adapter-specific config from agent_config
                adapter_config = getattr(self.agent_config, f"{self.adapter_type}_config", None)
                if adapter_config is None:
                    # Get namespace from agent_config
                    namespace = getattr(self.agent_config, "namespace", None) or self.agent_config.current_namespace

                    # Get the registered config class for this adapter type
                    metadata = semantic_adapter_registry.get_metadata(self.adapter_type)
                    if metadata and metadata.config_class:
                        # Use the adapter's config class
                        adapter_config = metadata.config_class(namespace=namespace)
                    else:
                        # Fallback to base config
                        from datus.tools.semantic_tools.config import SemanticAdapterConfig

                        adapter_config = SemanticAdapterConfig(namespace=namespace)

                self._adapter = semantic_adapter_registry.create_adapter(self.adapter_type, adapter_config)
                logger.info(f"Loaded semantic adapter: {self.adapter_type}")
            except Exception as e:
                logger.warning(f"Failed to load semantic adapter '{self.adapter_type}': {e}")
                self._adapter = None
        return self._adapter

    @property
    def attribution_tool(self) -> Optional[DimensionAttributionUtil]:
        """Lazy load attribution tool when adapter is available."""
        if self._attribution_tool is None and self.adapter is not None:
            self._attribution_tool = DimensionAttributionUtil(self.adapter)
        return self._attribution_tool

    def _reload_adapter(self) -> bool:
        """
        Reload the semantic adapter to pick up new configuration changes.

        This is useful after writing new metric/semantic model YAML files,
        as MetricFlow needs to reload the configuration to know about new metrics.

        Returns:
            True if reload succeeded, False otherwise
        """
        if not self.adapter_type:
            logger.warning("No adapter type configured, cannot reload")
            return False

        try:
            # Clear cached adapter and attribution tool
            self._adapter = None
            self._attribution_tool = None

            # Force reload by accessing the property
            if self.adapter is not None:
                logger.info(f"Successfully reloaded semantic adapter: {self.adapter_type}")
                return True
            else:
                logger.error("Failed to reload semantic adapter")
                return False

        except Exception as e:
            logger.error(f"Error reloading semantic adapter: {e}", exc_info=True)
            return False

    def available_tools(self) -> List[Tool]:
        """
        Get list of available tools.

        Returns:
            List of Tool objects for LLM function calling
        """
        tools = [
            trans_to_function_tool(self.search_metrics),
            trans_to_function_tool(self.list_metrics),
            trans_to_function_tool(self.get_dimensions),
            trans_to_function_tool(self.query_metrics),
        ]

        # Add adapter-dependent tools
        if self.adapter:
            tools.append(trans_to_function_tool(self.validate_semantic))

        # Add attribution tools if attribution_tool is available
        if self.attribution_tool:
            tools.append(trans_to_function_tool(self.attribution_analyze))

        return tools

    def search_metrics(
        self,
        query_text: str,
        subject_path: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> FuncToolResult:
        """
        Search metrics using vector search in unified storage.

        Args:
            query_text: Natural language query for metric search
            subject_path: Optional subject tree path filter (e.g., ["Finance", "Revenue"])
            top_n: Maximum number of results to return

        Returns:
            FuncToolResult with matching metrics
        """
        try:
            results = self.metric_rag.search_metrics(
                query_text=query_text,
                subject_path=subject_path,
                top_n=top_n,
            )

            if not results:
                return FuncToolResult(
                    success=0,
                    error=f"No metrics found matching '{query_text}'",
                    result=[],
                )

            # Format results for LLM
            formatted_metrics = []
            for metric in results:
                formatted_metrics.append(
                    {
                        "name": metric.get("name"),
                        "description": metric.get("description"),
                        "type": metric.get("metric_type"),
                        "dimensions": metric.get("dimensions", []),
                        "measures": metric.get("base_measures", []),
                        "subject_path": metric.get("subject_path", []),
                    }
                )

            return FuncToolResult(
                success=1,
                result=formatted_metrics,
            )

        except Exception as e:
            logger.error(f"Error searching metrics: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to search metrics: {str(e)}",
            )

    def list_metrics(
        self,
        path: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> FuncToolResult:
        """
        List available metrics from storage (or adapter if storage is empty).

        Args:
            path: Optional subject tree path filter (e.g., ["Finance", "Revenue"])
            limit: Maximum number of metrics to return
            offset: Number of metrics to skip

        Returns:
            FuncToolResult with list of metrics
        """
        try:
            # Try storage first
            all_metrics = self.metric_rag.search_all_metrics()

            # Filter by subject path if provided
            if path:
                all_metrics = [m for m in all_metrics if m.get("subject_path", [])[: len(path)] == path]

            # Apply pagination
            paginated_metrics = all_metrics[offset : offset + limit]

            if not paginated_metrics and self.adapter:
                # Fallback to adapter if storage is empty
                logger.info("Storage empty, falling back to adapter")
                async_result = _run_async(self.adapter.list_metrics(path=path, limit=limit, offset=offset))
                paginated_metrics = [
                    {
                        "name": m.name,
                        "description": m.description,
                        "type": m.type,
                        "dimensions": m.dimensions,
                        "measures": m.measures,
                        "unit": m.unit,
                        "format": m.format,
                        "path": m.path,
                    }
                    for m in async_result
                ]

            return FuncToolResult(
                success=1,
                result=paginated_metrics,
            )

        except Exception as e:
            logger.error(f"Error listing metrics: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to list metrics: {str(e)}",
            )

    def get_dimensions(
        self,
        metric_name: str,
        path: Optional[List[str]] = None,
    ) -> FuncToolResult:
        """
        Get available dimensions for a specific metric.

        Args:
            metric_name: Name of the metric
            path: Optional subject tree path (e.g., ["Finance", "Revenue"])

        Returns:
            FuncToolResult with list of dimension names
        """
        try:
            # Try to get from storage first
            metric_details = None
            if path:
                metric_details_list = self.metric_rag.storage.search_all_metrics(subject_path=path)
                metric_details_list = [m for m in metric_details_list if m.get("name") == metric_name]
                if metric_details_list:
                    metric_details = metric_details_list[0]
            else:
                # Search all metrics
                all_metrics = self.metric_rag.search_all_metrics()
                matching = [m for m in all_metrics if m.get("name") == metric_name]
                if matching:
                    metric_details = matching[0]

            if metric_details:
                dimensions = metric_details.get("dimensions", [])
                return FuncToolResult(
                    success=1,
                    result=dimensions,
                )

            # Fallback to adapter
            if self.adapter:
                logger.info(f"Metric '{metric_name}' not in storage, querying adapter")
                dimensions = _run_async(self.adapter.get_dimensions(metric_name=metric_name, path=path))
                return FuncToolResult(
                    success=1,
                    result=dimensions,
                )

            return FuncToolResult(
                success=0,
                error=f"Metric '{metric_name}' not found",
                result=[],
            )

        except Exception as e:
            logger.error(f"Error getting dimensions: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to get dimensions: {str(e)}",
            )

    def query_metrics(
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
    ) -> FuncToolResult:
        """
        Query metrics data (requires adapter).

        Args:
            metrics: List of metric names to query
            dimensions: Optional list of dimensions to group by
            path: Optional subject tree path
            time_start: Optional start time (ISO format like '2024-01-01' or relative like '-7d')
            time_end: Optional end time (ISO format like '2024-01-31' or relative like 'now')
            time_granularity: Optional time granularity for aggregation ('day', 'week', 'month', 'quarter', 'year')
            where: Optional SQL WHERE clause (without WHERE keyword)
            limit: Optional maximum number of rows
            order_by: Optional list of fields to sort by
            dry_run: If True, only validate and return query plan

        Returns:
            FuncToolResult with query results or explain plan
        """
        if not self.adapter:
            return FuncToolResult(
                success=0,
                error="No semantic adapter configured. Cannot execute queries without adapter.",
            )

        try:
            # Execute query via adapter
            result = _run_async(
                self.adapter.query_metrics(
                    metrics=metrics,
                    dimensions=dimensions or [],
                    path=path,
                    time_start=time_start,
                    time_end=time_end,
                    time_granularity=time_granularity,
                    where=where,
                    limit=limit,
                    order_by=order_by,
                    dry_run=dry_run,
                )
            )

            # Format result
            result_dict = {
                "columns": result.columns,
                "data": result.data,
                "metadata": result.metadata,
            }

            return FuncToolResult(
                success=1,
                result=result_dict,
            )

        except Exception as e:
            logger.error(f"Error querying metrics: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to query metrics: {str(e)}",
            )

    def validate_semantic(self) -> FuncToolResult:
        """
        Validate semantic layer configuration (requires adapter).

        After successful validation, the adapter is reloaded to pick up any new
        metrics or semantic model changes. This ensures that subsequent calls to
        query_metrics can find newly created metrics.

        Returns:
            FuncToolResult with validation status and issues
        """
        if not self.adapter:
            return FuncToolResult(
                success=0,
                error="No semantic adapter configured. Cannot validate without adapter.",
                result=None,
            )

        try:
            validation_result = _run_async(self.adapter.validate_semantic())

            # Serialize ValidationIssue objects to dicts
            issues_data = [
                issue.model_dump() if hasattr(issue, "model_dump") else {"severity": "error", "message": str(issue)}
                for issue in validation_result.issues
            ]

            # If validation succeeded, reload the adapter to pick up new metrics
            if validation_result.valid:
                logger.info("Validation succeeded, reloading adapter to pick up new metrics...")
                self._reload_adapter()

            return FuncToolResult(
                success=1 if validation_result.valid else 0,
                result={"valid": validation_result.valid, "issues": issues_data},
                error=None if validation_result.valid else f"{len(validation_result.issues)} validation errors",
            )

        except Exception as e:
            logger.error(f"Error validating semantic config: {e}", exc_info=True)
            return FuncToolResult(
                success=0,
                error=f"Failed to validate semantic config: {str(e)}",
                result=None,
            )

    def attribution_analyze(
        self,
        metric_name: str,
        candidate_dimensions: List[str],
        baseline_start: str,
        baseline_end: str,
        current_start: str,
        current_end: str,
        anomaly_context: Optional[AnomalyContext] = None,
        max_selected_dimensions: int = 3,
        top_n_values: int = 10,
    ) -> FuncToolResult:
        """
        Unified attribution analysis for anomaly investigation.

        Automatically ranks candidate dimensions by explanatory power and calculates
        delta contributions for the most important dimensions. Perfect for LLM-driven
        root cause analysis of metric anomalies.

        Args:
            metric_name: Metric to analyze (e.g., "payment_amount", "revenue")
            candidate_dimensions: List of dimensions to evaluate (e.g., ["region", "product", "channel"])
            baseline_start: Baseline period start date (e.g., "2026-01-01")
            baseline_end: Baseline period end date (e.g., "2026-01-01")
            current_start: Current period start date (e.g., "2026-01-08")
            current_end: Current period end date (e.g., "2026-01-08")
            anomaly_context: Optional anomaly detection context (AnomalyContext with rule and observed_change_pct)
            max_selected_dimensions: Maximum dimensions to select (default 3)
            top_n_values: Number of top dimension values to return (default 10)

        Returns:
            FuncToolResult with:
            - dimension_ranking: All dimensions ranked by importance score
            - selected_dimensions: Top dimensions selected for analysis
            - top_dimension_values: Delta contributions of dimension values
        """
        if not self.attribution_tool:
            return FuncToolResult(
                success=0,
                error="Attribution tool not available. Requires semantic adapter.",
            )

        try:
            # Convert AnomalyContext to dict for attribution_tool
            # Handle both dict (from LLM) and AnomalyContext object
            if anomaly_context is None:
                anomaly_context_dict = None
            elif isinstance(anomaly_context, dict):
                anomaly_context_dict = anomaly_context
            else:
                anomaly_context_dict = anomaly_context.model_dump()

            result = _run_async(
                self.attribution_tool.attribution_analyze(
                    metric_name=metric_name,
                    candidate_dimensions=candidate_dimensions,
                    baseline_start=baseline_start,
                    baseline_end=baseline_end,
                    current_start=current_start,
                    current_end=current_end,
                    anomaly_context=anomaly_context_dict,
                    max_selected_dimensions=max_selected_dimensions,
                    top_n_values=top_n_values,
                )
            )

            return FuncToolResult(
                success=1,
                result=result.model_dump(),
            )

        except Exception as e:
            logger.error(f"Error in attribution analysis: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to analyze attribution: {str(e)}",
            )
