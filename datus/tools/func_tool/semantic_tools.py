# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Function Tools

Provides unified interface to semantic layer services through adapters.
Tools delegate to registered semantic adapters while leveraging unified storage for performance.
"""

import json
from typing import List, Optional

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import MetricRAG
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.tools.func_tool.attribution_utils import DimensionAttributionUtil
from datus.tools.func_tool.base import FuncToolListResult, FuncToolResult, normalize_null, trans_to_function_tool
from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.tools.semantic_tools.models import AnomalyContext
from datus.tools.semantic_tools.registry import semantic_adapter_registry
from datus.utils.compress_utils import DataCompressor
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _normalize_dimension_rows(raw) -> list:
    """Normalize dimension payload into ``List[Dict[str, Any]]`` for the envelope.

    Adapters (MetricFlow) return pydantic ``DimensionInfo`` objects with a
    full schema; storage may hold bare strings (dimension name only) or
    dicts. FuncToolListResult.items must be ``List[Dict]`` either way, so
    wrap naked strings into ``{"name": str}`` and leave structured rows
    untouched.
    """
    if not raw:
        return []
    normalized = []
    for d in raw:
        if hasattr(d, "model_dump"):
            normalized.append(d.model_dump())
        elif isinstance(d, dict):
            normalized.append(d)
        elif isinstance(d, str):
            normalized.append({"name": d})
        else:
            normalized.append({"name": str(d)})
    return normalized


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

    @classmethod
    def all_tools_name(cls) -> List[str]:
        """Return list of all tool method names for wizard display."""
        return [
            "list_metrics",
            "get_dimensions",
            "query_metrics",
            "validate_semantic",
            "attribution_analyze",
        ]

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
        self.compressor = DataCompressor(model_name=agent_config.active_model().model)

        # Lazy load adapter and attribution tool
        self._adapter: Optional[BaseSemanticAdapter] = None
        self._attribution_tool: Optional[DimensionAttributionUtil] = None

    def _extract_db_config(self, namespace: str) -> Optional[dict]:
        """Extract db_config dict from the selected database config."""
        try:
            db_config_obj = self.agent_config.current_db_config(namespace)
        except Exception:
            return None
        if db_config_obj is None:
            return None
        raw = db_config_obj.to_dict()
        extra = raw.get("extra")
        db_config = {
            k: str(v)
            for k, v in raw.items()
            if v is not None and v != "" and k not in ("extra", "logic_name", "path_pattern", "catalog", "default")
        }
        # Preserve connector-specific `extra` fields without overwriting explicit top-level keys
        if isinstance(extra, dict):
            for k, v in extra.items():
                if v is None or v == "":
                    continue
                db_config.setdefault(k, str(v))
        return db_config

    @property
    def adapter(self) -> Optional[BaseSemanticAdapter]:
        """Lazy load semantic adapter if configured."""
        if self._adapter is None:
            try:
                resolved_adapter = self.adapter_type
                resolver = getattr(self.agent_config, "resolve_semantic_adapter", None)
                if callable(resolver):
                    resolved_adapter = resolver(self.adapter_type)
                if not resolved_adapter:
                    return None

                metadata = semantic_adapter_registry.get_metadata(resolved_adapter)
                builder = getattr(self.agent_config, "build_semantic_adapter_config", None)
                adapter_config = builder(resolved_adapter) if callable(builder) else None
                if adapter_config is None:
                    namespace = getattr(self.agent_config, "namespace", None) or self.agent_config.current_datasource
                    db_config = self._extract_db_config(namespace)
                    semantic_models_path = str(self.agent_config.path_manager.semantic_models_dir)

                    if metadata and metadata.config_class:
                        adapter_config = metadata.config_class(
                            namespace=namespace,
                            db_config=db_config,
                            semantic_models_path=semantic_models_path,
                        )
                    else:
                        from datus.tools.semantic_tools.config import SemanticAdapterConfig

                        adapter_config = SemanticAdapterConfig(namespace=namespace)
                elif isinstance(adapter_config, dict):
                    if metadata and metadata.config_class:
                        adapter_config = metadata.config_class(**adapter_config)
                    else:
                        from datus.tools.semantic_tools.config import SemanticAdapterConfig

                        adapter_config = SemanticAdapterConfig(**adapter_config)

                self.adapter_type = resolved_adapter
                self._adapter = semantic_adapter_registry.create_adapter(resolved_adapter, adapter_config)
                logger.info(f"Loaded semantic adapter: {resolved_adapter}")
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
            FuncToolResult with result as FuncToolListResult:
              - items (List[Dict]): metric rows, each with name, description, type,
                dimensions, measures, unit, format, path
              - total (int | None): full metric count before pagination
              - has_more (bool | None): True when offset + len(items) < total
              - extra (dict | None): {"next_offset": int} when has_more is True

            Pagination: call again with offset=extra.next_offset until
            has_more is False. Default limit=100; override if you need bigger
            pages. list_metrics never compresses — use the limit to control
            response size.
        """
        # Normalize null values from LLM
        path = normalize_null(path)
        logger.info(f"list_metrics called: path={path}, limit={limit}, offset={offset}")
        try:
            # Try storage first
            all_metrics = self.metric_rag.search_all_metrics()

            # Filter by subject path if provided
            if path:
                all_metrics = [m for m in all_metrics if m.get("subject_path", [])[: len(path)] == path]

            # Apply pagination
            paginated_metrics = all_metrics[offset : offset + limit]

            if paginated_metrics:
                formatted_metrics = [
                    {
                        "name": m.get("name"),
                        "description": m.get("description"),
                        "type": m.get("metric_type"),
                        "dimensions": m.get("dimensions", []),
                        "measures": m.get("base_measures", []),
                        "unit": m.get("unit"),
                        "format": m.get("format"),
                        "path": m.get("subject_path", []),
                    }
                    for m in paginated_metrics
                ]
                return self._build_metrics_envelope(formatted_metrics, total=len(all_metrics), offset=offset)

            # Empty storage AND no adapter → empty envelope (total still reflects
            # the filtered all_metrics, which may be >0 if offset overshot).
            if not self.adapter:
                return self._build_metrics_envelope([], total=len(all_metrics), offset=offset)

            logger.info("Storage empty, falling back to adapter")
            async_result = _run_async(self.adapter.list_metrics(path=path, limit=limit, offset=offset))
            adapter_metrics = [
                {
                    "name": m.name,
                    "description": m.description,
                    "type": getattr(m, "type", None),
                    "dimensions": getattr(m, "dimensions", []),
                    "measures": getattr(m, "measures", []),
                    "unit": getattr(m, "unit", None),
                    "format": getattr(m, "format", None),
                    "path": getattr(m, "path", None),
                }
                for m in async_result
            ]
            # Adapter path has no upstream total — leave it None so consumers
            # know to use has_more / len(items) < limit as the pagination hint.
            return self._build_metrics_envelope(adapter_metrics, total=None, offset=offset, limit=limit)

        except Exception as e:
            logger.error(f"Error listing metrics: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to list metrics: {str(e)}",
            )

    @staticmethod
    def _build_metrics_envelope(
        items: List[dict],
        *,
        total: Optional[int],
        offset: int,
        limit: Optional[int] = None,
    ) -> FuncToolResult:
        """Wrap paginated metric rows into a FuncToolListResult.

        When ``total`` is known (storage path) ``has_more`` is exact. When
        ``total`` is None (adapter path) ``has_more`` falls back to
        ``len(items) == limit`` — a heuristic, but good enough for the LLM
        to decide whether to fetch another page.
        """
        if total is not None:
            has_more: Optional[bool] = offset + len(items) < total
        elif limit is not None:
            has_more = len(items) == limit
        else:
            has_more = None
        extra = {"next_offset": offset + len(items)} if has_more else None
        return FuncToolResult(
            success=1,
            result=FuncToolListResult(items=items, total=total, has_more=has_more, extra=extra).model_dump(),
        )

    def get_dimensions(
        self,
        metric_name: str,
        path: Optional[List[str]] = None,
    ) -> FuncToolResult:
        """
        Get available dimensions for a specific metric.
        When an adapter is configured, returns dimension objects from the adapter.
        Otherwise falls back to dimension data from storage.

        Args:
            metric_name: Name of the metric
            path: Optional subject tree path (e.g., ["Finance", "Revenue"])

        Returns:
            FuncToolResult with result as FuncToolListResult:
              - items (List[Dict]): dimension rows. Adapter dimensions expose
                their full schema (name, type, expr, ...); storage dimensions
                fall back to a minimal {"name": ...} shape when only names are
                stored.
              - total, has_more, extra: dimensions isn't paginated, so total
                equals len(items) and has_more is False.
        """
        # Normalize null values from LLM
        path = normalize_null(path)
        logger.info(f"get_dimensions called: metric={metric_name}, path={path}")
        try:
            # Get dimensions from adapter (MetricFlow) to ensure consistency with query execution
            if self.adapter:
                dimensions = _run_async(self.adapter.get_dimensions(metric_name=metric_name, path=path))
                items = _normalize_dimension_rows(dimensions)
                return FuncToolResult(
                    success=1,
                    result=FuncToolListResult(items=items, total=len(items), has_more=False).model_dump(),
                )

            # Fallback to storage if no adapter configured
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
                raw = metric_details.get("dimensions", [])
                items = _normalize_dimension_rows(raw)
                return FuncToolResult(
                    success=1,
                    result=FuncToolListResult(items=items, total=len(items), has_more=False).model_dump(),
                )

            return FuncToolResult(
                success=0,
                error=f"Metric '{metric_name}' not found and no adapter configured",
                result=FuncToolListResult(items=[], total=0, has_more=False).model_dump(),
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
            dimensions: Optional list of dimensions to group by (from get_dimensions)
            path: Optional subject tree path (from list_subject_tree)
            time_start: Optional start time (ISO format like '2024-01-01' or relative like '-7d')
            time_end: Optional end time (ISO format like '2024-01-31' or relative like 'now')
            time_granularity: Optional time granularity for aggregation ('day', 'week', 'month', 'quarter', 'year')
            where: Optional SQL WHERE clause (without WHERE keyword)
            limit: Optional maximum number of rows
            order_by: Optional list of columns to sort by. Use column name for ascending,
                      prefix with '-' for descending. Examples: ['metric_time__day'] for ascending,
                      ['-message_count'] for descending. Do NOT use 'asc'/'desc' keywords.
            dry_run: If True, only validate and return query plan

        Returns:
            FuncToolResult with query results or explain plan
        """
        if not self.adapter:
            return FuncToolResult(
                success=0,
                error="No semantic adapter configured. Cannot execute queries without adapter.",
            )

        # Sanitize time parameters: LLM may pass string "null"/"None" instead of omitting
        time_start = normalize_null(time_start)
        time_end = normalize_null(time_end)
        logger.info(
            f"query_metrics called: metrics={metrics}, dimensions={dimensions}, path={path}, "
            f"time=[{time_start},{time_end}], granularity={time_granularity}, where={where}, "
            f"limit={limit}, dry_run={dry_run}"
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

            # Drop non-JSON-serializable metadata entries (MetricFlow puts a
            # ``DataflowPlan`` object under ``dataflow_plan``). ``str(v)`` on
            # those yields ``<... object at 0x...>`` which is useless to
            # both LLM callers and humans.
            safe_metadata = {}
            for k, v in (result.metadata or {}).items():
                try:
                    json.dumps(v)
                    safe_metadata[k] = v
                except (TypeError, ValueError):
                    continue

            result_dict = {
                "columns": result.columns,
                "data": self.compressor.compress(result.data),
                "metadata": safe_metadata,
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
        logger.info("validate_semantic called")
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
            metric_name: Metric to analyze(from list_metrics/search_metrics)
            candidate_dimensions: List of dimensions to evaluate (from get_dimensions)
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
