# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Function Tools

Provides unified interface to semantic layer services through adapters.
Tools delegate to registered semantic adapters while leveraging unified storage for performance.
"""

import inspect
import json
from typing import Dict, List, Literal, Optional, Set

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import MetricRAG
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.tools.func_tool.attribution_utils import DimensionAttributionUtil
from datus.tools.func_tool.base import FuncToolListResult, FuncToolResult, normalize_null, trans_to_function_tool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.tools.semantic_tools.models import AnomalyContext
from datus.tools.semantic_tools.registry import semantic_adapter_registry
from datus.utils.compress_utils import DataCompressor
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

NO_METRICS_PRESENT_MESSAGE = "No metrics present in the model."


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


def _normalize_name_list(value) -> List[str]:
    """Normalize LLM-provided string/list arguments into a clean list of names."""
    value = normalize_null(value)
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = [value]

    names = []
    for candidate in candidates:
        candidate = normalize_null(candidate)
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            names.append(text)
    return names


_TIME_GRANULARITIES = {"day", "week", "month", "quarter", "year"}


def _split_dimension_granularity(name: str) -> tuple[str, Optional[str]]:
    parts = name.rsplit("__", 1)
    if len(parts) != 2:
        return name, None
    base_name, suffix = parts[0], parts[1].lower()
    if suffix in _TIME_GRANULARITIES:
        return base_name, suffix
    return name, None


def _is_metric_time_dimension(name: str) -> bool:
    base_name, _ = _split_dimension_granularity(name.strip().lower())
    return base_name == "metric_time"


def _dimension_type(row: dict) -> str:
    return str(row.get("type") or row.get("dimension_type") or "").lower()


def _dimension_names_by_lookup(rows: List[dict]) -> Dict[str, str]:
    names = {}
    for row in rows:
        name = str(row.get("name") or "").strip()
        if name:
            names[name.lower()] = name
    return names


def _dimension_supported(requested_dimension: str, rows: List[dict]) -> bool:
    name = requested_dimension.strip().lower()
    if not name:
        return True

    names = _dimension_names_by_lookup(rows)
    if name in names:
        return True

    base_name, granularity = _split_dimension_granularity(name)
    if not granularity or base_name not in names:
        return False

    # If adapter metadata marks the base as non-time, do not treat a
    # granularity suffix as a valid alias. Missing type is intentionally
    # permissive so older adapters can still delegate final validation.
    for row in rows:
        row_name = str(row.get("name") or "").strip().lower()
        if row_name != base_name:
            continue
        dim_type = _dimension_type(row)
        return not dim_type or "time" in dim_type
    return False


def _serialize_validation_issue(issue) -> dict:
    if hasattr(issue, "model_dump"):
        issue_data = issue.model_dump(mode="json")
    else:
        issue_data = {"severity": "error", "message": str(issue)}

    severity = issue_data.get("severity")
    if severity is not None:
        issue_data["severity"] = str(severity).lower()
    return issue_data


def _is_no_metrics_present_issue(issue: dict) -> bool:
    message = str(issue.get("message") or "")
    return NO_METRICS_PRESENT_MESSAGE in message


def _validation_has_errors(issues: List[dict]) -> bool:
    return any(str(issue.get("severity") or "").lower() == "error" for issue in issues)


def _format_validation_error(issues: List[dict]) -> str:
    count = len(issues)
    if count == 0:
        return "0 validation errors"

    messages = []
    for issue in issues[:3]:
        message = str(issue.get("message") or "").strip()
        if message:
            messages.append(message)

    if not messages:
        return f"{count} validation errors"

    suffix = f"; ... {count - len(messages)} more" if count > len(messages) else ""
    return f"{count} validation errors: {'; '.join(messages)}{suffix}"


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
        generation_evidence: Optional[GenerationEvidence] = None,
    ):
        """
        Initialize semantic function tool.

        Args:
            agent_config: Agent configuration
            sub_agent_name: Optional sub-agent name for scoped storage
            adapter_type: Optional adapter type (e.g., "metricflow"). If not provided, tools will use storage only.
            generation_evidence: Optional shared tracker for validate_semantic and query_metrics(dry_run=True)
                publish-gate evidence.
        """
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.adapter_type = adapter_type
        self.generation_evidence = generation_evidence

        # Initialize storage RAG interfaces
        self.semantic_model_rag = SemanticModelRAG(agent_config, sub_agent_name)
        self.metric_rag = MetricRAG(agent_config, sub_agent_name)
        self.compressor = DataCompressor(model_name=agent_config.active_model().model)

        # Lazy load adapter and attribution tool
        self._adapter: Optional[BaseSemanticAdapter] = None
        self._attribution_tool: Optional[DimensionAttributionUtil] = None
        self._adapter_load_error: Optional[str] = None

    def _configured_adapter_type(self) -> Optional[str]:
        """Return the configured adapter type without instantiating the adapter."""
        if self.adapter_type:
            return self.adapter_type

        resolver = getattr(self.agent_config, "resolve_semantic_adapter", None)
        if not callable(resolver):
            return None

        try:
            resolved_adapter = resolver(self.adapter_type)
        except Exception as e:
            logger.debug(f"No semantic adapter configuration available: {e}")
            return None

        if resolved_adapter:
            self.adapter_type = resolved_adapter
        return resolved_adapter

    def _extract_db_config(self, datasource: str) -> Optional[dict]:
        """Extract db_config dict from the selected database config."""
        try:
            db_config_obj = self.agent_config.current_db_config(datasource)
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
                    datasource = self.agent_config.current_datasource
                    db_config = self._extract_db_config(datasource)
                    semantic_models_path = str(self.agent_config.path_manager.semantic_model_path(datasource))

                    if metadata and metadata.config_class:
                        adapter_config = metadata.config_class(
                            datasource=datasource,
                            db_config=db_config,
                            semantic_models_path=semantic_models_path,
                        )
                    else:
                        from datus.tools.semantic_tools.config import SemanticAdapterConfig

                        adapter_config = SemanticAdapterConfig(datasource=datasource)
                elif isinstance(adapter_config, dict):
                    if metadata and metadata.config_class:
                        adapter_config = metadata.config_class(**adapter_config)
                    else:
                        from datus.tools.semantic_tools.config import SemanticAdapterConfig

                        adapter_config = SemanticAdapterConfig(**adapter_config)

                self.adapter_type = resolved_adapter
                self._adapter = semantic_adapter_registry.create_adapter(resolved_adapter, adapter_config)
                self._adapter_load_error = None
                logger.info(f"Loaded semantic adapter: {resolved_adapter}")
            except Exception as e:
                logger.warning(f"Failed to load semantic adapter '{self.adapter_type}': {e}")
                self._adapter_load_error = str(e)
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

        # Add validation whenever an adapter is configured, even if the current
        # YAML makes adapter construction fail. In that case validate_semantic
        # returns the adapter-load error so the agent can fix the files.
        if self._configured_adapter_type():
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

    def _preflight_query_dimensions(
        self,
        metrics: List[str],
        dimensions: List[str],
        path: Optional[List[str]],
    ) -> Optional[FuncToolResult]:
        if not dimensions:
            return None

        metric_time_dimensions = list(dict.fromkeys(d for d in dimensions if _is_metric_time_dimension(d)))
        checked_dimensions = [d for d in dimensions if not _is_metric_time_dimension(d)]
        if not checked_dimensions:
            return None

        dimensions_by_metric: Dict[str, List[dict]] = {}
        dimension_names_by_metric: Dict[str, List[str]] = {}
        try:
            for metric_name in metrics:
                raw_dimensions = _run_async(self.adapter.get_dimensions(metric_name=metric_name, path=path or None))
                rows = _normalize_dimension_rows(raw_dimensions)
                dimensions_by_metric[metric_name] = rows
                dimension_names_by_metric[metric_name] = sorted(
                    _dimension_names_by_lookup(rows).values(), key=str.lower
                )
        except Exception as e:
            logger.debug(f"Skipping query_metrics dimension preflight: {e}")
            return None

        invalid_dimensions = []
        for dimension in checked_dimensions:
            unsupported_metrics = [
                metric_name
                for metric_name, rows in dimensions_by_metric.items()
                if not _dimension_supported(dimension, rows)
            ]
            if not unsupported_metrics:
                continue
            invalid_dimensions.append(
                {
                    "name": dimension,
                    "unsupported_metrics": unsupported_metrics,
                    "supported_metrics": [m for m in metrics if m not in unsupported_metrics],
                }
            )

        if not invalid_dimensions:
            return None

        common_dimensions: Optional[Set[str]] = None
        for rows in dimensions_by_metric.values():
            metric_dimensions = set(_dimension_names_by_lookup(rows).keys())
            common_dimensions = (
                metric_dimensions if common_dimensions is None else common_dimensions & metric_dimensions
            )

        suggested_groups: Dict[tuple[str, ...], List[str]] = {}
        for metric_name, rows in dimensions_by_metric.items():
            supported_requested_dimensions = tuple(
                dimension for dimension in checked_dimensions if _dimension_supported(dimension, rows)
            )
            suggested_groups.setdefault(supported_requested_dimensions, []).append(metric_name)

        suggestions = [
            {
                "metrics": group_metrics,
                "dimensions": list(dict.fromkeys([*metric_time_dimensions, *group_dimensions])),
            }
            for group_dimensions, group_metrics in suggested_groups.items()
        ]
        common_dimension_names = list(dict.fromkeys([*metric_time_dimensions, *sorted(common_dimensions or [])]))

        invalid_names = ", ".join(item["name"] for item in invalid_dimensions)
        return FuncToolResult(
            success=0,
            error=(
                "query_metrics dimension preflight failed: requested dimension(s) "
                f"{invalid_names} are not supported by all requested metrics. "
                "Use only common dimensions for a multi-metric query, or split the query by compatible metric groups."
            ),
            result={
                "metrics": metrics,
                "requested_dimensions": dimensions,
                "invalid_dimensions": invalid_dimensions,
                "common_dimensions": common_dimension_names,
                "dimensions_by_metric": dimension_names_by_metric,
                "suggested_metric_groups": suggestions,
            },
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
        metrics = _normalize_name_list(metrics)
        dimensions = _normalize_name_list(dimensions)
        path = _normalize_name_list(path)
        order_by = _normalize_name_list(order_by)

        if not metrics:
            return FuncToolResult(
                success=0,
                error=(
                    "query_metrics requires at least one metric name. "
                    "Call list_metrics first and pass one or more metric names exactly as returned."
                ),
            )

        if not self.adapter:
            return FuncToolResult(
                success=0,
                error="No semantic adapter configured. Cannot execute queries without adapter.",
            )

        # Sanitize time parameters: LLM may pass string "null"/"None" instead of omitting
        time_start = normalize_null(time_start)
        time_end = normalize_null(time_end)
        time_granularity = normalize_null(time_granularity)
        where = normalize_null(where)
        logger.info(
            f"query_metrics called: metrics={metrics}, dimensions={dimensions}, path={path}, "
            f"time=[{time_start},{time_end}], granularity={time_granularity}, where={where}, "
            f"limit={limit}, dry_run={dry_run}"
        )

        try:
            preflight_result = self._preflight_query_dimensions(metrics=metrics, dimensions=dimensions, path=path)
            if preflight_result is not None:
                return preflight_result

            # Execute query via adapter
            result = _run_async(
                self.adapter.query_metrics(
                    metrics=metrics,
                    dimensions=dimensions,
                    path=path or None,
                    time_start=time_start,
                    time_end=time_end,
                    time_granularity=time_granularity,
                    where=where,
                    limit=limit,
                    order_by=order_by or None,
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

            tool_result = FuncToolResult(
                success=1,
                result=result_dict,
            )
            if dry_run and self.generation_evidence:
                self.generation_evidence.record_metric_dry_run(
                    metrics,
                    tool_result,
                    dimensions=dimensions,
                    time_granularity=time_granularity,
                )
            return tool_result

        except Exception as e:
            logger.error(f"Error querying metrics: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to query metrics: {str(e)}",
            )

    def validate_semantic(self, scope: Literal["all", "semantic_model"] = "all") -> FuncToolResult:
        """
        Validate semantic layer configuration (requires adapter).

        After successful validation, the adapter is reloaded to pick up any new
        metrics or semantic model changes. This ensures that subsequent calls to
        query_metrics can find newly created metrics.

        Args:
            scope: Validation scope. Use "all" for full semantic-layer validation,
                including metrics. Use "semantic_model" when generating semantic
                models before metric definitions exist; this still fails on real
                semantic model errors but ignores the expected no-metrics issue.

        Returns:
            FuncToolResult with validation status and issues
        """
        scope = normalize_null(scope) or "all"
        if scope not in ("all", "semantic_model"):
            return FuncToolResult(
                success=0,
                error="scope must be one of: all, semantic_model",
                result=None,
            )

        logger.info(f"validate_semantic called scope={scope}")
        adapter = self.adapter
        if not adapter:
            if self._adapter_load_error:
                return FuncToolResult(
                    success=0,
                    error=f"Failed to load semantic adapter '{self.adapter_type}': {self._adapter_load_error}",
                    result=None,
                )
            return FuncToolResult(
                success=0,
                error="No semantic adapter configured. Cannot validate without adapter.",
                result=None,
            )

        try:
            validate_semantic = adapter.validate_semantic
            validation_kwargs = {}
            if scope != "all":
                try:
                    signature = inspect.signature(validate_semantic)
                    if "scope" in signature.parameters:
                        validation_kwargs["scope"] = scope
                    elif "validation_scope" in signature.parameters:
                        validation_kwargs["validation_scope"] = scope
                except (TypeError, ValueError):
                    validation_kwargs = {}

            validation_result = _run_async(validate_semantic(**validation_kwargs))

            # Serialize ValidationIssue objects to dicts
            issues_data = [_serialize_validation_issue(issue) for issue in validation_result.issues]

            ignored_issues = []
            effective_issues = issues_data
            if scope == "semantic_model":
                ignored_issues = [issue for issue in issues_data if _is_no_metrics_present_issue(issue)]
                effective_issues = [issue for issue in issues_data if not _is_no_metrics_present_issue(issue)]

            effective_valid = validation_result.valid or (
                scope == "semantic_model" and not _validation_has_errors(effective_issues)
            )

            if issues_data:
                logger.warning(
                    "Semantic validation issues scope=%s valid=%s effective_valid=%s issues=%s ignored=%s",
                    scope,
                    validation_result.valid,
                    effective_valid,
                    json.dumps(effective_issues, ensure_ascii=False),
                    json.dumps(ignored_issues, ensure_ascii=False),
                )

            # If validation succeeded, reload the adapter to pick up new metrics
            if effective_valid:
                logger.info("Validation succeeded, reloading adapter to pick up new metrics...")
                self._reload_adapter()

            tool_result = FuncToolResult(
                success=1 if effective_valid else 0,
                result={
                    "valid": effective_valid,
                    "issues": effective_issues,
                    "scope": scope,
                    "ignored_issues": ignored_issues,
                },
                error=None if effective_valid else _format_validation_error(effective_issues),
            )
            if self.generation_evidence:
                self.generation_evidence.record_validation_result(tool_result)
            return tool_result

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
