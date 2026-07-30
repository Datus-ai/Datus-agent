# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Function Tools

Provides unified interface to semantic layer services through adapters.
All public semantic tools require a successfully initialized semantic adapter.
"""

import csv
import hashlib
import inspect
import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Set, Tuple

from agents import Tool
from pydantic import BaseModel

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


def _normalize_metric_metadata(raw) -> dict:
    """Keep adapter-provided metric metadata only when it is tool-safe."""
    if not isinstance(raw, dict):
        return {}

    safe_metadata = {}
    for key, value in raw.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        safe_metadata[str(key)] = value
    return safe_metadata


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


def _normalize_validation_checks(value) -> Optional[List[str]]:
    """Normalize optional adapter validation check names."""
    value = normalize_null(value)
    if value is None:
        return None
    candidates: List[Any]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                candidates = parsed
            else:
                candidates = [part.strip() for part in text.split(",")]
        else:
            candidates = [part.strip() for part in text.split(",")]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = [value]

    checks = []
    for candidate in candidates:
        candidate = normalize_null(candidate)
        if candidate is None:
            continue
        check = str(candidate).strip()
        if check:
            checks.append(check)
    return checks or None


def _normalize_optional_path(value) -> Optional[List[str]]:
    """Normalize optional subject paths and drop null placeholders."""
    names = _normalize_name_list(value)
    return names or None


def _signature_accepts_parameter(parameters, name: str) -> bool:
    """Return true when a callable explicitly accepts ``name`` or arbitrary kwargs."""
    return name in parameters or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())


_TIME_GRANULARITY_ORDER = ("day", "week", "month", "quarter", "year")
_TIME_GRANULARITIES = set(_TIME_GRANULARITY_ORDER)


def extract_time_query_capabilities(raw_dimensions) -> Dict[str, Any]:
    """Extract the metric-level time contract carried by ``get_dimensions``."""
    candidates = []
    for dimension in raw_dimensions or []:
        if isinstance(dimension, dict):
            name = dimension.get("name")
            is_primary_time = dimension.get("is_primary_time")
            raw_granularities = dimension.get("time_granularities")
        else:
            name = getattr(dimension, "name", None)
            is_primary_time = getattr(dimension, "is_primary_time", None)
            raw_granularities = getattr(dimension, "time_granularities", None)

        granularities = {
            str(granularity).strip().lower()
            for granularity in _normalize_name_list(raw_granularities)
            if str(granularity).strip().lower() in _TIME_GRANULARITIES
        }
        ordered_granularities = [granularity for granularity in _TIME_GRANULARITY_ORDER if granularity in granularities]
        if name and (is_primary_time or ordered_granularities):
            candidates.append(
                (
                    bool(is_primary_time),
                    str(name),
                    ordered_granularities,
                )
            )

    if not candidates:
        return {"time_dimension": None, "time_granularities": []}

    _, time_dimension, time_granularities = next(
        (candidate for candidate in candidates if candidate[0]),
        candidates[0],
    )
    return {
        "time_dimension": time_dimension,
        "time_granularities": time_granularities,
    }


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


class _CompactValidationIssue(BaseModel):
    """Bounded validation issue returned by the semantic tool."""

    severity: str
    message: str
    location: Any | None = None


def _compact_validation_issues(
    issues: List[dict],
    *,
    limit: int = 8,
    message_limit: int = 600,
) -> List[_CompactValidationIssue]:
    """Keep validation tool output bounded while full details remain in logs."""
    compact: List[_CompactValidationIssue] = []
    for issue in issues[:limit]:
        message = str(issue.get("message") or "").strip()
        if len(message) > message_limit:
            message = f"{message[:message_limit]}... [truncated]"
        compact.append(
            _CompactValidationIssue(
                severity=str(issue.get("severity") or "error").lower(),
                message=message,
                location=issue.get("location"),
            )
        )
    if len(issues) > limit:
        compact.append(
            _CompactValidationIssue(
                severity="warning",
                message=f"{len(issues) - limit} additional validation issue(s) omitted; see logs for details.",
            )
        )
    return compact


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

    permission_category: str = "semantic_tools"

    MAX_QUERY_METRICS_RESULT_CACHE_SIZE = 100

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
        runtime_db_context_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
        warehouse_dry_run_provider: Optional[Callable[[str], Mapping[str, Any]]] = None,
    ):
        """
        Initialize semantic function tool.

        Args:
            agent_config: Agent configuration
            sub_agent_name: Optional sub-agent name for scoped storage
            adapter_type: Optional adapter type (e.g., "metricflow"). If not provided, tools will use storage only.
            generation_evidence: Optional shared tracker for validate_semantic and query_metrics(dry_run=True)
                publish-gate evidence.
            runtime_db_context_provider: Optional callback that returns the per-turn datasource/catalog/database/schema
                context used to initialize the semantic adapter.
            warehouse_dry_run_provider: Optional host callback that validates
                adapter-compiled SQL against the active warehouse.
        """
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.adapter_type = adapter_type
        self.generation_evidence = generation_evidence
        self._runtime_db_context_provider = runtime_db_context_provider
        self._warehouse_dry_run_provider = warehouse_dry_run_provider
        self._runtime_db_context_static: Dict[str, str] = {}
        self._runtime_db_context_static_set = False

        # Keep storage handles for compatibility with older call sites, but
        # public SemanticTools methods use the semantic adapter as their source
        # of truth. ContextSearchTools owns RAG/storage discovery.
        self.semantic_model_rag = SemanticModelRAG(agent_config, sub_agent_name)
        self.metric_rag = MetricRAG(agent_config, sub_agent_name)
        self.compressor = DataCompressor(model_name=agent_config.active_model().model)
        self._query_metrics_result_cache: OrderedDict[str, dict] = OrderedDict()
        self._query_metrics_result_cache_counter = 0

        # Lazy load adapter and attribution tool
        self._adapter: Optional[BaseSemanticAdapter] = None
        self._attribution_tool: Optional[DimensionAttributionUtil] = None
        self._adapter_load_error: Optional[str] = None
        self._adapter_context_key: Optional[Tuple[str, str, str, str, str]] = None

    @staticmethod
    def _query_data_row_count(data: Any) -> int:
        if data is None:
            return 0
        if hasattr(data, "num_rows"):
            try:
                return int(data.num_rows)
            except (TypeError, ValueError):
                return 0
        if hasattr(data, "shape"):
            try:
                return int(data.shape[0])
            except (TypeError, ValueError, IndexError):
                return 0
        try:
            return len(data)
        except TypeError:
            return 0

    @staticmethod
    def _query_data_to_csv(columns: List[str], data: Any) -> str:
        if data is None:
            return ""

        if hasattr(data, "to_pandas"):
            data = data.to_pandas()

        if hasattr(data, "to_csv"):
            return data.to_csv(index=False)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(columns)
        for row in data or []:
            if isinstance(row, dict):
                writer.writerow([row.get(column, "") for column in columns])
            elif isinstance(row, (list, tuple)):
                writer.writerow(row)
            else:
                writer.writerow([row])
        return buf.getvalue()

    def _cache_query_metrics_result(self, columns: List[str], data: Any) -> Optional[str]:
        if data is None:
            return None
        full_csv = self._query_data_to_csv(columns, data)
        if not full_csv:
            return None

        self._query_metrics_result_cache_counter += 1
        cache_key = f"query_metrics:{self._query_metrics_result_cache_counter}"
        self._query_metrics_result_cache[cache_key] = {
            "columns": list(columns),
            "csv": full_csv,
            "row_count": self._query_data_row_count(data),
        }
        while len(self._query_metrics_result_cache) > self.MAX_QUERY_METRICS_RESULT_CACHE_SIZE:
            self._query_metrics_result_cache.popitem(last=False)
        return cache_key

    def get_cached_query_metrics_result(self, cache_key: str) -> Optional[dict]:
        return self._query_metrics_result_cache.get(cache_key)

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

    def _semantic_model_artifact_evidence(self, semantic_model_name: str) -> Dict[str, str]:
        """Return exact Ossie artifact identity for target-bound validation evidence."""
        if str(self.adapter_type or "").strip().lower() != "osi" or not semantic_model_name:
            return {}
        try:
            from datus.agent.node.semantic_authoring import discover_osi_semantic_models

            matches = [
                model
                for model in discover_osi_semantic_models(self.agent_config)
                if str(model.get("semantic_model_name") or "") == semantic_model_name
            ]
            if len(matches) != 1:
                return {}
            path = Path(str(matches[0]["absolute_path"])).expanduser().resolve(strict=True)
            return {
                "semantic_model_name": semantic_model_name,
                "semantic_model_file": str(path),
                "semantic_model_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        except (KeyError, OSError, RuntimeError):
            return {}

    @staticmethod
    def _normalize_runtime_db_context(runtime_db_context: Optional[Mapping[str, Any]]) -> Dict[str, str]:
        if not runtime_db_context:
            return {}

        normalized: Dict[str, str] = {}
        for key in (
            "datasource",
            "catalog",
            "catalog_name",
            "database",
            "database_name",
            "schema",
            "db_schema",
            "schema_name",
        ):
            value = runtime_db_context.get(key)
            if value is None:
                continue
            text = value.strip() if isinstance(value, str) else str(value).strip()
            if text:
                normalized[key] = text

        if "catalog" not in normalized and "catalog_name" in normalized:
            normalized["catalog"] = normalized["catalog_name"]
        if "database" not in normalized and "database_name" in normalized:
            normalized["database"] = normalized["database_name"]
        if "schema" not in normalized:
            if "db_schema" in normalized:
                normalized["schema"] = normalized["db_schema"]
            elif "schema_name" in normalized:
                normalized["schema"] = normalized["schema_name"]
        return normalized

    def set_runtime_db_context(self, runtime_db_context: Optional[Mapping[str, Any]]) -> None:
        """Set a static runtime DB context and invalidate any adapter built for the old context."""
        normalized = self._normalize_runtime_db_context(runtime_db_context)
        if normalized == self._runtime_db_context_static and self._runtime_db_context_static_set:
            return
        self._runtime_db_context_static = normalized
        self._runtime_db_context_static_set = True
        self._adapter = None
        self._attribution_tool = None
        self._adapter_context_key = None

    def _runtime_db_context(self) -> Dict[str, str]:
        if callable(self._runtime_db_context_provider):
            try:
                return self._normalize_runtime_db_context(self._runtime_db_context_provider())
            except Exception as e:
                logger.debug("Failed to resolve runtime DB context for semantic adapter: %s", e)
                return {}
        if self._runtime_db_context_static_set:
            return dict(self._runtime_db_context_static)
        runtime_context_getter = getattr(self.agent_config, "runtime_db_context", None)
        if callable(runtime_context_getter):
            try:
                return self._normalize_runtime_db_context(runtime_context_getter())
            except Exception as e:
                logger.debug("Failed to resolve AgentConfig runtime DB context for semantic adapter: %s", e)
        return {}

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
            if v is not None and v != "" and k not in ("extra", "path_pattern", "default")
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
        try:
            resolved_adapter = self.adapter_type
            resolver = getattr(self.agent_config, "resolve_semantic_adapter", None)
            if callable(resolver):
                resolved_adapter = resolver(self.adapter_type)
            if not resolved_adapter:
                return None

            runtime_db_context = self._runtime_db_context()
            datasource = runtime_db_context.get("datasource") or self.agent_config.current_datasource
            context_key = (
                resolved_adapter,
                datasource or "",
                runtime_db_context.get("catalog", ""),
                runtime_db_context.get("database", ""),
                runtime_db_context.get("schema", ""),
            )
            if self._adapter is not None:
                if self._adapter_context_key is None or self._adapter_context_key == context_key:
                    return self._adapter
                self._adapter = None
                self._attribution_tool = None
                self._adapter_context_key = None

            metadata = semantic_adapter_registry.get_metadata(resolved_adapter)
            builder = getattr(self.agent_config, "build_semantic_adapter_config", None)
            adapter_config = None
            if callable(builder):
                builder_kwargs: Dict[str, Any] = {}
                try:
                    builder_params = inspect.signature(builder).parameters
                    if "database_name" in builder_params:
                        builder_kwargs["database_name"] = datasource or None
                    if "runtime_db_context" in builder_params:
                        builder_kwargs["runtime_db_context"] = runtime_db_context
                except (TypeError, ValueError):
                    pass
                adapter_config = builder(resolved_adapter, **builder_kwargs)
            if adapter_config is None:
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
            self._adapter_context_key = context_key
            self._adapter_load_error = None
            logger.info(f"Loaded semantic adapter: {resolved_adapter}")
        except Exception as e:
            logger.warning(f"Failed to load semantic adapter '{self.adapter_type}': {e}")
            self._adapter_load_error = str(e)
            self._adapter = None
            self._adapter_context_key = None
        return self._adapter

    @property
    def attribution_tool(self) -> Optional[DimensionAttributionUtil]:
        """Lazy load attribution tool when adapter is available."""
        if self._attribution_tool is None and self.adapter is not None:
            self._attribution_tool = DimensionAttributionUtil(self.adapter)
        return self._attribution_tool

    def _adapter_unavailable_message(self) -> str:
        """Return a consistent message for semantic-adapter failures."""
        if self._adapter_load_error:
            adapter_name = self.adapter_type or "configured"
            return f"Semantic adapter unavailable: failed to load '{adapter_name}': {self._adapter_load_error}"

        adapter_name = self._configured_adapter_type()
        if not adapter_name:
            return "Semantic adapter unavailable: no semantic adapter configured."

        return f"Semantic adapter unavailable: failed to load '{adapter_name}'."

    def _require_adapter(self, tool_name: str) -> tuple[Optional[BaseSemanticAdapter], Optional[FuncToolResult]]:
        """Load the semantic adapter or return a tool failure result."""
        adapter = self.adapter
        if adapter is not None:
            return adapter, None
        return None, FuncToolResult(
            success=0,
            error=f"{tool_name} requires a successfully initialized semantic adapter. "
            f"{self._adapter_unavailable_message()}",
        )

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
            self._adapter_context_key = None

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
        if not self._configured_adapter_type():
            logger.warning("SemanticTools unavailable: %s", self._adapter_unavailable_message())
            return []

        return [
            trans_to_function_tool(self.list_metrics),
            trans_to_function_tool(self.get_dimensions),
            trans_to_function_tool(self.query_metrics),
            trans_to_function_tool(self.validate_semantic),
            trans_to_function_tool(self.attribution_analyze),
        ]

    def list_metrics(
        self,
        path: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> FuncToolResult:
        """
        List available metrics from the semantic adapter.

        Args:
            path: Optional subject tree path filter (e.g., ["Finance", "Revenue"])
            limit: Maximum number of metrics to return
            offset: Number of metrics to skip

        Returns:
            FuncToolResult with result as FuncToolListResult:
              - items (List[Dict]): metric rows, each with name, description, type,
                dimensions, measures, unit, format, path, metadata
              - total (int | None): full metric count before pagination
              - has_more (bool | None): True when offset + len(items) < total
              - extra (dict | None): {"next_offset": int} when has_more is True

            Pagination: call again with offset=extra.next_offset until
            has_more is False. Default limit=100; override if you need bigger
            pages. list_metrics never compresses — use the limit to control
            response size.
        """
        # Normalize null values from LLM
        path = _normalize_optional_path(path)
        logger.info(f"list_metrics called: path={path}, limit={limit}, offset={offset}")
        adapter, error = self._require_adapter("list_metrics")
        if error:
            return error

        try:
            async_result = _run_async(adapter.list_metrics(path=path, limit=limit, offset=offset))
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
                    "metadata": _normalize_metric_metadata(getattr(m, "metadata", None)),
                }
                for m in async_result
            ]
            # Adapter path has no guaranteed upstream total — leave it None so consumers
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
        Returns dimension objects from the semantic adapter.

        Args:
            metric_name: Name of the metric
            path: Optional subject tree path (e.g., ["Finance", "Revenue"])

        Returns:
            FuncToolResult with result as FuncToolListResult:
              - items (List[Dict]): dimension rows. Adapter dimensions expose
                their full schema (name, type, expr, ...); storage dimensions
                fall back to a minimal {"name": ...} shape when only names are
                stored.
              - total, has_more: dimensions isn't paginated, so total equals
                len(items) and has_more is False.
              - extra.time_dimension: canonical metric time dimension, or None.
              - extra.time_granularities: legal grains ordered finest to
                coarsest; the first item is the default. Empty when the metric
                has no discoverable time contract.
        """
        # Normalize null values from LLM
        path = _normalize_optional_path(path)
        logger.info(f"get_dimensions called: metric={metric_name}, path={path}")
        adapter, error = self._require_adapter("get_dimensions")
        if error:
            return error

        try:
            dimensions = _run_async(adapter.get_dimensions(metric_name=metric_name, path=path))
            items = _normalize_dimension_rows(dimensions)
            extra = extract_time_query_capabilities(dimensions)
            for item in items:
                item.pop("is_primary_time", None)
                item.pop("time_granularities", None)
            return FuncToolResult(
                success=1,
                result=FuncToolListResult(
                    items=items,
                    total=len(items),
                    has_more=False,
                    extra=extra,
                ).model_dump(),
            )

        except Exception as e:
            logger.error(f"Error getting dimensions: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to get dimensions: {str(e)}",
            )

    def _load_dimensions_by_metric(
        self,
        metrics: List[str],
        path: Optional[List[str]],
    ) -> Optional[Dict[str, List[dict]]]:
        try:
            return {
                metric_name: _normalize_dimension_rows(
                    _run_async(
                        self.adapter.get_dimensions(
                            metric_name=metric_name,
                            path=path or None,
                        )
                    )
                )
                for metric_name in metrics
            }
        except Exception as e:
            logger.debug(f"Skipping query_metrics metadata preflight: {e}")
            return None

    def _preflight_query_dimensions(
        self,
        metrics: List[str],
        dimensions: List[str],
        path: Optional[List[str]],
        dimensions_by_metric: Optional[Dict[str, List[dict]]] = None,
    ) -> Optional[FuncToolResult]:
        if not dimensions:
            return None

        metric_time_dimensions = list(dict.fromkeys(d for d in dimensions if _is_metric_time_dimension(d)))
        checked_dimensions = [d for d in dimensions if not _is_metric_time_dimension(d)]
        if not checked_dimensions:
            return None

        dimensions_by_metric = (
            dimensions_by_metric if dimensions_by_metric is not None else self._load_dimensions_by_metric(metrics, path)
        )
        if dimensions_by_metric is None:
            return None
        dimension_names_by_metric = {
            metric_name: sorted(_dimension_names_by_lookup(rows).values(), key=str.lower)
            for metric_name, rows in dimensions_by_metric.items()
        }

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

    @staticmethod
    def _preflight_time_granularity(
        metrics: List[str],
        dimensions: List[str],
        time_granularity: Optional[str],
        dimensions_by_metric: Optional[Dict[str, List[dict]]],
    ) -> Optional[FuncToolResult]:
        requested_grain = str(time_granularity or "").strip().lower()
        if dimensions_by_metric is None:
            return None

        capabilities_by_metric = {
            metric_name: extract_time_query_capabilities(rows) for metric_name, rows in dimensions_by_metric.items()
        }
        known_capabilities = {
            metric_name: capability
            for metric_name, capability in capabilities_by_metric.items()
            if capability["time_granularities"]
        }
        if not known_capabilities:
            return None

        if not requested_grain:
            time_dimension_names = {
                str(capability["time_dimension"]).lower()
                for capability in known_capabilities.values()
                if capability["time_dimension"]
            }
            for dimension in dimensions:
                base_name, dimension_grain = _split_dimension_granularity(dimension.strip().lower())
                if dimension_grain and (_is_metric_time_dimension(dimension) or base_name in time_dimension_names):
                    requested_grain = dimension_grain
                    break
        if not requested_grain:
            return None

        unsupported_metrics = [
            metric_name
            for metric_name, capability in known_capabilities.items()
            if requested_grain not in capability["time_granularities"]
        ]
        if not unsupported_metrics:
            return None

        common_granularities = [
            grain
            for grain in _TIME_GRANULARITY_ORDER
            if all(grain in capability["time_granularities"] for capability in known_capabilities.values())
        ]
        suggested_grain = common_granularities[0] if common_granularities else None
        details = "; ".join(
            f"{metric}: {', '.join(capability['time_granularities'])}"
            for metric, capability in known_capabilities.items()
        )
        suggested_retry = (
            {
                "metrics": metrics,
                "time_granularity": suggested_grain,
            }
            if suggested_grain
            else None
        )
        return FuncToolResult(
            success=0,
            error=(
                "query_metrics time granularity preflight failed: "
                f"`{requested_grain}` is not supported by "
                f"{', '.join(unsupported_metrics)}. Supported granularities: "
                f"{details}."
            ),
            result={
                "error_type": "semantic_validation_error",
                "code": "unsupported_time_granularity",
                "metrics": metrics,
                "requested_time_granularity": requested_grain,
                "required_time_granularity": suggested_grain,
                "time_granularities": common_granularities,
                "time_capabilities_by_metric": capabilities_by_metric,
                "suggested_retry": suggested_retry,
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
            dry_run: If True, compile and return the query plan. Live OSI
                backends also validate the compiled SQL with a warehouse dry-run.

        Returns:
            FuncToolResult with query results or explain plan
        """
        metrics = _normalize_name_list(metrics)
        dimensions = _normalize_name_list(dimensions)
        path = _normalize_name_list(path)
        order_by = _normalize_name_list(order_by)

        adapter, error = self._require_adapter("query_metrics")
        if error:
            return error

        if not metrics:
            return FuncToolResult(
                success=0,
                error=(
                    "query_metrics requires at least one metric name. "
                    "Call list_metrics first and pass one or more metric names exactly as returned."
                ),
            )

        # Sanitize time parameters: LLM may pass string "null"/"None" instead of omitting
        path = _normalize_optional_path(path)
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
            needs_dimension_metadata = (
                bool(time_granularity)
                or any(not _is_metric_time_dimension(dimension) for dimension in dimensions)
                or any(_split_dimension_granularity(dimension)[1] for dimension in dimensions)
            )
            dimensions_by_metric = self._load_dimensions_by_metric(metrics, path) if needs_dimension_metadata else None
            preflight_result = self._preflight_query_dimensions(
                metrics=metrics,
                dimensions=dimensions,
                path=path,
                dimensions_by_metric=dimensions_by_metric,
            )
            if preflight_result is not None:
                return preflight_result
            preflight_result = self._preflight_time_granularity(
                metrics=metrics,
                dimensions=dimensions,
                time_granularity=time_granularity,
                dimensions_by_metric=dimensions_by_metric,
            )
            if preflight_result is not None:
                return preflight_result

            # Execute query via adapter
            adapter_query_kwargs = {
                "metrics": metrics,
                "dimensions": dimensions,
                "path": path or None,
                "time_start": time_start,
                "time_end": time_end,
                "time_granularity": time_granularity,
                "where": where,
                "limit": limit,
                "order_by": order_by or None,
                "dry_run": dry_run,
            }
            result = _run_async(adapter.query_metrics(**adapter_query_kwargs))

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
            warehouse_error = None
            if (
                dry_run
                and callable(self._warehouse_dry_run_provider)
                and not (
                    isinstance(safe_metadata.get("warehouse_dry_run"), dict)
                    and safe_metadata["warehouse_dry_run"].get("status") == "success"
                )
            ):
                sql = str(safe_metadata.get("sql") or "").strip()
                if not sql:
                    warehouse_evidence: Mapping[str, Any] = {
                        "status": "failed",
                        "error": "Semantic adapter dry-run did not return compiled SQL.",
                    }
                else:
                    try:
                        warehouse_evidence = self._warehouse_dry_run_provider(sql)
                    except Exception as exc:
                        warehouse_evidence = {"status": "failed", "error": str(exc)}
                safe_metadata["warehouse_dry_run"] = dict(warehouse_evidence)
                if warehouse_evidence.get("status") != "success":
                    warehouse_error = str(warehouse_evidence.get("error") or "Warehouse EXPLAIN failed.")
            cache_key = None
            if not dry_run:
                cache_key = self._cache_query_metrics_result(result.columns, result.data)
                if cache_key:
                    safe_metadata["_full_result_cache_key"] = cache_key
                    safe_metadata["_full_result_cached"] = True
                    safe_metadata["_full_result_row_count"] = self._query_data_row_count(result.data)
                    safe_metadata["_full_result_note"] = (
                        "The complete uncompressed query result is cached and will be used for final output; "
                        "do not re-query only because the returned data is a compressed preview."
                    )

            result_dict = {
                "result_id": cache_key,
                "columns": result.columns,
                "data": self.compressor.compress(result.data),
                "metadata": safe_metadata,
            }

            tool_result = FuncToolResult(
                success=0 if warehouse_error else 1,
                error=f"Warehouse dry-run failed: {warehouse_error}" if warehouse_error else None,
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
            # Surface backend validation rejections as structured planner guidance.
            # Duck-typed so the tool layer stays decoupled from any specific adapter.
            payload = getattr(e, "payload", None)
            if payload is not None and getattr(payload, "error_type", None) == "semantic_validation_error":
                data = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
                logger.info(f"query_metrics validation rejection: code={data.get('code')}")
                return FuncToolResult(
                    success=0,
                    error=getattr(payload, "message", "") or "query_metrics validation failed",
                    result=data,
                )
            logger.error(f"Error querying metrics: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to query metrics: {str(e)}",
            )

    def validate_semantic(
        self,
        scope: Literal["all", "semantic_model"] = "all",
        semantic_model_name: str = "",
        checks: Optional[List[str] | str] = None,
        baseline_artifact_json: str = "",
    ) -> FuncToolResult:
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
            semantic_model_name: Optional target model for scoped Ossie validation.
                Required when a datasource contains multiple semantic models.
            checks: Optional adapter-specific validation checks. Adapters that do
                not support named checks return an error when this is supplied.
            baseline_artifact_json: Optional JSON-encoded semantic artifact used
                by adapters that support mutation guard validation.

        Returns:
            FuncToolResult with validation status and issues
        """
        scope = normalize_null(scope) or "all"
        semantic_model_name = str(normalize_null(semantic_model_name) or "").strip()
        if scope not in ("all", "semantic_model"):
            return FuncToolResult(
                success=0,
                error="scope must be one of: all, semantic_model",
                result=None,
            )

        checks_list = _normalize_validation_checks(checks)
        baseline_artifact = None
        baseline_artifact_text = normalize_null(baseline_artifact_json)
        if baseline_artifact_text:
            try:
                baseline_artifact = json.loads(str(baseline_artifact_text))
            except (TypeError, json.JSONDecodeError) as e:
                return FuncToolResult(
                    success=0,
                    error=f"baseline_artifact_json must be valid JSON: {e}",
                    result=None,
                )
            if not isinstance(baseline_artifact, dict):
                return FuncToolResult(
                    success=0,
                    error="baseline_artifact_json must decode to a JSON object",
                    result=None,
                )

        logger.info(f"validate_semantic called scope={scope} checks={checks_list}")
        adapter, error = self._require_adapter("validate_semantic")
        if error:
            error.result = None
            return error

        try:
            validate_semantic = adapter.validate_semantic
            validation_kwargs = {}
            try:
                signature = inspect.signature(validate_semantic)
                params = signature.parameters
                if _signature_accepts_parameter(params, "scope"):
                    validation_kwargs["scope"] = scope
                elif scope != "all" and "validation_scope" in params:
                    validation_kwargs["validation_scope"] = scope
                if semantic_model_name:
                    if not _signature_accepts_parameter(params, "semantic_model_name"):
                        return FuncToolResult(
                            success=0,
                            error=(
                                "Targeted semantic-model validation is not supported by the current semantic adapter"
                            ),
                            result=None,
                        )
                    validation_kwargs["semantic_model_name"] = semantic_model_name
                if checks_list is not None:
                    if not _signature_accepts_parameter(params, "checks"):
                        return FuncToolResult(
                            success=0,
                            error="validate_semantic checks are not supported by the current semantic adapter",
                            result=None,
                        )
                    validation_kwargs["checks"] = checks_list
                if baseline_artifact is not None:
                    if not _signature_accepts_parameter(params, "baseline_artifact"):
                        return FuncToolResult(
                            success=0,
                            error="validate_semantic baseline_artifact is not supported by the current semantic adapter",
                            result=None,
                        )
                    validation_kwargs["baseline_artifact"] = baseline_artifact
            except (TypeError, ValueError):
                if checks_list is not None or baseline_artifact is not None:
                    return FuncToolResult(
                        success=0,
                        error="validate_semantic validation options are not supported by the current semantic adapter",
                        result=None,
                    )
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
                    "Semantic validation issues scope=%s valid=%s effective_valid=%s issues=%d ignored=%d",
                    scope,
                    validation_result.valid,
                    effective_valid,
                    len(effective_issues),
                    len(ignored_issues),
                )
                logger.debug(
                    "Full semantic validation issues=%s ignored=%s",
                    json.dumps(effective_issues, ensure_ascii=False),
                    json.dumps(ignored_issues, ensure_ascii=False),
                )

            # If validation succeeded, reload the adapter to pick up new metrics
            if effective_valid:
                logger.info("Validation succeeded, reloading adapter to pick up new metrics...")
                self._reload_adapter()

            compact_issues = [
                issue.model_dump(exclude_none=True) for issue in _compact_validation_issues(effective_issues)
            ]
            compact_ignored_issues = [
                issue.model_dump(exclude_none=True) for issue in _compact_validation_issues(ignored_issues)
            ]
            result_payload = {
                "valid": effective_valid,
                "issues": compact_issues,
                "scope": scope,
                "checks": checks_list,
                "ignored_issues": compact_ignored_issues,
                "issue_count": len(effective_issues),
                "ignored_issue_count": len(ignored_issues),
            }
            if effective_valid and semantic_model_name:
                result_payload.update(self._semantic_model_artifact_evidence(semantic_model_name))

            tool_result = FuncToolResult(
                success=1 if effective_valid else 0,
                result=result_payload,
                error=None if effective_valid else _format_validation_error(compact_issues),
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
        _, error = self._require_adapter("attribution_analyze")
        if error:
            return error

        attribution_tool = self.attribution_tool
        if not attribution_tool:
            return FuncToolResult(
                success=0,
                error="Attribution tool not available. Requires a successfully initialized semantic adapter.",
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
                attribution_tool.attribution_analyze(
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
