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
from copy import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple

from agents import Tool
from pydantic import BaseModel

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import MetricRAG
from datus.tools.func_tool.attribution_utils import (
    AttributionValidationException,
    DimensionAttributionUtil,
)
from datus.tools.func_tool.base import FuncToolListResult, FuncToolResult, normalize_null, trans_to_function_tool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.tools.semantic_tools.models import AnomalyContext
from datus.tools.semantic_tools.paging import (
    METRIC_CATALOG_MAX_PAGES,
    METRIC_CATALOG_PAGE_SIZE,
    metric_catalog_paging,
)
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


def _normalize_dataset_names(raw: Any) -> List[str]:
    """Dataset names an adapter reports for a metric, as a list of non-empty strings.

    A lone string names one dataset; iterating it would yield characters. Only
    string entries are kept — coercing anything else would invent names like
    "None" that a policy could match on.
    """
    if isinstance(raw, str):
        name = raw.strip()
        return [name] if name else []
    if isinstance(raw, (list, tuple, set)):
        return [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    return []


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


# Defaults for `metric_datasets` paging; override per adapter under
# `services.semantic_layer.<adapter>` with `metric_catalog_page_size` /
# `metric_catalog_max_pages`. The page is large because an adapter may rebuild
# its whole catalog per request, making each extra page cost a full pass.
_METRIC_DATASETS_PAGE_SIZE = METRIC_CATALOG_PAGE_SIZE
_METRIC_DATASETS_MAX_PAGES = METRIC_CATALOG_MAX_PAGES

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
        semantic_model_path_provider: Optional[Callable[[], Optional[str]]] = None,
        semantic_metric_names_provider: Optional[Callable[[], Optional[List[str]]]] = None,
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
            semantic_model_path_provider: Optional callback that returns the exact
                request-local semantic model selected for authoring or validation.
            semantic_metric_names_provider: Optional callback returning the
                present touched metrics that require compiled publish evidence.
        """
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.adapter_type = adapter_type
        self.generation_evidence = generation_evidence
        self._runtime_db_context_provider = runtime_db_context_provider
        self._warehouse_dry_run_provider = warehouse_dry_run_provider
        self._semantic_model_path_provider = semantic_model_path_provider
        self._semantic_metric_names_provider = semantic_metric_names_provider
        self._runtime_db_context_static: Dict[str, str] = {}
        self._runtime_db_context_static_set = False

        # Keep storage handles for compatibility with older call sites, but
        # public SemanticTools methods use the semantic adapter as their source
        # of truth. ContextSearchTools owns RAG/storage discovery.
        self.metric_rag = MetricRAG(agent_config, sub_agent_name)
        self.compressor = DataCompressor(model_name=agent_config.active_model().model)
        self._query_metrics_result_cache: OrderedDict[str, dict] = OrderedDict()
        self._query_metrics_result_cache_counter = 0

        # Lazy load adapter and attribution tool
        self._adapter: Optional[BaseSemanticAdapter] = None
        self._attribution_tool: Optional[DimensionAttributionUtil] = None
        self._adapter_load_error: Optional[str] = None
        self._adapter_context_key: Optional[Tuple[str, ...]] = None

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

    def metric_datasets(self) -> Optional[Dict[str, List[str]]]:
        """Metric name -> datasets it reads, for the tool-transformer context.

        ``None`` when no adapter is configured; ``{}`` when the catalog is empty.
        Consumers treat a name missing from the map as "no such metric", so every
        page is read. Result is fresh on each call, making a metric added after a
        model reload visible.
        """
        adapter = self.adapter
        if adapter is None:
            return None

        lightweight = getattr(adapter, "metric_datasets", None)
        if callable(lightweight):
            reported = _run_async(lightweight())
            if not isinstance(reported, Mapping):
                logger.warning(
                    "Adapter metric_datasets returned %s; reporting the mapping as unavailable.",
                    type(reported).__name__,
                )
                return None
            mapping = {}
            for name, datasets in reported.items():
                metric_name = str(name or "").strip()
                if metric_name:
                    mapping[metric_name] = _normalize_dataset_names(datasets)
            return mapping

        page_size, max_pages = self._metric_catalog_paging()
        mapping: Dict[str, List[str]] = {}
        offset = 0
        for _ in range(max_pages):
            page = list(_run_async(adapter.list_metrics(limit=page_size, offset=offset)))
            if not page:
                return mapping
            for metric in page:
                name = str(getattr(metric, "name", "") or "")
                if not name:
                    continue
                metadata = _normalize_metric_metadata(getattr(metric, "metadata", None))
                mapping[name] = _normalize_dataset_names(metadata.get("datasets"))
            offset += len(page)

        # A catalog that fills the bound exactly is complete; one more request tells them apart.
        if not list(_run_async(adapter.list_metrics(limit=page_size, offset=offset))):
            return mapping

        logger.warning(
            "Metric catalog still returning rows after %d pages; reporting the mapping as unavailable.",
            max_pages,
        )
        return None

    def _metric_catalog_paging(self) -> Tuple[int, int]:
        """Page size and page cap for `metric_datasets`, from the adapter's config."""
        return metric_catalog_paging(self.agent_config, self.adapter_type)

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
        from datus.agent.node.semantic_authoring import is_osi_semantic_adapter

        if not is_osi_semantic_adapter(self.adapter_type) or not semantic_model_name:
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

    def _selected_semantic_model_path(self) -> str:
        if not callable(self._semantic_model_path_provider):
            return ""
        try:
            return str(self._semantic_model_path_provider() or "").strip()
        except Exception as e:
            logger.debug("Failed to resolve the selected semantic model path: %s", e)
            return ""

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
            semantic_model_path = self._selected_semantic_model_path()
            context_key = (
                resolved_adapter,
                datasource or "",
                runtime_db_context.get("catalog", ""),
                runtime_db_context.get("database", ""),
                runtime_db_context.get("schema", ""),
                semantic_model_path,
            )
            if self._adapter is not None:
                if self._adapter_context_key is None or self._adapter_context_key == context_key:
                    return self._adapter
                self._adapter = None
                self._attribution_tool = None
                self._adapter_context_key = None

            metadata = semantic_adapter_registry.get_metadata(resolved_adapter)
            config_class = metadata.config_class if metadata and metadata.config_class else None
            config_fields = getattr(config_class, "model_fields", {})
            artifact_overrides = (
                {"semantic_model_path": semantic_model_path}
                if semantic_model_path and "semantic_model_path" in config_fields
                else {}
            )
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

                if config_class:
                    adapter_config = config_class(
                        datasource=datasource,
                        db_config=db_config,
                        semantic_models_path=semantic_models_path,
                        **artifact_overrides,
                    )
                else:
                    from datus.tools.semantic_tools.config import SemanticAdapterConfig

                    adapter_config = SemanticAdapterConfig(datasource=datasource)
            elif isinstance(adapter_config, dict):
                adapter_config = {**adapter_config, **artifact_overrides}
                if config_class:
                    adapter_config = config_class(**adapter_config)
                else:
                    from datus.tools.semantic_tools.config import SemanticAdapterConfig

                    adapter_config = SemanticAdapterConfig(**adapter_config)
            elif artifact_overrides:
                model_copy = getattr(adapter_config, "model_copy", None)
                if callable(model_copy):
                    adapter_config = model_copy(update=artifact_overrides)
                else:
                    adapter_config = copy(adapter_config)
                    for key, value in artifact_overrides.items():
                        setattr(adapter_config, key, value)

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
            # Parameterized Dosi metrics need arbitrary declaration-driven
            # binding names, so this one schema cannot be strict/closed.
            trans_to_function_tool(self.query_metrics, strict_mode=False),
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
              - extra.time_granularities: adapter-advertised grains ordered
                finest to coarsest; the first item is the default. These are
                discovery hints rather than an exhaustive allowlist; the
                adapter validates explicitly requested grains.
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
        params: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> FuncToolResult:
        """
        Query metrics data (requires adapter).

        Return complete metric results by default. Do not pass limit just to
        preview data, reduce output size, or be conservative; the visible tool
        output is compressed while the full result is cached and used for the
        final output. Use limit only when the user explicitly asks for Top N,
        first N, maximum N rows, a preview, or another row-count restriction.
        When using limit for Top N/Bottom N, also pass order_by so the
        truncation has stable business meaning.

        Args:
            metrics: List of metric names to query
            dimensions: Optional list of dimensions to group by (from get_dimensions).
                        With Dosi, use reserved `metric_time` for the selected metric's
                        primary time axis and pass its grain via `time_granularity`.
            path: Optional subject tree path (from list_subject_tree)
            time_start: Optional inclusive start of an OSI half-open range (ISO format like '2024-01-01'
                        or relative like '-7d')
            time_end: Optional exclusive end of an OSI half-open range (for example, use '2024-02-01'
                      to include all of January, or a relative value like 'now')
            time_granularity: Optional time granularity for aggregation ('day', 'week', 'month', 'quarter', 'year')
            where: Optional SQL WHERE clause (without WHERE keyword)
            limit: Optional maximum number of rows
            order_by: Optional list of result columns to sort by. Use column name for ascending,
                      prefix with '-' for descending. A Dosi input dimension `metric_time` may
                      produce a result/order key such as `metric_time__day`. Examples:
                      ['metric_time__day'] for ascending, ['-message_count'] for descending.
                      Do NOT use 'asc'/'desc' keywords.
            params: Optional parameter bindings for Dosi parameterized metrics.
                Values may be scalars or lists and are passed through unchanged.
            dry_run: If True, compile and return the query plan. Live OSI
                backends also validate the compiled SQL with a warehouse dry-run.

        Returns:
            FuncToolResult with query results or explain plan
        """
        metrics = _normalize_name_list(metrics)
        dimensions = _normalize_name_list(dimensions)
        path = _normalize_name_list(path)
        order_by = _normalize_name_list(order_by)
        params = normalize_null(params)

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
        if params is not None and not isinstance(params, dict):
            return FuncToolResult(
                success=0,
                error="query_metrics params must be an object of parameter bindings",
            )
        adapter_name = str(self.adapter_type or getattr(adapter, "service_type", "") or "").strip().lower()
        if params and adapter_name != "dosi":
            return FuncToolResult(
                success=0,
                error="query_metrics params are supported only by the Dosi semantic adapter",
            )

        # Sanitize optional parameters: LLMs may pass string null placeholders
        # instead of omitting them. In particular, an empty-string limit must not
        # reach adapters that convert a present limit with ``int(limit)``.
        path = _normalize_optional_path(path)
        time_start = normalize_null(time_start)
        time_end = normalize_null(time_end)
        time_granularity = normalize_null(time_granularity)
        where = normalize_null(where)
        limit = normalize_null(limit)
        logger.info(
            f"query_metrics called: metrics={metrics}, dimensions={dimensions}, path={path}, "
            f"time=[{time_start},{time_end}], granularity={time_granularity}, where={where}, "
            f"limit={limit}, dry_run={dry_run}"
        )

        try:
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
            if params:
                try:
                    query_parameters = inspect.signature(adapter.query_metrics).parameters
                except (TypeError, ValueError):
                    query_parameters = {}
                if not _signature_accepts_parameter(query_parameters, "params"):
                    return FuncToolResult(
                        success=0,
                        error="The current Dosi adapter does not support parameterized metrics; upgrade it first",
                    )
                adapter_query_kwargs["params"] = params
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
            validation_metric_names: Optional[List[str]] = None
            if callable(self._semantic_metric_names_provider):
                validation_metric_names = _normalize_name_list(self._semantic_metric_names_provider())
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
                if validation_metric_names is not None and _signature_accepts_parameter(params, "metric_names"):
                    validation_kwargs["metric_names"] = validation_metric_names
            except (TypeError, ValueError):
                if checks_list is not None or baseline_artifact is not None:
                    return FuncToolResult(
                        success=0,
                        error="validate_semantic validation options are not supported by the current semantic adapter",
                        result=None,
                    )
                validation_kwargs = {}

            validation_result = _run_async(validate_semantic(**validation_kwargs))
            validation_metadata = getattr(validation_result, "metadata", None)

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
            if effective_valid and isinstance(validation_metadata, dict):
                contract_digest = str(validation_metadata.get("contract_digest") or "").strip()
                metric_digests = validation_metadata.get("compiled_metric_digests")
                if contract_digest:
                    result_payload["contract_digest"] = contract_digest
                if isinstance(metric_digests, dict):
                    result_payload["compiled_metric_digests"] = dict(metric_digests)
                    result_payload["compiled_metric_count"] = len(metric_digests)
            if effective_valid and semantic_model_name:
                result_payload.update(self._semantic_model_artifact_evidence(semantic_model_name))

            tool_result = FuncToolResult(
                success=1 if effective_valid else 0,
                result=result_payload,
                error=None if effective_valid else _format_validation_error(compact_issues),
            )
            if self.generation_evidence:
                self.generation_evidence.record_validation_result(tool_result)
                if effective_valid:
                    self.generation_evidence.record_compiled_validation(
                        validation_metadata,
                        metric_names=validation_metric_names,
                    )
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
        where: Optional[str] = None,
        path: Optional[List[str]] = None,
        max_dimension_values: int = 500,
    ) -> FuncToolResult:
        """
        Descriptive dimension analysis for metric changes.

        Ranks candidate dimensions by change concentration and calculates delta
        contributions for selected dimensions. Results describe where a metric change
        is concentrated; they do not establish causation. Failed and truncated dimensions
        are excluded from rankings and contribution output.

        Args:
            metric_name: Metric to analyze(from list_metrics/search_metrics)
            candidate_dimensions: List of dimensions to evaluate (from get_dimensions)
            baseline_start: Inclusive baseline start date in an OSI half-open range (e.g., "2026-01-01")
            baseline_end: Exclusive baseline end date (e.g., "2026-01-08" for Jan 1-7)
            current_start: Inclusive current start date in an OSI half-open range (e.g., "2026-01-08")
            current_end: Exclusive current end date (e.g., "2026-01-15" for Jan 8-14)
            anomaly_context: Optional anomaly detection context (AnomalyContext with rule and observed_change_pct)
            max_selected_dimensions: Maximum dimensions to select (default 3)
            top_n_values: Number of top dimension values to return (default 10)
            where: Optional SQL boolean expression applied to every attribution query
            path: Optional subject tree path for metric scoping
            max_dimension_values: Maximum grouped values per dimension (hard-capped at 1000)

        Returns:
            FuncToolResult with:
            - dimension_ranking: All dimensions ranked by importance score
            - selected_dimensions: Top dimensions selected for analysis
            - top_dimension_values: Delta contributions of dimension values
            - dimension_analysis_status: complete, partial, unavailable, or not_requested
              when individual dimension queries fail, successful dimensions remain available
              while failed and truncated dimensions are excluded from rankings and contribution output
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
                    where=where,
                    path=path,
                    max_dimension_values=max_dimension_values,
                )
            )

            return FuncToolResult(
                success=1,
                result=result.model_dump(),
            )

        except AttributionValidationException as e:
            logger.warning("Attribution result validation failed: %s", e.payload.message)
            return FuncToolResult(
                success=0,
                error=e.payload.message,
                result=e.payload.model_dump(),
            )
        except Exception as e:
            logger.error(f"Error in attribution analysis: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to analyze attribution: {str(e)}",
            )
