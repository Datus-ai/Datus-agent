# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""AskMetricsAgenticNode for fast metric-based question answering."""

from __future__ import annotations

import csv
import io
import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from datus.agent.workflow import Workflow

from agents import Tool

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionRole, ActionStatus
from datus.schemas.ask_metrics_agentic_node_models import AskMetricsNodeInput, AskMetricsNodeResult
from datus.tools.func_tool import (
    ContextSearchTools,
    DateParsingTools,
    DBFuncTool,
    FilesystemFuncTool,
    PlatformDocSearchTool,
    trans_to_function_tool,
)
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.reference_template_tools import ReferenceTemplateTools
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class AskMetricsAgenticNode(AgenticNode):
    """Fast metric QA node backed by existing semantic metrics."""

    NODE_NAME = "ask_metrics"
    result_class = AskMetricsNodeResult
    SUBJECT_TREE_PROMPT_LIMIT = 100
    DEFAULT_TOOLS = (
        "context_search_tools.search_metrics",
        "context_search_tools.get_metrics",
        "semantic_tools.list_metrics",
        "semantic_tools.get_dimensions",
        "semantic_tools.query_metrics",
        "semantic_tools.attribution_analyze",
        "context_search_tools.list_subject_tree",
    )

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[AskMetricsNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[List[Tool]] = None,
        node_name: Optional[str] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        scope: Optional[str] = None,
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        self.execution_mode = execution_mode
        self.configured_node_name = node_name
        self.max_turns = 50
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[node_name]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", self.max_turns)

        self.semantic_tools: Optional[SemanticTools] = None
        self.context_search_tools: Optional[ContextSearchTools] = None
        self.db_func_tool: Optional[DBFuncTool] = None
        self.reference_template_tools: Optional[ReferenceTemplateTools] = None
        self.date_parsing_tools: Optional[DateParsingTools] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self._platform_doc_tool: Optional[PlatformDocSearchTool] = None
        self.subject_tree: Dict[str, Any] = {}
        self.subject_tree_metric_entries: List[Dict[str, Any]] = []
        self.subject_tree_mode: str = "none"
        self.subject_tree_prompt: str = ""
        self._metric_catalog_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._selected_final_metric_result_id: Optional[str] = None
        self.startup_error: Optional[str] = None

        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools or [],
            mcp_servers={},
            scope=scope,
            is_subagent=is_subagent,
            session_id=session_id,
        )

        # AskMetrics defaults to metric QA tools; explicitly configured custom
        # agents can still opt into additional function-tool categories.
        self.bash_tool = None
        self.skill_func_tool = None
        self.ask_user_tool = None
        self.sub_agent_task_tool = None
        self.require_final_result_selection = self._resolve_bool_config("require_final_result_selection", False)
        self.subject_tree_prompt_limit = self._resolve_subject_tree_prompt_limit()
        self.setup_tools()
        self._populate_tool_registry()

        logger.debug("AskMetricsAgenticNode tools: %s", [tool.name for tool in self.tools])

    def get_node_name(self) -> str:
        return self.configured_node_name or self.NODE_NAME

    def _resolve_adapter_type(self) -> Optional[str]:
        adapter_type = self.node_config.get("adapter_type") or self.node_config.get("semantic_adapter") or "metricflow"
        resolver = getattr(self.agent_config, "resolve_semantic_adapter", None)
        if callable(resolver):
            return resolver(adapter_type)
        return adapter_type

    def _resolve_subject_tree_prompt_limit(self) -> int:
        raw_limit = self.node_config.get("subject_tree_prompt_limit", self.SUBJECT_TREE_PROMPT_LIMIT)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid ask_metrics subject_tree_prompt_limit=%r; using default %s",
                raw_limit,
                self.SUBJECT_TREE_PROMPT_LIMIT,
            )
            return self.SUBJECT_TREE_PROMPT_LIMIT

        if isinstance(raw_limit, bool) or limit <= 0:
            logger.warning(
                "Invalid ask_metrics subject_tree_prompt_limit=%r; using default %s",
                raw_limit,
                self.SUBJECT_TREE_PROMPT_LIMIT,
            )
            return self.SUBJECT_TREE_PROMPT_LIMIT
        return limit

    def _resolve_bool_config(self, key: str, default: bool = False) -> bool:
        raw_value = self.node_config.get(key, default)
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off", ""}:
                return False
        if raw_value in (0, 1):
            return bool(raw_value)
        logger.warning("Invalid ask_metrics %s=%r; using default %s", key, raw_value, default)
        return default

    def setup_tools(self) -> None:
        if not self.agent_config:
            return

        self.tools = []
        sub_agent_name = self.get_node_name()
        tool_patterns = self._configured_tool_patterns()

        try:
            self.context_search_tools = ContextSearchTools(self.agent_config, sub_agent_name=sub_agent_name)
            self._prepare_subject_tree_context()
        except Exception as exc:  # noqa: BLE001
            message = self._record_context_search_degraded(exc)
            logger.warning("AskMetrics context search unavailable: %s", message)
            self.context_search_tools = None
            self._set_subject_tree_prompt({}, [])

        try:
            self.semantic_tools = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name=sub_agent_name,
                adapter_type=self._resolve_adapter_type(),
                runtime_db_context_provider=self._semantic_runtime_db_context,
            )
            if not self.semantic_tools._configured_adapter_type():
                self.startup_error = self.semantic_tools._adapter_unavailable_message()
                logger.warning("AskMetrics semantic adapter unavailable: %s", self.startup_error)
                return
        except Exception as exc:  # noqa: BLE001
            self.startup_error = f"Semantic adapter unavailable: {exc}"
            logger.warning("AskMetrics semantic adapter setup failed: %s", exc)
            return

        for pattern in tool_patterns:
            self._setup_tool_pattern(pattern)
        if self.require_final_result_selection:
            self._append_tool(trans_to_function_tool(self.select_final_metric_result))

    def _configured_tool_patterns(self) -> List[str]:
        config_value = self.node_config.get("tools")
        if not config_value:
            return list(self.DEFAULT_TOOLS)
        if isinstance(config_value, str):
            items = config_value.split(",")
        elif isinstance(config_value, (list, tuple)):
            items = config_value
        else:
            logger.warning("Invalid ask_metrics tools config %r; using default tools", config_value)
            return list(self.DEFAULT_TOOLS)
        patterns = [str(item).strip() for item in items if str(item).strip()]
        return patterns or list(self.DEFAULT_TOOLS)

    def _setup_tool_pattern(self, pattern: str) -> None:
        try:
            if "." not in pattern:
                category, method_name = pattern, "*"
            else:
                category, method_name = pattern.split(".", 1)

            if method_name in {"", "*"}:
                self._setup_tool_category(category)
            else:
                self._setup_specific_tool_method(category, method_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to setup ask_metrics tool pattern %r: %s", pattern, exc)

    def _setup_tool_category(self, category: str) -> None:
        if category == "context_search_tools":
            self._add_available_context_tools()
            return

        tool_instance = self._ensure_tool_instance(category)
        if not tool_instance:
            logger.warning("Ignoring unsupported ask_metrics tool category: %s", category)
            return
        self._add_available_tools(tool_instance)

    def _setup_specific_tool_method(self, category: str, method_name: str) -> None:
        if category == "context_search_tools" and method_name == "list_subject_tree":
            self._add_subject_tree_tool()
            return

        tool_instance = self._ensure_tool_instance(category)
        if not tool_instance:
            logger.warning("Ignoring unsupported ask_metrics tool category: %s", category)
            return
        if not hasattr(tool_instance, method_name):
            logger.warning("Ignoring unsupported ask_metrics tool: %s.%s", category, method_name)
            return

        method = (
            self.query_metrics
            if category == "semantic_tools" and method_name == "query_metrics"
            else getattr(tool_instance, method_name)
        )
        if category == "db_tools" and callable(getattr(tool_instance, "to_function_tool", None)):
            tool = tool_instance.to_function_tool(method)
        else:
            tool = trans_to_function_tool(method)
        self._append_tool(tool)

    def _ensure_tool_instance(self, category: str) -> Optional[Any]:
        sub_agent_name = self.get_node_name()
        if category == "context_search_tools":
            return self.context_search_tools
        if category == "semantic_tools":
            return self.semantic_tools
        if category == "db_tools":
            if not self.db_func_tool:
                self.db_func_tool = DBFuncTool(agent_config=self.agent_config, sub_agent_name=sub_agent_name)
            return self.db_func_tool
        if category == "reference_template_tools":
            if not self.reference_template_tools:
                db_tool = self.db_func_tool
                if not db_tool:
                    try:
                        db_tool = DBFuncTool(agent_config=self.agent_config, sub_agent_name=sub_agent_name)
                        self.db_func_tool = db_tool
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Reference template tools will run without db_tools: %s", exc)
                self.reference_template_tools = ReferenceTemplateTools(
                    self.agent_config,
                    sub_agent_name=sub_agent_name,
                    db_func_tool=db_tool,
                )
            return self.reference_template_tools
        if category == "date_parsing_tools":
            if not self.date_parsing_tools:
                self.date_parsing_tools = DateParsingTools(self.agent_config, self.model)
            return self.date_parsing_tools
        if category == "filesystem_tools":
            if not self.filesystem_func_tool:
                self.filesystem_func_tool = self._make_filesystem_tool()
            return self.filesystem_func_tool
        if category == "platform_doc_tools":
            if not self._platform_doc_tool:
                self._platform_doc_tool = PlatformDocSearchTool(self.agent_config)
            return self._platform_doc_tool
        return None

    def _add_available_tools(self, tool_instance: Any) -> None:
        available_tools = getattr(tool_instance, "available_tools", None)
        if not callable(available_tools):
            return
        for tool in available_tools():
            if tool_instance is self.semantic_tools and getattr(tool, "name", "") == "query_metrics":
                tool = trans_to_function_tool(self.query_metrics)
            self._append_tool(tool)

    def _add_available_context_tools(self) -> None:
        if not self.context_search_tools:
            return
        available_tools = getattr(self.context_search_tools, "available_tools", None)
        if not callable(available_tools):
            return
        for tool in available_tools():
            if getattr(tool, "name", "") == "list_subject_tree":
                self._add_subject_tree_tool()
            else:
                self._append_tool(tool)

    def _add_subject_tree_tool(self) -> None:
        if self.subject_tree_mode == "partial":
            self._append_tool(trans_to_function_tool(self.list_subject_tree))

    def _append_tool(self, tool: Optional[Tool]) -> None:
        if not tool:
            return
        tool_name = getattr(tool, "name", "")
        if tool_name and any(getattr(existing, "name", "") == tool_name for existing in self.tools):
            return
        self.tools.append(tool)

    def _prepare_subject_tree_context(self) -> None:
        if not self.context_search_tools:
            self._set_subject_tree_prompt({}, [])
            return

        result = self.context_search_tools.list_subject_tree()
        if not isinstance(result, FuncToolResult) or result.success == 0 or not isinstance(result.result, dict):
            logger.warning("AskMetrics subject tree unavailable: %s", getattr(result, "error", None))
            self._set_subject_tree_prompt({}, [])
            return

        metric_entries = self._extract_subject_tree_metric_entries(result.result)
        self._set_subject_tree_prompt(result.result, metric_entries)

    def _set_subject_tree_prompt(self, tree: Dict[str, Any], metric_entries: List[Dict[str, Any]]) -> None:
        self.subject_tree = tree
        self.subject_tree_metric_entries = metric_entries
        count = len(metric_entries)
        if count == 0:
            self.subject_tree_mode = "none"
            self.subject_tree_prompt = ""
        elif count <= self.subject_tree_prompt_limit:
            self.subject_tree_mode = "full"
            self.subject_tree_prompt = json.dumps(metric_entries, ensure_ascii=False, indent=2, default=str)
        else:
            self.subject_tree_mode = "partial"
            excerpt = {
                "shown_entries": metric_entries[: self.subject_tree_prompt_limit],
                "total_entries": count,
            }
            self.subject_tree_prompt = json.dumps(excerpt, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def _extract_subject_tree_metric_entries(cls, tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []

        def _walk(node: Any, path: List[str]) -> None:
            if not isinstance(node, dict):
                return

            metrics = node.get("metrics")
            if isinstance(metrics, list) and metrics:
                entries.append({"path": path, "metrics": metrics})

            for key, value in node.items():
                if key in {"metrics", "reference_sql", "knowledge", "reference_template"}:
                    continue
                if isinstance(value, dict):
                    _walk(value, [*path, str(key)])

        _walk(tree, [])
        return entries

    def list_subject_tree(self) -> FuncToolResult:
        """
        List metric subject entries available to ask_metrics.

        Returns only subject paths and metric names. Non-metric subject-tree
        content such as reference SQL, external knowledge, and templates is
        intentionally omitted from this subagent surface.
        """
        return FuncToolResult(
            result={
                "entries": self.subject_tree_metric_entries,
                "total_entries": len(self.subject_tree_metric_entries),
            }
        )

    def query_metrics(
        self,
        metrics: Optional[List[str]] = None,
        dimensions: Optional[List[str]] = None,
        path: Optional[List[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_granularity: Optional[str] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[List[str]] = None,
        join_policy: Optional[
            Literal["auto", "match_only", "fact_preserving", "dimension_preserving", "unmatched_only"]
        ] = None,
        zero_fill: bool = False,
        dry_run: bool = False,
    ) -> FuncToolResult:
        """
        Query metric values.

        Return complete metric results by default. Do not pass limit just to
        preview data, reduce output size, or be conservative; the system
        compresses visible tool output and caches the full result. Use limit
        only when the user explicitly asks for Top N, first N, maximum N rows,
        a preview, or another row-count restriction. When using limit for Top
        N/Bottom N, also pass order_by so the truncation has stable business
        meaning.

        For period-over-period and window metrics, AskMetrics expands the
        request to include related executable metrics when they are already
        present in the catalog.

        For joined dimensions, use semantic join policies instead of SQL join
        types: match_only for normal matched dimension grouping,
        fact_preserving for unmatched fact analysis, dimension_preserving with
        zero_fill for all dimension values including those with no facts, and
        unmatched_only for only unmapped fact rows.
        """
        if not self.semantic_tools:
            return FuncToolResult(success=0, error="semantic tools unavailable")

        expanded_metrics = self._expand_metric_dependencies(metrics)
        if not expanded_metrics:
            return FuncToolResult(
                success=0,
                error=(
                    "query_metrics requires at least one metric name. "
                    "Call list_metrics first and pass one or more metric names exactly as returned."
                ),
            )
        normalized_dimensions = None if dimensions is None else self._normalize_string_list(dimensions)
        normalized_order_by = None if order_by is None else self._normalize_string_list(order_by)
        normalized_dimensions, time_granularity, normalized_order_by = self._apply_window_time_grouping(
            expanded_metrics,
            normalized_dimensions,
            time_granularity,
            normalized_order_by,
        )
        query_kwargs = {
            "metrics": expanded_metrics,
            "dimensions": normalized_dimensions,
            "path": path,
            "time_start": time_start,
            "time_end": time_end,
            "time_granularity": time_granularity,
            "where": where,
            "limit": limit,
            "order_by": normalized_order_by,
            "dry_run": dry_run,
        }
        if join_policy:
            query_kwargs["join_policy"] = join_policy
        if zero_fill:
            query_kwargs["zero_fill"] = zero_fill
        result = self.semantic_tools.query_metrics(**query_kwargs)
        return self._apply_query_result_column_aliases(result)

    def select_final_metric_result(self, result_id: str) -> FuncToolResult:
        """
        Select the query_metrics result that should be stored as the final structured result.

        This tool is only exposed when require_final_result_selection is enabled.
        The result_id must be returned by a successful query_metrics call.
        """
        result_id = str(result_id or "").strip()
        if not result_id:
            return FuncToolResult(success=0, error="result_id is required")
        if (
            self.semantic_tools
            and hasattr(self.semantic_tools, "get_cached_query_metrics_result")
            and self.semantic_tools.get_cached_query_metrics_result(result_id) is None
        ):
            return FuncToolResult(success=0, error=f"Unknown query_metrics result_id: {result_id}")

        self._selected_final_metric_result_id = result_id
        return FuncToolResult(result={"result_id": result_id})

    def _expand_period_over_period_metrics(self, metrics: Optional[List[str]]) -> List[str]:
        requested_metrics = self._normalize_string_list(metrics)
        if not requested_metrics:
            return []

        catalog = self._metric_catalog()
        if not catalog:
            return requested_metrics

        expanded: List[str] = []
        for metric_name in requested_metrics:
            metric = catalog.get(metric_name)
            if not metric:
                self._append_unique(expanded, metric_name)
                continue

            for bundled_metric in self._period_over_period_metric_bundle(metric_name, metric, catalog):
                self._append_unique(expanded, bundled_metric)

        return expanded

    def _expand_metric_dependencies(self, metrics: Optional[List[str]]) -> List[str]:
        expanded_metrics = self._expand_period_over_period_metrics(metrics)
        return self._expand_window_metrics(expanded_metrics)

    def _expand_window_metrics(self, metrics: Optional[List[str]]) -> List[str]:
        requested_metrics = self._normalize_string_list(metrics)
        if not requested_metrics:
            return []

        catalog = self._metric_catalog()
        if not catalog:
            return requested_metrics

        expanded: List[str] = []
        for metric_name in requested_metrics:
            metric = catalog.get(metric_name)
            if not metric:
                self._append_unique(expanded, metric_name)
                continue

            for bundled_metric in self._window_metric_bundle(metric_name, metric, catalog):
                self._append_unique(expanded, bundled_metric)

        return expanded

    def _apply_window_time_grouping(
        self,
        metrics: List[str],
        dimensions: Optional[List[str]],
        time_granularity: Optional[str],
        order_by: Optional[List[str]],
    ) -> tuple[Optional[List[str]], Optional[str], Optional[List[str]]]:
        catalog = self._metric_catalog()
        if not catalog:
            return dimensions, time_granularity, order_by

        time_dimension = ""
        inferred_grain = self._normalize_time_grain(time_granularity)
        for metric_name in metrics:
            metric = catalog.get(metric_name)
            if not metric:
                continue
            metadata = self._metric_metadata(metric)
            if not self._is_window_metric(metadata):
                continue
            inferred_grain = inferred_grain or self._infer_window_time_grain(metadata)
            if inferred_grain:
                time_dimension = f"metric_time__{inferred_grain}"
                break

        if not time_dimension:
            return dimensions, time_granularity, order_by

        updated_dimensions = list(dimensions or [])
        if not any(str(dimension).startswith("metric_time__") for dimension in updated_dimensions):
            self._append_unique(updated_dimensions, time_dimension)

        updated_order_by = list(order_by or [])
        if not updated_order_by:
            updated_order_by = [time_dimension]

        return updated_dimensions, time_granularity or inferred_grain, updated_order_by

    @classmethod
    def _window_metric_bundle(
        cls,
        metric_name: str,
        metric: Dict[str, Any],
        catalog: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        metadata = cls._metric_metadata(metric)
        if not cls._is_window_metric(metadata):
            return [metric_name]

        if cls._metadata_text(metadata, "window_aggregation").lower() == "row_count":
            return [metric_name]

        bundle: List[str] = []
        base_metric = cls._find_window_base_metric(metric_name, metric, catalog)
        if base_metric:
            cls._append_unique(bundle, base_metric)

        if cls._window_metric_needs_window_count(metadata):
            for count_metric in cls._matching_window_count_metrics(metric_name, metadata, catalog):
                cls._append_unique(bundle, count_metric)

        cls._append_unique(bundle, metric_name)
        return bundle

    def _metric_catalog(self) -> Dict[str, Dict[str, Any]]:
        if self._metric_catalog_cache is not None:
            return self._metric_catalog_cache
        if not self.semantic_tools:
            return {}

        catalog: Dict[str, Dict[str, Any]] = {}
        offset = 0
        limit = 500
        for _ in range(10):
            try:
                result = self.semantic_tools.list_metrics(limit=limit, offset=offset)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Unable to load ask_metrics metric catalog: %s", exc)
                break

            if not isinstance(result, FuncToolResult) or result.success == 0:
                logger.debug("Unable to load ask_metrics metric catalog: %s", getattr(result, "error", None))
                break

            payload = result.result if isinstance(result.result, dict) else {}
            items = payload.get("items", []) if isinstance(payload, dict) else []
            for item in items:
                if isinstance(item, dict) and item.get("name"):
                    catalog[str(item["name"])] = item

            if not isinstance(payload, dict) or not payload.get("has_more"):
                break
            extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
            next_offset = extra.get("next_offset")
            if not isinstance(next_offset, int) or next_offset <= offset:
                break
            offset = next_offset

        self._metric_catalog_cache = catalog
        return catalog

    @classmethod
    def _find_window_base_metric(
        cls,
        metric_name: str,
        metric: Dict[str, Any],
        catalog: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        metadata = cls._metric_metadata(metric)
        explicit_name = cls._first_metadata_text(
            metadata,
            ("base_metric", "base_metric_name", "source_metric", "source_metric_name"),
        )
        if explicit_name and explicit_name in catalog and explicit_name != metric_name:
            return explicit_name

        metric_signature = cls._metric_measure_signature(metric)
        if not metric_signature:
            return None

        dataset = cls._metadata_text(metadata, "dataset")
        for candidate_name, candidate in catalog.items():
            if candidate_name == metric_name:
                continue
            candidate_metadata = cls._metric_metadata(candidate)
            if cls._is_window_metric(candidate_metadata):
                continue
            if not cls._is_base_metric_candidate(candidate_metadata):
                continue
            candidate_dataset = cls._metadata_text(candidate_metadata, "dataset")
            if dataset and candidate_dataset and dataset != candidate_dataset:
                continue
            if metric_signature.intersection(cls._metric_measure_signature(candidate)):
                return candidate_name

        return None

    @classmethod
    def _matching_window_count_metrics(
        cls,
        metric_name: str,
        metadata: Dict[str, Any],
        catalog: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        matches: List[str] = []
        dataset = cls._metadata_text(metadata, "dataset")
        time_dimension = cls._metadata_text(metadata, "time_dimension")
        window = cls._metadata_text(metadata, "window")
        grain_to_date = cls._metadata_text(metadata, "grain_to_date")

        for candidate_name, candidate in catalog.items():
            if candidate_name == metric_name:
                continue
            candidate_metadata = cls._metric_metadata(candidate)
            if cls._metadata_text(candidate_metadata, "window_aggregation").lower() != "row_count":
                continue
            if dataset and cls._metadata_text(candidate_metadata, "dataset") != dataset:
                continue
            if time_dimension and cls._metadata_text(candidate_metadata, "time_dimension") != time_dimension:
                continue
            if window and cls._metadata_text(candidate_metadata, "window") != window:
                continue
            if grain_to_date and cls._metadata_text(candidate_metadata, "grain_to_date") != grain_to_date:
                continue
            cls._append_unique(matches, candidate_name)

        return matches

    @classmethod
    def _window_metric_needs_window_count(cls, metadata: Dict[str, Any]) -> bool:
        aggregation = cls._metadata_text(metadata, "window_aggregation").lower()
        return aggregation in {"avg", "average", "mean"}

    @classmethod
    def _is_window_metric(cls, metadata: Dict[str, Any]) -> bool:
        if not metadata:
            return False
        if cls._metadata_text(metadata, "window") or cls._metadata_text(metadata, "grain_to_date"):
            return True
        return bool(cls._metadata_text(metadata, "window_aggregation"))

    @classmethod
    def _is_base_metric_candidate(cls, metadata: Dict[str, Any]) -> bool:
        metric_kind = cls._metadata_text(metadata, "metric_kind").lower()
        if not metric_kind:
            return True
        return metric_kind in {"aggregate", "measure_proxy", "simple"}

    @classmethod
    def _infer_window_time_grain(cls, metadata: Dict[str, Any]) -> str:
        for key in ("time_granularity", "time_grain", "grain_to_date"):
            grain = cls._normalize_time_grain(cls._metadata_text(metadata, key))
            if grain:
                return grain

        time_dimension = cls._metadata_text(metadata, "time_dimension")
        if time_dimension.startswith("metric_time__"):
            grain = cls._normalize_time_grain(time_dimension.split("__", 1)[1])
            if grain:
                return grain

        window = cls._metadata_text(metadata, "window")
        if window:
            parts = re.findall(r"[a-zA-Z]+", window.lower())
            if parts:
                return cls._normalize_time_grain(parts[-1])
        return ""

    @staticmethod
    def _normalize_time_grain(value: Optional[str]) -> str:
        grain = str(value or "").strip().lower()
        if grain.endswith("s"):
            grain = grain[:-1]
        return grain if grain in {"day", "week", "month", "quarter", "year"} else ""

    @staticmethod
    def _metric_metadata(metric: Dict[str, Any]) -> Dict[str, Any]:
        metadata = metric.get("metadata") if isinstance(metric, dict) else {}
        return metadata if isinstance(metadata, dict) else {}

    @classmethod
    def _metric_measure_signature(cls, metric: Dict[str, Any]) -> set[str]:
        metadata = cls._metric_metadata(metric)
        values: List[str] = []
        for key in ("measure", "measure_expr", "expr"):
            value = metadata.get(key)
            if isinstance(value, str):
                values.append(value)

        for key in ("measures", "base_measures"):
            value = metric.get(key)
            if isinstance(value, list):
                values.extend(str(item) for item in value if str(item).strip())
            elif isinstance(value, str):
                values.append(value)

        return {cls._normalize_metric_expression(value) for value in values if cls._normalize_metric_expression(value)}

    @staticmethod
    def _normalize_metric_expression(value: str) -> str:
        return "".join(str(value or "").lower().split())

    @staticmethod
    def _metadata_text(metadata: Dict[str, Any], key: str) -> str:
        value = metadata.get(key)
        return value.strip() if isinstance(value, str) else ""

    @classmethod
    def _first_metadata_text(cls, metadata: Dict[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = cls._metadata_text(metadata, key)
            if value:
                return value
        return ""

    @classmethod
    def _period_over_period_metric_bundle(
        cls,
        metric_name: str,
        metric: Dict[str, Any],
        catalog: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        metadata = metric.get("metadata") if isinstance(metric, dict) else {}
        if not isinstance(metadata, dict):
            return [metric_name]

        inputs = metadata.get("inputs")
        if not isinstance(inputs, list):
            return [metric_name]

        base_metrics: List[str] = []
        previous_metrics: List[str] = []
        has_current_input = False
        has_offset_input = False
        for item in inputs:
            if not isinstance(item, dict):
                continue

            base_name = item.get("name")
            if not isinstance(base_name, str) or base_name not in catalog:
                continue

            if item.get("offset_window"):
                has_offset_input = True
                alias = item.get("alias")
                if isinstance(alias, str) and alias in catalog:
                    cls._append_unique(previous_metrics, alias)
                for equivalent_metric in cls._equivalent_offset_identity_metrics(base_name, item, catalog):
                    cls._append_unique(previous_metrics, equivalent_metric)
            else:
                has_current_input = True
                cls._append_unique(base_metrics, base_name)

        if not has_current_input or not has_offset_input:
            return [metric_name]

        return [*base_metrics, *previous_metrics, metric_name]

    @classmethod
    def _equivalent_offset_identity_metrics(
        cls,
        base_name: str,
        offset_input: Dict[str, Any],
        catalog: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        offset_window = str(offset_input.get("offset_window") or "").strip()
        offset_to_grain = str(offset_input.get("offset_to_grain") or "").strip()
        if not offset_window:
            return []

        equivalent_metrics: List[str] = []
        for candidate_name, candidate in catalog.items():
            metadata = candidate.get("metadata") if isinstance(candidate, dict) else {}
            inputs = metadata.get("inputs") if isinstance(metadata, dict) else None
            if not isinstance(inputs, list) or len(inputs) != 1:
                continue
            candidate_input = inputs[0]
            if not isinstance(candidate_input, dict):
                continue
            if candidate_input.get("name") != base_name:
                continue
            if str(candidate_input.get("offset_window") or "").strip() != offset_window:
                continue
            if str(candidate_input.get("offset_to_grain") or "").strip() != offset_to_grain:
                continue
            alias = candidate_input.get("alias")
            expr = metadata.get("expr")
            if isinstance(alias, str) and isinstance(expr, str) and expr.strip() == alias and candidate_name == alias:
                cls._append_unique(equivalent_metrics, candidate_name)
        return equivalent_metrics

    @staticmethod
    def _normalize_string_list(value: Optional[List[str]]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            return []
        return [str(item).strip() for item in values if str(item).strip()]

    @staticmethod
    def _is_joined_dimension(dimension: str) -> bool:
        return "__" in dimension and not str(dimension).startswith("metric_time__")

    @classmethod
    def _apply_query_result_column_aliases(cls, result: FuncToolResult) -> FuncToolResult:
        if not isinstance(result, FuncToolResult) or result.success == 0 or not isinstance(result.result, dict):
            return result

        payload = result.result
        columns = payload.get("columns")
        if not isinstance(columns, list):
            return result

        aliases = cls._query_metric_column_aliases(columns)
        if not aliases:
            return result

        payload["columns"] = cls._apply_column_aliases_to_columns(columns, aliases)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["_display_column_aliases"] = aliases
        payload["metadata"] = metadata

        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("compressed_data"), str):
            aliased_data = dict(data)
            aliased_data["compressed_data"] = cls._apply_column_aliases_to_csv(
                aliased_data["compressed_data"],
                aliases,
            )
            payload["data"] = aliased_data
        return result

    @classmethod
    def _query_metric_column_aliases(cls, columns: List[Any]) -> Dict[str, str]:
        normalized_columns = [str(column) for column in columns]
        suffix_counts: Dict[str, int] = {}
        for column in normalized_columns:
            if not cls._is_joined_dimension(column):
                continue
            suffix = column.rsplit("__", 1)[-1]
            if suffix:
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

        existing_columns = set(normalized_columns)
        aliases: Dict[str, str] = {}
        for column in normalized_columns:
            if not cls._is_joined_dimension(column):
                continue
            suffix = column.rsplit("__", 1)[-1]
            if not suffix or suffix_counts.get(suffix) != 1 or suffix in existing_columns:
                continue
            aliases[column] = suffix
        return aliases

    @staticmethod
    def _column_aliases_from_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        raw_aliases = metadata.get("_display_column_aliases")
        if not isinstance(raw_aliases, dict):
            return {}
        return {str(source): str(target) for source, target in raw_aliases.items() if source and target}

    @staticmethod
    def _apply_column_aliases_to_columns(columns: List[Any], aliases: Dict[str, str]) -> List[str]:
        if not aliases:
            return [str(column) for column in columns]
        return [aliases.get(str(column), str(column)) for column in columns]

    @classmethod
    def _apply_column_aliases_to_csv(cls, csv_text: str, aliases: Dict[str, str]) -> str:
        if not aliases or not isinstance(csv_text, str) or not csv_text:
            return csv_text

        reader = csv.reader(io.StringIO(csv_text))
        try:
            header = next(reader)
        except StopIteration:
            return csv_text

        aliased_header = cls._apply_column_aliases_to_columns(header, aliases)
        if aliased_header == header:
            return csv_text

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(aliased_header)
        writer.writerows(reader)
        return buf.getvalue()

    @staticmethod
    def _append_unique(values: List[str], value: str) -> None:
        if value not in values:
            values.append(value)

    async def _before_stream(self, ctx: StreamRunContext) -> None:
        if self.startup_error:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"ask_metrics is unavailable: {self.startup_error}"},
            )

    def _runtime_context_current_date(self) -> str:
        """Honor the per-input ``reference_date`` in the frozen runtime context."""
        from datus.utils.time_utils import get_default_current_date

        input_ref_date = getattr(self.input, "reference_date", None) if self.input else None
        return get_default_current_date(input_ref_date)

    def _get_system_prompt(self, prompt_version: Optional[str] = None) -> str:
        context: Dict[str, Any] = {
            "agent_config": self.agent_config,
            "rules": self.node_config.get("rules", []),
            "agent_description": self.node_config.get("agent_description", ""),
            "subject_tree_mode": self.subject_tree_mode,
            "subject_tree_count": len(self.subject_tree_metric_entries),
            "subject_tree_prompt": self.subject_tree_prompt,
            "subject_tree_prompt_limit": self.subject_tree_prompt_limit,
            "require_final_result_selection": self.require_final_result_selection,
        }

        if self.agent_config:
            from datus.utils.node_utils import build_datasource_prompt_context

            context.update(build_datasource_prompt_context(self.agent_config))
            context["db_name"] = context.get("datasource")

        version_value = prompt_version if prompt_version not in (None, "") else self.node_config.get("prompt_version")
        version = None if version_value in (None, "") else str(version_value)
        system_prompt_name = self.node_config.get("system_prompt") or self.get_node_name()
        template_name = f"{system_prompt_name}_system"

        from datus.prompts.prompt_manager import get_prompt_manager

        pm = get_prompt_manager(agent_config=self.agent_config)
        try:
            base_prompt = pm.render_template(template_name=template_name, version=version, **context)
        except FileNotFoundError:
            logger.warning(
                "Template %r missing, falling back to ask_metrics_system",
                system_prompt_name,
            )
            base_prompt = pm.render_template(template_name="ask_metrics_system", version=version, **context)
        return self._finalize_system_prompt(base_prompt)

    def _build_success_result(self, ctx: StreamRunContext) -> AskMetricsNodeResult:
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            response_content = (
                ctx.last_successful_output.get("content", "")
                or ctx.last_successful_output.get("text", "")
                or ctx.last_successful_output.get("response", "")
                or str(ctx.last_successful_output)
            )

        all_actions = ctx.action_history_manager.get_actions()
        tokens_used = self._extract_total_tokens(all_actions)
        tool_calls = [a for a in all_actions if a.role == ActionRole.TOOL and a.status == ActionStatus.SUCCESS]
        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        return AskMetricsNodeResult(
            success=True,
            response=response_content,
            markdown_report=response_content,
            tokens_used=int(tokens_used),
            action_history=[a.model_dump() for a in all_actions],
            execution_stats={
                "total_actions": len(all_actions),
                "tool_calls_count": len(tool_calls),
                "tools_used": sorted({a.action_type for a in tool_calls}),
                "total_tokens": int(tokens_used),
            },
        )

    @staticmethod
    def _query_action_arguments(action: Dict[str, Any]) -> Dict[str, Any]:
        raw_arguments = (action.get("input") or {}).get("arguments")
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str) and raw_arguments.strip():
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _query_result_payload(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        output = action.get("output", {})
        if not isinstance(output, dict):
            return None
        raw_output = output.get("raw_output", output)
        if not isinstance(raw_output, dict) or not raw_output.get("success"):
            return None
        result = raw_output.get("result", {})
        return result if isinstance(result, dict) else None

    @staticmethod
    def _query_result_columns(result: Dict[str, Any]) -> List[str]:
        columns = result.get("columns", [])
        if not isinstance(columns, list):
            return []
        return [str(column) for column in columns if str(column)]

    @staticmethod
    def _query_result_row_count(result: Dict[str, Any]) -> int:
        data = result.get("data")
        if isinstance(data, dict):
            raw_count = data.get("original_rows", 0)
            try:
                return int(raw_count)
            except (TypeError, ValueError):
                return 0
        if isinstance(data, list):
            return len(data)
        return 0

    @staticmethod
    def _query_result_id(result: Dict[str, Any]) -> Optional[str]:
        result_id = result.get("result_id")
        if isinstance(result_id, str) and result_id.strip():
            return result_id.strip()
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            cache_key = metadata.get("_full_result_cache_key")
            if isinstance(cache_key, str) and cache_key.strip():
                return cache_key.strip()
        return None

    @classmethod
    def _select_last_query_metrics_action(cls, actions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        selected: Optional[Dict[str, Any]] = None

        for action in actions:
            if not isinstance(action, dict):
                continue
            if action.get("action_type") != "query_metrics":
                continue
            if action.get("status") != "success":
                continue
            result = cls._query_result_payload(action)
            if not result:
                continue
            columns = cls._query_result_columns(result)
            data = result.get("data")
            if not columns or data is None:
                continue
            selected = action

        return selected

    @classmethod
    def _selected_final_result_id_from_actions(cls, actions: List[Dict[str, Any]]) -> Optional[str]:
        selected_result_id: Optional[str] = None
        for action in actions:
            if not isinstance(action, dict):
                continue
            if action.get("action_type") != "select_final_metric_result":
                continue
            if action.get("status") != "success":
                continue
            result = cls._query_result_payload(action)
            if not result:
                continue
            result_id = result.get("result_id")
            if isinstance(result_id, str) and result_id.strip():
                selected_result_id = result_id.strip()
        return selected_result_id

    @classmethod
    def _select_query_metrics_action_by_result_id(
        cls,
        actions: List[Dict[str, Any]],
        result_id: str,
    ) -> Optional[Dict[str, Any]]:
        for action in actions:
            if not isinstance(action, dict):
                continue
            if action.get("action_type") != "query_metrics" or action.get("status") != "success":
                continue
            result = cls._query_result_payload(action)
            if result and cls._query_result_id(result) == result_id:
                return action
        return None

    def update_context(self, workflow: "Workflow") -> Dict:
        """Extract the selected query_metrics result into sql_context for OutputNode."""
        actions = getattr(self.result, "action_history", None) or []
        require_selection = getattr(self, "require_final_result_selection", False)
        require_selection = require_selection if isinstance(require_selection, bool) else False

        if require_selection:
            selected_result_id = self._selected_final_result_id_from_actions(actions)
            if not selected_result_id:
                selected_result_id = getattr(self, "_selected_final_metric_result_id", None)
            if not selected_result_id:
                logger.warning("ask_metrics requires select_final_metric_result, but no final result was selected")
                return {"success": False, "message": "final query_metrics result was not selected"}
            action = self._select_query_metrics_action_by_result_id(actions, selected_result_id)
            if not action:
                logger.warning("Selected query_metrics result_id was not found: %s", selected_result_id)
                return {"success": False, "message": f"selected query_metrics result not found: {selected_result_id}"}
        else:
            action = self._select_last_query_metrics_action(actions)

        if action:
            result = self._query_result_payload(action) or {}

            columns = result.get("columns", [])
            data = result.get("data")
            if not columns or data is None:
                return super().update_context(workflow)

            metadata = result.get("metadata", {}) or {}
            column_aliases = self._column_aliases_from_metadata(metadata)
            cached_result = None
            cache_key = metadata.get("_full_result_cache_key")
            if cache_key and self.semantic_tools and hasattr(self.semantic_tools, "get_cached_query_metrics_result"):
                cached_result = self.semantic_tools.get_cached_query_metrics_result(cache_key)

            if isinstance(data, dict) and data.get("compressed_data"):
                row_count = data.get("original_rows", 0)
                if isinstance(cached_result, dict) and cached_result.get("csv"):
                    sql_return = cached_result["csv"]
                    row_count = cached_result.get("row_count", row_count)
                    sql_return = self._apply_column_aliases_to_csv(sql_return, column_aliases)
                else:
                    sql_return = data["compressed_data"]
                    sql_return = self._apply_column_aliases_to_csv(sql_return, column_aliases)
            elif isinstance(data, list):
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(self._apply_column_aliases_to_columns(columns, column_aliases))
                writer.writerows(data)
                sql_return = buf.getvalue()
                row_count = len(data)
            else:
                return super().update_context(workflow)

            sql_query = ""
            for key in ("sql", "compiled_sql", "generated_sql"):
                if metadata.get(key):
                    sql_query = metadata[key]
                    break

            from datus.schemas.node_models import SQLContext

            workflow.context.sql_contexts.append(
                SQLContext(sql_query=sql_query, sql_return=sql_return, row_count=row_count)
            )
            logger.info("Captured query_metrics result: %d columns, %d rows", len(columns), row_count)
            return {"success": True, "message": "query_metrics result captured"}

        return super().update_context(workflow)
