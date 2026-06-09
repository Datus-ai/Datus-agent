# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""AskMetricsAgenticNode for fast metric-based question answering."""

from __future__ import annotations

import json
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

    _TOOL_NAMES_BY_CATEGORY = {
        "db_tools": set(DBFuncTool.all_tools_name()),
        "context_search_tools": set(ContextSearchTools.all_tools_name()),
        "semantic_tools": set(SemanticTools.all_tools_name()),
        "reference_template_tools": set(ReferenceTemplateTools.all_tools_name()),
        "date_parsing_tools": {"parse_temporal_expressions"},
        "filesystem_tools": set(FilesystemFuncTool.all_tools_name()),
        "platform_doc_tools": set(PlatformDocSearchTool.all_tools_name()),
    }

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
        self.max_turns = 12
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
            )
            if self.semantic_tools.adapter is None:
                self.startup_error = self.semantic_tools._adapter_unavailable_message()
                logger.warning("AskMetrics semantic adapter unavailable: %s", self.startup_error)
                return
        except Exception as exc:  # noqa: BLE001
            self.startup_error = f"Semantic adapter unavailable: {exc}"
            logger.warning("AskMetrics semantic adapter setup failed: %s", exc)
            return

        for pattern in tool_patterns:
            self._setup_tool_pattern(pattern)

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

        method = getattr(tool_instance, method_name)
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

    async def _before_stream(self, ctx: StreamRunContext) -> None:
        if self.startup_error:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"ask_metrics is unavailable: {self.startup_error}"},
            )

    def _tool_category_map(self) -> Dict[str, List[Any]]:
        mapping = super()._tool_category_map()
        for category, names in self._TOOL_NAMES_BY_CATEGORY.items():
            category_tools = [tool for tool in self.tools if getattr(tool, "name", "") in names]
            if category_tools:
                mapping[category] = category_tools
        return mapping

    def _get_system_prompt(self, prompt_version: Optional[str] = None) -> str:
        context: Dict[str, Any] = {
            "agent_config": self.agent_config,
            "rules": self.node_config.get("rules", []),
            "agent_description": self.node_config.get("agent_description", ""),
            "subject_tree_mode": self.subject_tree_mode,
            "subject_tree_count": len(self.subject_tree_metric_entries),
            "subject_tree_prompt": self.subject_tree_prompt,
            "subject_tree_prompt_limit": self.subject_tree_prompt_limit,
        }

        if self.agent_config:
            from datus.utils.node_utils import build_datasource_prompt_context

            context.update(build_datasource_prompt_context(self.agent_config))
            context["db_name"] = context.get("datasource")

        from datus.utils.time_utils import get_default_current_date

        context["current_date"] = get_default_current_date(None)

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

    def update_context(self, workflow: "Workflow") -> Dict:
        """Extract last query_metrics result into sql_context for OutputNode."""
        actions = getattr(self.result, "action_history", None) or []
        for action in reversed(actions):
            if not isinstance(action, dict):
                continue
            if action.get("action_type") != "query_metrics":
                continue
            if action.get("status") != "success":
                continue

            output = action.get("output", {})
            if not isinstance(output, dict):
                continue
            raw_output = output.get("raw_output", output)
            if not isinstance(raw_output, dict) or not raw_output.get("success"):
                continue
            result = raw_output.get("result", {})
            if not isinstance(result, dict):
                continue

            columns = result.get("columns", [])
            data = result.get("data")
            if not columns or not data:
                continue

            if isinstance(data, dict) and data.get("compressed_data"):
                sql_return = data["compressed_data"]
                row_count = data.get("original_rows", 0)
            elif isinstance(data, list):
                import csv
                import io

                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(columns)
                writer.writerows(data)
                sql_return = buf.getvalue()
                row_count = len(data)
            else:
                continue

            metadata = result.get("metadata", {}) or {}
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
