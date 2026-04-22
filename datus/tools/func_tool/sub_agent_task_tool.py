# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SubAgentTaskTool for delegating specialized tasks to AgenticNode instances.

This module provides a tool that enables ChatAgenticNode to delegate tasks
(e.g., SQL generation) to specialized AgenticNode instances (e.g., GenSQLAgenticNode),
giving each subagent full node capabilities: independent session, config-driven
tools, template rendering, and action history.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from agents import FunctionTool, Tool

from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import (
    SUBAGENT_COMPLETE_ACTION_TYPE,
    ActionHistory,
    ActionHistoryManager,
    ActionRole,
    ActionStatus,
)
from datus.schemas.agent_models import SubAgentConfig
from datus.tools.func_tool.base import FuncToolResult
from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.agent.node.agentic_node import AgenticNode
    from datus.cli.execution_state import InteractionBroker
    from datus.schemas.action_bus import ActionBus

logger = get_logger(__name__)

# Mapping from subagent type string to NodeType constants
NODE_CLASS_MAP = {
    "gen_sql": NodeType.TYPE_GENSQL,
    "chat": NodeType.TYPE_CHAT,
    "gen_report": NodeType.TYPE_GEN_REPORT,
    "ext_knowledge": NodeType.TYPE_EXT_KNOWLEDGE,
    "semantic": NodeType.TYPE_SEMANTIC,
    "sql_summary": NodeType.TYPE_SQL_SUMMARY,
    "explore": NodeType.TYPE_EXPLORE,
    "gen_table": NodeType.TYPE_GEN_TABLE,
    "gen_job": NodeType.TYPE_GEN_JOB,
    "gen_skill": NodeType.TYPE_GEN_SKILL,
    "gen_dashboard": NodeType.TYPE_GEN_DASHBOARD,
    "scheduler": NodeType.TYPE_SCHEDULER,
}

# Descriptions for built-in system subagents (used in task tool description for LLM)
BUILTIN_SUBAGENT_DESCRIPTIONS = {
    "gen_sql": (
        "Generate optimized SQL queries. Returns JSON with {sql, response, tokens_used}. "
        "For complex SQL (50+ lines), returns {sql_file_path, sql_preview, response} instead - "
        "pass sql_file_path directly to read_query() to execute (no need to read_file() first). "
        "Modifications return sql_diff in unified diff format. "
        "Use for data queries, analysis, and report SQL. Prompt: provide the question directly."
    ),
    "explore": (
        "Read-only data exploration. Supports 3 exploration directions:\n"
        "  * Schema+Sample: database schema structure, table columns, types, "
        "sample data, date context\n"
        "    Prompt example: 'Explore schema for tables related to sales: "
        "list tables, describe columns, sample 10 rows'\n"
        "  * Knowledge: business metrics, reference SQL patterns, "
        "domain knowledge, semantic objects\n"
        "    Prompt example: 'Search knowledge base for sales-related metrics, "
        "reference SQL, and business rules'\n"
        "  * File: workspace SQL files, documentation, configuration files\n"
        "    Prompt example: 'Browse workspace for SQL files and documentation "
        "related to sales'\n"
        '  For comprehensive exploration, call task(type="explore") MULTIPLE TIMES '
        "in PARALLEL with direction-specific prompts.\n"
        "  Returns JSON with {response, tokens_used}."
    ),
    "gen_report": (
        "Analyze and attribute metrics using reference SQL and semantic layer. "
        "Use when the question involves metric attribution, root cause analysis, metric trend explanation, "
        "or analyzing why a metric changed. "
        "Prompt: provide the metric question, include reference SQL or metric name if available. "
        "Returns JSON with {response, report_result, tokens_used}."
    ),
    "gen_semantic_model": (
        "Generate MetricFlow semantic model YAML files from database table structures. "
        "Use when asked to create or update semantic models, define entities, relationships, or dimensions. "
        "Prompt MUST contain table name(s), e.g. 'orders' or 'orders, customers, products'. "
        "Returns JSON with {response, semantic_models (list of file paths), tokens_used}."
    ),
    "gen_metrics": (
        "Define and generate MetricFlow metric definitions. "
        "Three input modes: "
        "(1) SQL-based: provide SQL queries for metric extraction. "
        "(2) Natural language: describe the business metric or calculation rules, "
        "the agent will guide through interactive Q&A to define the metric. "
        "(3) Batch: provide multiple SQL queries for core metric extraction. "
        "For batch input, if the user provides a CSV file path, YOU (the parent agent) must read the file content first "
        "and include the full content in the prompt — the metrics agent cannot access files outside its workspace. "
        "The metrics agent will deduplicate aggregation patterns and propose only core base metrics. "
        "Returns JSON with {response, tokens_used}."
    ),
    "gen_sql_summary": (
        "Analyze and summarize SQL queries into reusable knowledge base entries for semantic search. "
        "Use when asked to summarize, document, or index SQL queries for future reference. "
        "Prompt MUST contain a complete SQL query, optionally with business context description. "
        "Returns JSON with {response, sql_summary_file, tokens_used}."
    ),
    "gen_skill": (
        "Create new skills or optimize existing skills. "
        "For new skills: capture intent, interview user, write SKILL.md, scaffold directory. "
        "For optimization: load existing skill, analyze usage sessions and tool call patterns, rewrite. "
        "Prompt: describe what skill to create, or 'optimize <skill-name>' to improve an existing skill. "
        "Returns JSON with {response, skill_name, skill_path, tokens_used}."
    ),
    "gen_ext_knowledge": (
        "Discover and extract business knowledge through blind-test → verify → extract workflow. "
        "Use when asked to generate business knowledge entries, define domain concepts, or build knowledge base. "
        "Prompt MUST contain a natural language business question AND the gold SQL (reference answer), "
        "e.g. 'What is the total revenue by region? SELECT region, SUM(revenue) FROM sales GROUP BY region'. "
        "The agent will parse both from the prompt, autonomously explore the database, write SQL, "
        "verify against the gold SQL reference, and extract knowledge. "
        "Returns JSON with {response, ext_knowledge_file, tokens_used}."
    ),
    "gen_table": (
        "Create database tables with two input modes: "
        "(1) SQL-based: provide a JOIN/SELECT SQL → CTAS to create a wide table for query acceleration. "
        "(2) Natural language: describe the table structure (columns, types, purpose) → generate CREATE TABLE DDL. "
        "Both modes: the agent analyzes the input, proposes a table schema, asks for confirmation, "
        "and executes the DDL. For semantic model generation on the new table, "
        "use gen_semantic_model separately. Returns JSON with {response, tokens_used}."
    ),
    "gen_job": (
        "Execute data pipeline jobs — BOTH intra-database ETL AND cross-database migration. "
        "For intra-database ETL: builds a target table from source tables using SQL "
        "(CREATE TABLE AS SELECT, INSERT from SELECT, etc.) within the SAME database. "
        "For cross-database migration: transfers data between different database engines "
        "(e.g., DuckDB to Greenplum, MySQL to StarRocks, Postgres to ClickHouse). "
        "Handles cross-dialect type mapping, target DDL generation, data transfer via "
        "transfer_query_result, and mandatory post-migration reconciliation "
        "(row count, null ratio, min/max, aggregates) when source and target differ. "
        "Inspects source and target schemas, generates DDL, writes data, validates results. "
        "Prompt: describe what you want to build or migrate; specify source/target databases "
        "and tables. Returns JSON with {response, tokens_used}."
    ),
    "gen_dashboard": (
        "Create, update, and manage BI dashboards (Superset, Grafana). "
        "Handles the full workflow: write_query to materialize data, create datasets, "
        "create charts with appropriate visualizations, assemble dashboards. "
        "Also supports read operations: list/get dashboards, list charts and datasets. "
        "Prompt: describe what you want to visualize or which dashboard to inspect. "
        "Returns JSON with {response, dashboard_result, tokens_used}."
    ),
    "scheduler": (
        "Submit, monitor, update, and troubleshoot scheduled jobs on Airflow. "
        "Handles the full lifecycle: submit SQL/SparkSQL jobs with cron schedules, "
        "monitor job status and run history, view run logs, troubleshoot failures, "
        "update job SQL/config, pause/resume/delete jobs, trigger manual runs. "
        "Prompt: describe what scheduler operation you need. "
        "Returns JSON with {response, scheduler_result, tokens_used}."
    ),
}


class SubAgentTaskTool:
    """Delegate specialized tasks to AgenticNode instances within ChatAgenticNode.

    Supports an internal ``gen_sql`` type (always available) and any custom
    subagents declared in ``agent.yml`` under ``agentic_nodes``.

    Each subagent is a real AgenticNode instance (e.g., GenSQLAgenticNode)
    with its own session, tools, and configuration. A fresh node is created
    for every task invocation to ensure fully independent context.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        allowed_subagents: Optional[List[str]] = None,
        parent_node_name: Optional[str] = None,
    ):
        self.agent_config = agent_config
        self._allowed_subagents = allowed_subagents
        self._parent_node_name = parent_node_name
        self._action_bus: Optional["ActionBus"] = None
        self._interaction_broker: Optional["InteractionBroker"] = None
        self._parent_node: Optional["AgenticNode"] = None

    def set_action_bus(self, bus: "ActionBus") -> None:
        """Inject the :class:`ActionBus` for forwarding sub-agent actions."""
        self._action_bus = bus

    def set_interaction_broker(self, broker: "InteractionBroker") -> None:
        """Inject the parent's :class:`InteractionBroker` for transparent pass-through.

        When set, sub-agent hooks will use the parent's broker for user interactions.
        This ensures that CLI/Web ``submit()`` calls on ``current_node.interaction_broker``
        correctly resolve sub-agent interaction futures.
        """
        self._interaction_broker = broker

    def set_parent_node(self, node: "AgenticNode") -> None:
        """Store a reference to the parent :class:`AgenticNode`.

        The parent's ``proxy_tool_patterns`` and ``tool_channel`` are read
        lazily in :meth:`_execute_node` so sub-agent tools are automatically
        proxied when the parent has proxy tools configured.
        """
        self._parent_node = node

    # ── public API ──────────────────────────────────────────────────────

    def available_tools(self) -> List[Tool]:
        """Return a single ``task`` FunctionTool with a dynamic description."""
        description = self._build_task_description()
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "The subagent type to delegate to",
                },
                "prompt": {
                    "type": "string",
                    "description": "The task/question to send to the subagent",
                },
                "description": {
                    "type": "string",
                    "description": "A short one-line summary of the task goal (shown in compact display)",
                },
            },
            "required": ["type", "prompt", "description"],
        }

        async def _invoke(_tool_ctx, args_str) -> dict:
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else dict(args_str or {})
            except (TypeError, json.JSONDecodeError):
                return FuncToolResult(success=0, error="Invalid JSON arguments for task tool").model_dump()
            # Resolve parent call_id from SDK ToolContext for action linking
            call_id = getattr(_tool_ctx, "tool_call_id", None) if _tool_ctx else None
            result = await self.task(call_id=call_id, **args)
            return result.model_dump()

        return [
            FunctionTool(
                name="task",
                description=description,
                params_json_schema=schema,
                on_invoke_tool=_invoke,
                strict_json_schema=False,
            )
        ]

    async def task(
        self, type: str = "", prompt: str = "", description: str = "", call_id: Optional[str] = None
    ) -> FuncToolResult:
        """Execute a subagent task of the given *type*."""
        if not type:
            return FuncToolResult(success=0, error="Missing required parameter: type")
        if not prompt:
            return FuncToolResult(success=0, error="Missing required parameter: prompt")

        try:
            return await self._execute_node(type, prompt, description=description, call_id=call_id)
        except Exception as e:
            logger.error(f"Task tool execution error (type={type}): {e}")
            return FuncToolResult(success=0, error=f"Task execution failed: {str(e)}")

    # ── node creation ─────────────────────────────────────────────────

    def _create_node(self, subagent_type: str):
        """Create a new AgenticNode instance for the given subagent type.

        Both builtin (SYS_SUB_AGENTS) and custom agents propagate
        ``is_subagent=True`` so the child's constructor skips SubAgentTaskTool
        setup entirely — enforcing strict 2-level depth at the source rather
        than stripping tools post-construction.
        """
        # Builtin system subagents have non-standard constructors
        if subagent_type in SYS_SUB_AGENTS:
            return self._create_builtin_node(subagent_type)

        node_type, node_name = self._resolve_node_type(subagent_type)
        node_id = f"task_{subagent_type}_{uuid.uuid4().hex[:8]}"
        description = f"SubAgent task: {subagent_type}"

        from datus.agent.node.node import Node

        return Node.new_instance(
            node_id=node_id,
            description=description,
            node_type=node_type,
            agent_config=self.agent_config,
            node_name=node_name,
            is_subagent=True,
        )

    def _resolve_execution_mode(self) -> Literal["interactive", "workflow"]:
        """Resolve execution_mode from the parent node, defaulting to 'interactive'."""
        if self._parent_node and hasattr(self._parent_node, "execution_mode"):
            mode = self._parent_node.execution_mode
            if mode in ("interactive", "workflow"):
                return mode
        return "interactive"

    def _create_builtin_node(self, subagent_type: str):
        """Create a builtin system subagent node with its non-standard constructor."""
        if subagent_type == "gen_semantic_model":
            from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

            return GenSemanticModelAgenticNode(
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
            )
        elif subagent_type == "gen_metrics":
            from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

            return GenMetricsAgenticNode(
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
            )
        elif subagent_type == "gen_sql_summary":
            from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

            return SqlSummaryAgenticNode(
                node_name="gen_sql_summary",
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
            )
        elif subagent_type == "gen_ext_knowledge":
            from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode

            return GenExtKnowledgeAgenticNode(
                node_name="gen_ext_knowledge",
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
            )
        elif subagent_type == "gen_sql":
            from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

            return GenSQLAgenticNode(
                node_id=f"task_gen_sql_{uuid.uuid4().hex[:8]}",
                description="SQL generation node for gen_sql",
                node_type="gensql",
                input_data=None,
                agent_config=self.agent_config,
                tools=None,
                node_name="gen_sql",
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
            )
        elif subagent_type == "gen_report":
            from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

            return GenReportAgenticNode(
                node_id=f"task_gen_report_{uuid.uuid4().hex[:8]}",
                description="Report generation node for gen_report",
                node_type="gen_report",
                input_data=None,
                agent_config=self.agent_config,
                tools=None,
                node_name="gen_report",
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
            )
        elif subagent_type == "gen_table":
            from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

            return GenTableAgenticNode(
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                node_id=f"task_gen_table_{uuid.uuid4().hex[:8]}",
                is_subagent=True,
            )
        elif subagent_type == "gen_job":
            from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

            return GenJobAgenticNode(
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
            )
        elif subagent_type == "gen_skill":
            from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

            return SkillCreatorAgenticNode(
                node_id=f"task_gen_skill_{uuid.uuid4().hex[:8]}",
                description="Skill generation node",
                node_type="gen_skill",
                input_data=None,
                agent_config=self.agent_config,
                tools=None,
                node_name="gen_skill",
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
            )
        elif subagent_type == "gen_dashboard":
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            return GenDashboardAgenticNode(
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                node_id=f"task_gen_dashboard_{uuid.uuid4().hex[:8]}",
                is_subagent=True,
            )
        elif subagent_type == "scheduler":
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            return SchedulerAgenticNode(
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                node_id=f"task_scheduler_{uuid.uuid4().hex[:8]}",
                is_subagent=True,
            )
        else:
            raise ValueError(f"Unknown builtin subagent type: {subagent_type}")

    def _resolve_node_type(self, subagent_type: str) -> tuple:
        """Resolve subagent type string to (NodeType, node_name) tuple.

        Returns:
            Tuple of (node_type_constant, node_name_for_config).

        Raises:
            ValueError: If the subagent type is not recognized.
        """
        # Built-in gen_sql type
        if subagent_type == "gen_sql":
            for key in ("gen_sql", "gensql"):
                if key in self.agent_config.agentic_nodes:
                    return NodeType.TYPE_GENSQL, key
            return NodeType.TYPE_GENSQL, "gen_sql"

        # Built-in explore type
        if subagent_type == "explore":
            return NodeType.TYPE_EXPLORE, "explore"

        # Built-in gen_report type
        if subagent_type == "gen_report":
            return NodeType.TYPE_GEN_REPORT, "gen_report"

        # Built-in system subagents (SYS_SUB_AGENTS)
        builtin_type_map = {
            "gen_semantic_model": (NodeType.TYPE_SEMANTIC, "gen_semantic_model"),
            "gen_metrics": (NodeType.TYPE_SEMANTIC, "gen_metrics"),
            "gen_sql_summary": (NodeType.TYPE_SQL_SUMMARY, "gen_sql_summary"),
            "gen_ext_knowledge": (NodeType.TYPE_EXT_KNOWLEDGE, "gen_ext_knowledge"),
            "gen_table": (NodeType.TYPE_GEN_TABLE, "gen_table"),
            "gen_job": (NodeType.TYPE_GEN_JOB, "gen_job"),
            "gen_dashboard": (NodeType.TYPE_GEN_DASHBOARD, "gen_dashboard"),
            "scheduler": (NodeType.TYPE_SCHEDULER, "scheduler"),
        }
        if subagent_type in builtin_type_map:
            return builtin_type_map[subagent_type]

        # Custom subagent from agent.yml agentic_nodes
        sub_config = self.agent_config.sub_agent_config(subagent_type)
        if not sub_config:
            raise ValueError(f"Unknown subagent type: {subagent_type}")

        node_class = (
            sub_config.get("node_class") if isinstance(sub_config, dict) else getattr(sub_config, "node_class", None)
        )
        node_type = NODE_CLASS_MAP.get(node_class or "gen_sql", NodeType.TYPE_GENSQL)
        return node_type, subagent_type

    # ── broker injection ──────────────────────────────────────────────

    def _inject_broker(self, node, broker: "InteractionBroker") -> None:
        """Inject the parent's InteractionBroker into a sub-agent node and its hooks.

        This replaces the sub-agent's own broker so that INTERACTION actions
        are routed through the parent's broker queue.  The parent's
        ``action_bus.merge(execute_stream, broker.fetch())`` then picks them up
        and the CLI/Web ``submit()`` call on ``current_node.interaction_broker``
        correctly resolves the pending futures.
        """
        node.interaction_broker = broker

        # Update broker reference on ask_user_tool that was already initialised
        # with the node's original (now stale) broker.
        ask_user_tool = getattr(node, "ask_user_tool", None)
        if ask_user_tool is not None and hasattr(ask_user_tool, "_broker"):
            ask_user_tool._broker = broker

        # Update broker references on hooks that were already initialised
        # with the node's original (now stale) broker.
        for attr in ("hooks", "permission_hooks", "plan_hooks"):
            hooks_obj = getattr(node, attr, None)
            if hooks_obj is None:
                continue
            # Direct hook (GenerationHooks, PermissionHooks, PlanModeHooks)
            if hasattr(hooks_obj, "broker"):
                hooks_obj.broker = broker
            # CompositeHooks wrapping multiple hooks
            if hasattr(hooks_obj, "hooks_list"):
                for h in hooks_obj.hooks_list:
                    if hasattr(h, "broker"):
                        h.broker = broker

    # ── execution via execute_stream ───────────────────────────────────

    async def _execute_node(
        self, subagent_type: str, prompt: str, description: str = "", call_id: Optional[str] = None
    ) -> FuncToolResult:
        """Execute a subagent by running an AgenticNode's execute_stream."""
        # Validate subagent type against the allowlist to prevent privilege escalation.
        # Normalize the LLM-supplied value first — strip whitespace/newlines and the
        # surrounding quotes some models wrap around string arguments — before
        # comparing against the discoverable allowlist.
        allowed_types = self._get_available_types()
        raw_subagent_type = subagent_type
        normalized = subagent_type.strip().strip("\"'") if isinstance(subagent_type, str) else subagent_type
        if normalized in allowed_types:
            subagent_type = normalized
        else:
            logger.warning(
                "Subagent type rejected: raw=%r normalized=%r parent=%r allowed=%r",
                raw_subagent_type,
                normalized,
                self._parent_node_name,
                allowed_types,
            )
            return FuncToolResult(
                success=0,
                error=(
                    f"Unknown or disallowed subagent type: {raw_subagent_type!r} "
                    f"(normalized {normalized!r}). Available types: {allowed_types}"
                ),
            )

        node = self._create_node(subagent_type)
        node.ephemeral = True  # Use in-memory session — no SQLite persistence for sub-agents

        # Set input on the node
        node.input = self._build_node_input(node, prompt)

        # Inject parent's InteractionBroker so that sub-agent INTERACTION
        # actions are routed through the parent's broker queue.  When injected,
        # we call execute_stream() (not execute_stream_with_interactions()) to
        # avoid dual-consuming the same broker.fetch() stream.
        if self._interaction_broker is not None:
            self._inject_broker(node, self._interaction_broker)

        # Propagate proxy tool config from parent node so sub-agent tools are
        # also proxied.  Uses the parent's tool_channel so stdin dispatch can
        # resolve futures for both parent and sub-agent tools.
        # Note: apply_proxy_tools internally detects fs-dependent nodes and
        # excludes their filesystem_tools category from proxying.
        if self._parent_node and self._parent_node.proxy_tool_patterns:
            from datus.tools.proxy.proxy_tool import apply_proxy_tools

            apply_proxy_tools(node, self._parent_node.proxy_tool_patterns, channel=self._parent_node.tool_channel)

        # Iterate the async generator directly (we're already in async context)
        action_history_manager = ActionHistoryManager()
        final_output = None

        # When parent broker is injected, INTERACTION actions flow through the
        # parent's broker.fetch() → parent merge → CLI.  We only need
        # execute_stream() here; otherwise fall back to the full merge.
        if self._interaction_broker is not None:
            stream = node.execute_stream(action_history_manager)
        else:
            stream = node.execute_stream_with_interactions(action_history_manager)

        stream_start_time = datetime.now()
        tool_count = 0
        subagent_status = ActionStatus.SUCCESS
        first_user_seen = False

        try:
            async for action in stream:
                # Inject _task_description into the first USER action for display
                if not first_user_seen and action.role == ActionRole.USER:
                    if description:
                        if action.input is None:
                            action.input = {}
                        if isinstance(action.input, dict):
                            action.input["_task_description"] = description
                    first_user_seen = True

                # Forward sub-action to the ActionBus (real-time CoT streaming)
                if self._action_bus is not None:
                    action.depth = 1
                    if call_id:
                        action.parent_action_id = call_id
                    logger.debug(
                        "SubAgentTaskTool bus.put",
                        action_type=action.action_type,
                        role=str(action.role),
                        status=str(action.status),
                    )
                    self._action_bus.put(action)

                if action.role == ActionRole.TOOL:
                    tool_count += 1

                if action.status == ActionStatus.FAILED:
                    subagent_status = ActionStatus.FAILED
                    if action.output:
                        final_output = action.output
                elif action.status == ActionStatus.SUCCESS and action.output:
                    final_output = action.output
        except Exception:
            subagent_status = ActionStatus.FAILED
            raise
        finally:
            self._emit_complete_action(subagent_type, call_id, stream_start_time, tool_count, subagent_status)
            # Cleanup node resources (MCP connections, sessions, file handles)
            try:
                node.delete_session()
            except Exception:
                logger.debug("Failed to cleanup sub-agent node session", exc_info=True)

        return self._convert_to_func_result(final_output)

    def _emit_complete_action(
        self,
        subagent_type: str,
        call_id: Optional[str],
        stream_start_time: datetime,
        tool_count: int,
        status: ActionStatus,
    ) -> None:
        """Emit a ``subagent_complete`` action to signal that a sub-agent has finished."""
        if self._action_bus is None:
            return

        complete = ActionHistory.create_action(
            role=ActionRole.SYSTEM,
            action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
            messages="",
            input_data=None,
            status=status,
        )
        complete.depth = 1
        complete.parent_action_id = call_id
        complete.end_time = datetime.now()
        complete.start_time = stream_start_time
        complete.output = {"subagent_type": subagent_type, "tool_count": tool_count}
        self._action_bus.put(complete)

    # ── input building ─────────────────────────────────────────────────

    def _build_node_input(self, node, prompt: str):
        """Build the appropriate input object for the given node."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.schemas.explore_agentic_node_models import ExploreNodeInput
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        if isinstance(node, ExploreAgenticNode):
            return ExploreNodeInput(
                user_message=prompt,
                database=self.agent_config.current_datasource,
            )

        if isinstance(node, GenSQLAgenticNode):
            return GenSQLNodeInput(
                user_message=prompt,
                database=self.agent_config.current_datasource,
            )

        # Built-in system subagent input types
        from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode
        from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

        if isinstance(node, (GenTableAgenticNode, GenJobAgenticNode)):
            from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

            return SemanticNodeInput(
                user_message=prompt,
                database=self.agent_config.current_datasource,
            )

        if isinstance(node, (GenSemanticModelAgenticNode, GenMetricsAgenticNode)):
            from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

            return SemanticNodeInput(
                user_message=prompt,
                database=self.agent_config.current_datasource,
            )

        if isinstance(node, SqlSummaryAgenticNode):
            from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput

            return SqlSummaryNodeInput(
                user_message=prompt,
                database=self.agent_config.current_datasource,
            )

        if isinstance(node, GenExtKnowledgeAgenticNode):
            from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput

            return ExtKnowledgeNodeInput(user_message=prompt)

        from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

        if isinstance(node, GenDashboardAgenticNode):
            from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput

            return GenDashboardNodeInput(
                user_message=prompt,
                database=self.agent_config.current_datasource,
            )

        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

        if isinstance(node, SchedulerAgenticNode):
            from datus.schemas.scheduler_agentic_node_models import SchedulerNodeInput

            return SchedulerNodeInput(
                user_message=prompt,
                database=self.agent_config.current_datasource,
            )

        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        if isinstance(node, GenReportAgenticNode):
            from datus.schemas.gen_report_agentic_node_models import GenReportNodeInput

            return GenReportNodeInput(
                user_message=prompt,
                database=self.agent_config.current_datasource,
            )

        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        if isinstance(node, SkillCreatorAgenticNode):
            from datus.schemas.gen_skill_agentic_node_models import SkillCreatorNodeInput

            return SkillCreatorNodeInput(user_message=prompt)

        # Generic fallback for other agentic node types
        from datus.schemas.base import BaseInput

        # Try to use the node's type-specific input if available
        try:
            input_cls = NodeType.type_input(node.type, {}, ignore_require_check=True)
            if hasattr(input_cls, "user_message"):
                input_cls.user_message = prompt
            return input_cls
        except Exception as e:
            logger.debug(f"Failed to build type-specific input for {node.type}: {e}")

        return BaseInput()

    # ── result conversion ──────────────────────────────────────────────

    def _convert_to_func_result(self, output) -> FuncToolResult:
        """Convert AgenticNode output to FuncToolResult."""
        if not output or not isinstance(output, dict):
            return FuncToolResult(success=0, error="No result from subagent")

        # Check for explicit failure from subagent
        if output.get("success") is False:
            return FuncToolResult(
                success=0,
                error=output.get("error") or output.get("response") or output.get("content", "Subagent failed"),
            )

        response = output.get("response", "")
        tokens = output.get("tokens_used", 0)

        # File-based SQL result: sql_file_path present
        sql_file_path = output.get("sql_file_path")
        if sql_file_path:
            result_dict: Dict[str, Any] = {
                "sql_file_path": sql_file_path,
                "sql_preview": output.get("sql_preview", ""),
                "response": response,
                "tokens_used": tokens,
            }
            sql_diff = output.get("sql_diff")
            if sql_diff:
                result_dict["sql_diff"] = sql_diff
            return FuncToolResult(result=result_dict)

        # Inline SQL result: has 'sql' key
        sql = output.get("sql")
        if sql is not None:
            return FuncToolResult(
                result={
                    "sql": sql,
                    "response": response,
                    "tokens_used": tokens,
                }
            )

        # Semantic model result: has 'semantic_models' key
        semantic_models = output.get("semantic_models")
        if semantic_models is not None:
            return FuncToolResult(
                result={
                    "response": response,
                    "semantic_models": semantic_models,
                    "tokens_used": tokens,
                }
            )

        # SQL summary result: has 'sql_summary_file' key
        sql_summary_file = output.get("sql_summary_file")
        if sql_summary_file is not None:
            return FuncToolResult(
                result={
                    "response": response,
                    "sql_summary_file": sql_summary_file,
                    "tokens_used": tokens,
                }
            )

        # External knowledge result: has 'ext_knowledge_file' key
        ext_knowledge_file = output.get("ext_knowledge_file")
        if ext_knowledge_file is not None:
            return FuncToolResult(
                result={
                    "response": response,
                    "ext_knowledge_file": ext_knowledge_file,
                    "tokens_used": tokens,
                }
            )

        # Report result: has 'report_result' key
        report_result = output.get("report_result")
        if report_result is not None:
            return FuncToolResult(
                result={
                    "response": response,
                    "report_result": report_result,
                    "tokens_used": tokens,
                }
            )

        # Skill creator result: has 'skill_path' key
        skill_path = output.get("skill_path")
        if skill_path is not None:
            return FuncToolResult(
                result={
                    "response": response,
                    "skill_name": output.get("skill_name", ""),
                    "skill_path": skill_path,
                    "tokens_used": tokens,
                }
            )

        # Dashboard result: has 'dashboard_result' key
        dashboard_result = output.get("dashboard_result")
        if dashboard_result is not None:
            return FuncToolResult(
                result={
                    "response": response,
                    "dashboard_result": dashboard_result,
                    "tokens_used": tokens,
                }
            )

        # Scheduler result: has 'scheduler_result' key
        scheduler_result = output.get("scheduler_result")
        if scheduler_result is not None:
            return FuncToolResult(
                result={
                    "response": response,
                    "scheduler_result": scheduler_result,
                    "tokens_used": tokens,
                }
            )

        # Feedback result: has 'items_saved' key
        items_saved = output.get("items_saved")
        if items_saved is not None:
            return FuncToolResult(
                result={
                    "response": response,
                    "items_saved": items_saved,
                    "storage_summary": output.get("storage_summary"),
                    "tokens_used": tokens,
                }
            )

        # Generic format
        return FuncToolResult(
            result={
                "response": response or output.get("content", ""),
                "tokens_used": tokens,
            }
        )

    # ── description builder ────────────────────────────────────────────

    def _build_task_description(self) -> str:
        """Build a dynamic description for the task tool."""
        available = self._get_available_types()

        lines = [
            "Delegate a complex task to a specialized subagent. "
            "Only use this for questions that require deep exploration or multi-step SQL reasoning. "
            "For simple/direct questions, use your own tools (list_tables, describe_table, read_query, etc.) instead.",
            "",
            "Available types:",
        ]

        for t in available:
            if t in BUILTIN_SUBAGENT_DESCRIPTIONS:
                lines.append(f"- {t}: {BUILTIN_SUBAGENT_DESCRIPTIONS[t]}")
            else:
                sub_raw = self.agent_config.sub_agent_config(t)
                desc = ""
                if isinstance(sub_raw, dict):
                    desc = sub_raw.get("agent_description", "") or ""
                elif hasattr(sub_raw, "agent_description"):
                    desc = getattr(sub_raw, "agent_description", "") or ""
                lines.append(f"- {t}: {desc}" if desc else f"- {t}")

        lines.extend(
            [
                "",
                "Guidelines:",
                "- For simple questions, handle directly with your own tools — no need to launch subagents",
                '- For complex questions requiring deep exploration, call multiple task(type="explore") '
                "in PARALLEL, each with a direction-specific prompt (schema+sample, knowledge, file)",
                '- For quick single-direction lookups, call one task(type="explore") with a focused prompt',
                '- Use task(type="gen_sql") for SQL generation requiring multi-step reasoning, '
                "complex joins, or domain-specific logic",
                '- Use task(type="gen_report") for metric attribution, root cause analysis, '
                "or analyzing why a metric/reference_sql result changed",
                '- Use task(type="gen_skill") when the user wants to create a new skill or optimize an existing skill',
                '- Use task(type="gen_dashboard") for creating/updating/inspecting BI dashboards, '
                "charts, and datasets on Superset or Grafana",
                '- Use task(type="scheduler") for submitting, monitoring, updating, '
                "and troubleshooting scheduled jobs on Airflow",
                "- In plan mode, use task() for each SQL sub-step",
                "- Always provide a short 'description' summarizing the task goal",
            ]
        )

        return "\n".join(lines)

    def _get_available_types(self) -> List[str]:
        """Discover available subagent types, filtered by allowed_subagents and excluding self.

        In explicit list mode, unknown types are filtered out with a warning so
        that a misconfigured ``subagents: "foo, bar"`` surfaces as a log message
        instead of a cryptic ``_create_node`` failure at runtime.
        """
        if self._allowed_subagents is not None:
            # Explicit list mode: filter against the discoverable universe,
            # warn on unknown names, and exclude self.
            discoverable = self._discover_all_types()
            result: List[str] = []
            for t in self._allowed_subagents:
                if t == self._parent_node_name:
                    continue
                if t not in discoverable:
                    logger.warning(
                        f"Subagent type '{t}' in allowed_subagents is not a known type "
                        f"(parent={self._parent_node_name}); skipping. "
                        f"Known types: {sorted(discoverable)}"
                    )
                    continue
                result.append(t)
            return result

        # Wildcard mode (*): all discovered types, excluding self.
        return [t for t in self._discover_all_types() if t != self._parent_node_name]

    def _discover_all_types(self) -> List[str]:
        """Return every subagent type that can currently be instantiated.

        'feedback' is a top-level AgenticNode (invoked directly by the CLI/API),
        not a delegatable subagent, so it is excluded here even though it lives
        in SYS_SUB_AGENTS (which only guards reserved system names).
        """
        types = ["explore"]
        types.extend(sorted(name for name in SYS_SUB_AGENTS if name != "feedback"))

        if self.agent_config and hasattr(self.agent_config, "agentic_nodes"):
            current_datasource = self.agent_config.current_datasource

            for name, config in self.agent_config.agentic_nodes.items():
                if name in ("chat", "explore", "feedback") or name in SYS_SUB_AGENTS:
                    continue

                # If scoped_context is configured, namespace must match current namespace
                try:
                    sub_config = SubAgentConfig.model_validate(config)
                    if sub_config.has_scoped_context() and not sub_config.is_in_namespace(current_datasource):
                        continue
                except Exception as e:
                    logger.debug(f"Skipping invalid subagent config '{name}': {e}")
                    continue

                types.append(name)

        return types
