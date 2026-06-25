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
from contextlib import nullcontext
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from agents import FunctionTool, Tool

from datus.configuration.agent_config import AgentConfig
from datus.configuration.inherited_memory_overrides import inherited_memory
from datus.configuration.node_type import NodeType
from datus.configuration.scoped_context_overrides import effective_subagent
from datus.schemas.action_history import (
    SUBAGENT_COMPLETE_ACTION_TYPE,
    ActionHistory,
    ActionHistoryManager,
    ActionRole,
    ActionStatus,
)
from datus.schemas.agent_models import ScopedContext, SubAgentConfig
from datus.tools.func_tool.base import FuncToolResult
from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.loggings import get_logger
from datus.utils.memory_loader import resolve_memory_node

if TYPE_CHECKING:
    from datus.agent.node.agentic_node import AgenticNode
    from datus.cli.execution_state import InteractionBroker
    from datus.schemas.action_bus import ActionBus

logger = get_logger(__name__)

# Mapping from subagent type string to NodeType constants
NODE_CLASS_MAP = {
    "gen_sql": NodeType.TYPE_GEN_SQL,
    "chat": NodeType.TYPE_CHAT,
    "ask_metrics": NodeType.TYPE_ASK_METRICS,
    "gen_report": NodeType.TYPE_GEN_REPORT,
    "gen_visual_report": NodeType.TYPE_GEN_VISUAL_REPORT,
    "gen_visual_dashboard": NodeType.TYPE_GEN_VISUAL_DASHBOARD,
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
        "pass sql_file_path directly to execute_sql() to execute (no need to read_file() first). "
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
    "ask_metrics": (
        "Answer metric-based questions quickly using existing semantic metrics. "
        "Use for KPI values, trends, grouped metric results, and metric attribution "
        "when the question can be answered by the semantic layer. Does not fall back "
        "to raw SQL. Returns JSON with {response, markdown_report, tokens_used}."
    ),
    "gen_report": (
        "Legacy Markdown report subagent. Use only when the user explicitly asks to use "
        "the gen_report subagent; do not automatically route attribution, root-cause, "
        "trend explanation, or report requests here. "
        "Prompt: provide the user's explicit gen_report request and any supplied context. "
        "Returns JSON with {response, report_result, tokens_used}."
    ),
    "gen_visual_report": (
        "Produce a visualizable report artifact under "
        "<project_root>/reports/<id>/ (render/*.jsx + queries/<slug>.sql/.json). "
        "The subagent writes a render/ tree of React modules — the entry "
        "render/app.jsx imports siblings like render/kpi-banner.jsx via relative "
        "paths and pulls pre-executed query JSON via useQuerySql. "
        "Use this — instead of writing HTML/Markdown directly — whenever the user asks "
        "for a *report* with charts/tables, a dashboard, or any answer that benefits "
        "from persisted SQL + rendered visualisations (the artifact is consumed by "
        "Datus-CLI HTML export and Datus-SaaS dynamic iframe renderer). "
        "Prompt: provide the analytical question and any context (time range, scope, "
        "metrics of interest); the agent will discover data, save queries via save_query, "
        "write_file the render/*.jsx components, then finalize with validate_render. "
        "Returns JSON with {response, report_id, app_jsx_path, render_file_count, "
        "html_path, query_count, tokens_used}."
    ),
    "gen_visual_dashboard": (
        "Produce a parameterized visual dashboard artifact under "
        "<project_root>/dashboards/<slug>/ (render/*.jsx + queries/<slug>.sql.j2 + "
        "queries/<slug>.params.json + manifest.json). Unlike gen_visual_report, "
        "queries are persisted as Jinja2 SQL templates with declared parameter "
        "metadata; at view time the backend renders the template with user-selected "
        "filter values and executes it live against the bound datasource. "
        "Use this whenever the user asks for an interactive dashboard with filters "
        "(date range, region, segment, etc.) that should re-query data on demand, "
        "instead of a one-shot pre-baked report. The subagent picks a fresh slug "
        "via start_new_dashboard (or reuses one via bind_existing_dashboard), "
        "persists each query via save_query_template, write_file's the render/*.jsx "
        "components, then finalizes with validate_render. "
        "Prompt: provide the dashboard question/topic plus the filter dimensions "
        "the user wants exposed; the agent will discover data, build templates, and "
        "wire them to the JSX filter state. Returns JSON with {response, "
        "dashboard_slug, app_jsx_path, render_file_count, template_count, tokens_used}."
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
        "(3) Batch: provide multiple SQL queries for AST-backed metric candidate extraction. "
        "For batch input, if the user provides a CSV file path, YOU (the parent agent) must read the file content first "
        "and include the full content in the prompt — the metrics agent cannot access files outside its workspace. "
        "The metrics agent will preserve final business output expressions and treat base measures as dependencies. "
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
        "For cross-database transfer: transfers data between different database engines "
        "(e.g., DuckDB to Greenplum, MySQL to StarRocks, Postgres to ClickHouse). "
        "Handles cross-dialect type mapping, target DDL generation, data transfer via "
        "transfer_query_result, and lightweight post-transfer reconciliation "
        "(tool-reported row count parity plus target-side sanity checks) when source and target differ. "
        "Inspects source and target schemas, generates DDL, writes data, validates results. "
        "Prompt: describe what you want to build or migrate; specify source/target databases "
        "and tables. Returns JSON with {response, tokens_used}."
    ),
    "gen_dashboard": (
        "Create, update, and manage BI dashboards on the configured BI platform "
        "(Superset, Grafana, or any future adapter). Builds BI assets on top of "
        "tables or SQL datasets that already exist in a BI-registered database. "
        "Data preparation belongs to a separate gen_job or scheduler step before "
        "calling gen_dashboard. Prompt: provide the BI platform, serving table "
        "or SQL dataset, dimensions, time range, chart type, and dashboard title. "
        "Also supports read-only ops (list/get dashboards, list charts and "
        "datasets). Returns JSON with {response, dashboard_result, tokens_used}."
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

    permission_category: str = "sub_agent_tools"

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
                "session_id": {
                    "type": "string",
                    "description": (
                        "Optional. Pass back a session_id from a previous task() result to "
                        "CONTINUE the same subagent's conversation with full prior context. "
                        "Use for iterative refinement (e.g. 'rewrite the previous SQL using "
                        "INNER JOIN' or 'narrow the report to the EU region'). Must belong "
                        "to a subagent of the SAME `type`. Omit to start a fresh session."
                    ),
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
        self,
        type: str = "",
        prompt: str = "",
        description: str = "",
        call_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> FuncToolResult:
        """Execute a subagent task of the given *type*.

        When ``session_id`` is provided the subagent resumes that prior session
        from disk so the new ``prompt`` is appended to existing turn history.
        """
        if not type:
            return FuncToolResult(success=0, error="Missing required parameter: type")
        if not prompt:
            return FuncToolResult(success=0, error="Missing required parameter: prompt")

        try:
            return await self._execute_node(
                type, prompt, description=description, call_id=call_id, session_id=session_id
            )
        except Exception as e:
            logger.error(f"Task tool execution error (type={type}): {e}")
            return FuncToolResult(success=0, error=f"Task execution failed: {str(e)}")

    # ── node creation ─────────────────────────────────────────────────

    def _create_node(self, subagent_type: str, session_id: Optional[str] = None):
        """Create a new AgenticNode instance for the given subagent type.

        Both builtin (SYS_SUB_AGENTS) and custom agents propagate
        ``is_subagent=True`` so the child's constructor skips SubAgentTaskTool
        setup entirely — enforcing strict 2-level depth at the source rather
        than stripping tools post-construction.

        ``session_id`` is forwarded to the constructor so resume flows open the
        existing session DB on first turn — no post-construct mutation.
        """
        # Builtin system subagents have non-standard constructors
        if subagent_type in SYS_SUB_AGENTS:
            return self._create_builtin_node(subagent_type, session_id=session_id)

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
            session_id=session_id,
        )

    def _resolve_execution_mode(self) -> Literal["interactive", "workflow"]:
        """Resolve execution_mode from the parent node, defaulting to 'interactive'."""
        if self._parent_node and hasattr(self._parent_node, "execution_mode"):
            mode = self._parent_node.execution_mode
            if mode in ("interactive", "workflow"):
                return mode
        return "interactive"

    def _create_builtin_node(self, subagent_type: str, session_id: Optional[str] = None):
        """Create a builtin system subagent node with its non-standard constructor."""
        if subagent_type == "gen_semantic_model":
            from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

            return GenSemanticModelAgenticNode(
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
                session_id=session_id,
            )
        elif subagent_type == "gen_metrics":
            from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

            return GenMetricsAgenticNode(
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
                session_id=session_id,
            )
        elif subagent_type == "gen_sql_summary":
            from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

            return SqlSummaryAgenticNode(
                node_name="gen_sql_summary",
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
                session_id=session_id,
            )
        elif subagent_type == "gen_sql":
            from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

            return GenSQLAgenticNode(
                node_id=f"task_gen_sql_{uuid.uuid4().hex[:8]}",
                description="SQL generation node for gen_sql",
                node_type="gen_sql",
                input_data=None,
                agent_config=self.agent_config,
                tools=None,
                node_name="gen_sql",
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
                session_id=session_id,
            )
        elif subagent_type == "ask_metrics":
            from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

            return AskMetricsAgenticNode(
                node_id=f"task_ask_metrics_{uuid.uuid4().hex[:8]}",
                description="Metric question-answering node for ask_metrics",
                node_type="ask_metrics",
                input_data=None,
                agent_config=self.agent_config,
                tools=None,
                node_name="ask_metrics",
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
                session_id=session_id,
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
                session_id=session_id,
            )
        elif subagent_type == "gen_visual_report":
            # Same session_id contract as gen_visual_dashboard below —
            # ``BaseVisualArtifactAgenticNode.__init__`` doesn't accept
            # ``session_id``, so silently dropping it would let resume
            # loops spawn a fresh session per turn while the LLM
            # thinks it picked up an existing one. Fail loud.
            if session_id is not None:
                raise ValueError(
                    "gen_visual_report does not support session resume "
                    "(BaseVisualArtifactAgenticNode constructor has no "
                    "session_id parameter). Drop the session_id kwarg, "
                    "or use a resumable subagent type."
                )
            from datus.agent.node.gen_visual_report_agentic_node import GenVisualReportAgenticNode

            return GenVisualReportAgenticNode(
                node_id=f"task_gen_visual_report_{uuid.uuid4().hex[:8]}",
                description="Visual report generation node for gen_visual_report",
                node_type="gen_visual_report",
                input_data=None,
                agent_config=self.agent_config,
                tools=None,
                node_name="gen_visual_report",
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
            )
        elif subagent_type == "gen_visual_dashboard":
            # ``GenVisualDashboardAgenticNode`` inherits from
            # ``BaseVisualArtifactAgenticNode`` whose ``__init__`` does
            # NOT accept ``session_id`` (no resume flow yet — the
            # artifact directory + manifest itself is the persistence
            # boundary). Silently dropping a caller-supplied session_id
            # the way other visual subagents do would let resume loops
            # spawn a fresh session every turn while the LLM thinks it
            # picked up an existing one. Fail loud instead so the
            # caller can either drop the kwarg or wait for a
            # constructor-level resume API.
            if session_id is not None:
                raise ValueError(
                    "gen_visual_dashboard does not support session resume "
                    "(BaseVisualArtifactAgenticNode constructor has no "
                    "session_id parameter). Drop the session_id kwarg, "
                    "or use a resumable subagent type."
                )
            from datus.agent.node.gen_visual_dashboard_agentic_node import GenVisualDashboardAgenticNode

            return GenVisualDashboardAgenticNode(
                node_id=f"task_gen_visual_dashboard_{uuid.uuid4().hex[:8]}",
                description="Visual dashboard generation node for gen_visual_dashboard",
                node_type="gen_visual_dashboard",
                input_data=None,
                agent_config=self.agent_config,
                tools=None,
                node_name="gen_visual_dashboard",
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
                session_id=session_id,
            )
        elif subagent_type == "gen_job":
            from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

            return GenJobAgenticNode(
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                is_subagent=True,
                session_id=session_id,
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
                session_id=session_id,
            )
        elif subagent_type == "gen_dashboard":
            from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

            return GenDashboardAgenticNode(
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                node_id=f"task_gen_dashboard_{uuid.uuid4().hex[:8]}",
                is_subagent=True,
                session_id=session_id,
            )
        elif subagent_type == "scheduler":
            from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

            return SchedulerAgenticNode(
                agent_config=self.agent_config,
                execution_mode=self._resolve_execution_mode(),
                node_id=f"task_scheduler_{uuid.uuid4().hex[:8]}",
                is_subagent=True,
                session_id=session_id,
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
            return NodeType.TYPE_GEN_SQL, "gen_sql"

        # Built-in explore type
        if subagent_type == "explore":
            return NodeType.TYPE_EXPLORE, "explore"

        # Built-in gen_report type
        if subagent_type == "gen_report":
            return NodeType.TYPE_GEN_REPORT, "gen_report"

        if subagent_type == "ask_metrics":
            return NodeType.TYPE_ASK_METRICS, "ask_metrics"

        # Built-in gen_visual_report type
        if subagent_type == "gen_visual_report":
            return NodeType.TYPE_GEN_VISUAL_REPORT, "gen_visual_report"

        # Built-in gen_visual_dashboard type
        if subagent_type == "gen_visual_dashboard":
            return NodeType.TYPE_GEN_VISUAL_DASHBOARD, "gen_visual_dashboard"

        # Built-in system subagents (SYS_SUB_AGENTS)
        builtin_type_map = {
            "gen_semantic_model": (NodeType.TYPE_SEMANTIC, "gen_semantic_model"),
            "gen_metrics": (NodeType.TYPE_SEMANTIC, "gen_metrics"),
            "gen_sql_summary": (NodeType.TYPE_SQL_SUMMARY, "gen_sql_summary"),
            "ask_metrics": (NodeType.TYPE_ASK_METRICS, "ask_metrics"),
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
        node_type = NODE_CLASS_MAP.get(node_class or "gen_sql", NodeType.TYPE_GEN_SQL)
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
        for attr in ("hooks", "permission_hooks"):
            hooks_obj = getattr(node, attr, None)
            if hooks_obj is None:
                continue
            # Direct hook (GenerationHooks, PermissionHooks)
            if hasattr(hooks_obj, "broker"):
                hooks_obj.broker = broker
            # CompositeHooks wrapping multiple hooks
            if hasattr(hooks_obj, "hooks_list"):
                for h in hooks_obj.hooks_list:
                    if hasattr(h, "broker"):
                        h.broker = broker

    # ── execution via execute_stream ───────────────────────────────────

    async def _execute_node(
        self,
        subagent_type: str,
        prompt: str,
        description: str = "",
        call_id: Optional[str] = None,
        session_id: Optional[str] = None,
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

        effective_cfg = self._resolve_effective_sub_agent_config(subagent_type)
        inherited_parent = self._resolve_inherited_memory_node(subagent_type)
        inherited_cm = inherited_memory(subagent_type, inherited_parent) if inherited_parent else nullcontext()

        # Validate the session_id format up-front (path-injection defence) so
        # we fail fast before paying the node-construction cost.
        from datus.models.session_manager import SessionManager, extract_agent_from_session_id

        if session_id is not None:
            try:
                SessionManager._validate_session_id(session_id)
            except ValueError as e:
                return FuncToolResult(success=0, error=f"Invalid session_id format: {e}")

        with effective_subagent(subagent_type, effective_cfg), inherited_cm:
            # Resume an existing session when caller provides a session_id of the SAME type.
            # Ownership validation runs *before* node construction so a mismatched
            # session_id never reaches __init__ (where it would trigger an
            # unnecessary ``restore_plan_mode_state`` for the wrong session).
            if session_id is not None:
                actual_owner = extract_agent_from_session_id(session_id)
                if subagent_type in SYS_SUB_AGENTS:
                    expected_owner = subagent_type
                else:
                    _, expected_owner = self._resolve_node_type(subagent_type)
                allowed_owners = {expected_owner}
                if actual_owner not in allowed_owners:
                    return FuncToolResult(
                        success=0,
                        error=(
                            f"session_id {session_id!r} belongs to subagent type {actual_owner!r} "
                            f"but task requested type {subagent_type!r}. Each session is bound "
                            "to one subagent type."
                        ),
                    )

            node = self._create_node(subagent_type, session_id=session_id)

            # Nest this subagent's session under the launching main session so the
            # parent LLM can later resume by passing back the returned session_id.
            # Path: {sessions_dir}/{user_scope}/{parent_session_id}/{subagent_session_id}.db
            parent_sid = getattr(self._parent_node, "session_id", None)
            if isinstance(parent_sid, str) and parent_sid:
                try:
                    SessionManager._validate_session_id(parent_sid)
                    node.session_subdir = parent_sid
                except ValueError:
                    logger.warning(
                        "Parent session_id %r failed validation; falling back to flat layout",
                        parent_sid,
                    )

            # Verify the .db file actually exists after session_subdir is wired up
            # — session_manager resolves the nested directory layout per
            # ``AgenticNode.session_manager``.
            if session_id is not None and not node.session_manager.session_exists(session_id):
                return FuncToolResult(
                    success=0,
                    error=(
                        f"session_id {session_id!r} not found on disk under the current "
                        "main session. It may have been cleaned up or never existed."
                    ),
                )

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
            # parent's broker.fetch() → parent merge → CLI, so we must NOT consume
            # the sub-agent's own broker (that would dual-consume the injected
            # stream). We still merge the sub-agent's ``action_bus`` though: hook-
            # enqueued actions — notably ``TokenUsageHook``'s per-call
            # ``token_usage`` — are delivered via ``action_bus.put`` and would
            # otherwise never be yielded by the bare ``execute_stream`` (which only
            # surfaces ``_stream_once`` output). Without this merge the parent never
            # sees the sub-agent's token usage and its pinned-header counter stays 0.
            if self._interaction_broker is not None:
                node.action_bus.reset()
                stream = node.action_bus.merge(node.execute_stream(action_history_manager))
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
            except Exception as e:
                # Surface the failure as an envelope (not a re-raise) so the
                # parent agent can resume the partial subagent session via
                # the returned session_id — e.g. when MaxTurnsExceeded
                # interrupts a long workflow mid-task.
                subagent_status = ActionStatus.FAILED
                logger.error(
                    "Subagent stream error (type=%s, session_id=%s): %s",
                    subagent_type,
                    getattr(node, "session_id", None),
                    e,
                    exc_info=True,
                )
                final_output = {"success": False, "error": f"Subagent stream failed: {e}"}
            finally:
                self._emit_complete_action(subagent_type, call_id, stream_start_time, tool_count, subagent_status)
                # Release in-memory handles WITHOUT deleting the .db file — the parent
                # LLM may resume this session_id on a later turn.
                try:
                    if node._session_manager is not None:
                        node._session_manager.close_all_sessions()
                    node._session = None
                except Exception:
                    logger.debug("Failed to release sub-agent session handle", exc_info=True)

        return self._convert_to_func_result(final_output, session_id=node.session_id)

    def _resolve_inherited_memory_node(self, subagent_type: str) -> Optional[str]:
        """Pick the memory node a sub-agent should inherit (read-only inline).

        Every sub-agent (built-in or custom) sees its parent's memory inlined
        read-only — sub-agents never write memory. Returns the parent's resolved
        memory node (``resolve_memory_node`` maps built-in parents to the shared
        ``chat`` memory; custom parents to their own name), or ``None`` when:
        - the child is ``feedback`` — it injects the caller's memory via
          ``override_node_name`` and would otherwise double-render;
        - no parent node is registered.
        """
        if subagent_type == "feedback":
            return None
        if self._parent_node is None:
            return None
        try:
            parent_name = self._parent_node.get_node_name()
        except Exception:
            return None
        if not parent_name:
            return None
        return resolve_memory_node(parent_name)

    def _resolve_effective_sub_agent_config(self, subagent_type: str) -> SubAgentConfig:
        """Build an effective SubAgentConfig that inherits parent scoped_context when child has none."""
        parent_sc: Optional[ScopedContext] = None
        parent_cfg = getattr(self._parent_node, "node_config", None)
        if isinstance(parent_cfg, dict):
            raw = parent_cfg.get("scoped_context")
            if isinstance(raw, ScopedContext):
                parent_sc = raw
            elif isinstance(raw, dict):
                parent_sc = ScopedContext.model_validate(raw)

        raw_child = self.agent_config.sub_agent_config(subagent_type)
        child_dict = raw_child if isinstance(raw_child, dict) else {}
        child_cfg = SubAgentConfig.model_validate(child_dict)
        return child_cfg.with_effective_scoped_context(parent_sc)

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
        """Build the appropriate input object for the given node.

        The ``database`` context field is intentionally left unset: it denotes a physical
        database name, not a datasource. Each node is constructed with ``agent_config`` and
        routes through ``current_datasource``'s default database on its own, so stuffing the
        datasource name into ``database`` here would only mislabel the context.
        """
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.schemas.explore_agentic_node_models import ExploreNodeInput
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        if isinstance(node, ExploreAgenticNode):
            return ExploreNodeInput(
                user_message=prompt,
            )

        if isinstance(node, GenSQLAgenticNode):
            return GenSQLNodeInput(
                user_message=prompt,
            )

        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode

        if isinstance(node, AskMetricsAgenticNode):
            from datus.schemas.ask_metrics_agentic_node_models import AskMetricsNodeInput

            return AskMetricsNodeInput(
                user_message=prompt,
            )

        # Built-in system subagent input types
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode
        from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

        if isinstance(node, (GenTableAgenticNode, GenJobAgenticNode)):
            from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

            return SemanticNodeInput(
                user_message=prompt,
            )

        if isinstance(node, (GenSemanticModelAgenticNode, GenMetricsAgenticNode)):
            from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

            return SemanticNodeInput(
                user_message=prompt,
            )

        if isinstance(node, SqlSummaryAgenticNode):
            from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput

            return SqlSummaryNodeInput(
                user_message=prompt,
            )

        from datus.agent.node.gen_dashboard_agentic_node import GenDashboardAgenticNode

        if isinstance(node, GenDashboardAgenticNode):
            from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput

            return GenDashboardNodeInput(
                user_message=prompt,
            )

        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

        if isinstance(node, SchedulerAgenticNode):
            from datus.schemas.scheduler_agentic_node_models import SchedulerNodeInput

            return SchedulerNodeInput(
                user_message=prompt,
            )

        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        if isinstance(node, GenReportAgenticNode):
            from datus.schemas.gen_report_agentic_node_models import GenReportNodeInput

            return GenReportNodeInput(
                user_message=prompt,
            )

        from datus.agent.node.gen_visual_report_agentic_node import GenVisualReportAgenticNode

        if isinstance(node, GenVisualReportAgenticNode):
            from datus.schemas.gen_visual_report_models import GenVisualReportNodeInput

            return GenVisualReportNodeInput(
                user_message=prompt,
            )

        from datus.agent.node.gen_visual_dashboard_agentic_node import GenVisualDashboardAgenticNode

        if isinstance(node, GenVisualDashboardAgenticNode):
            from datus.schemas.gen_visual_dashboard_models import GenVisualDashboardNodeInput

            return GenVisualDashboardNodeInput(
                user_message=prompt,
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

    def _convert_to_func_result(self, output, *, session_id: Optional[str] = None) -> FuncToolResult:
        """Convert AgenticNode output to FuncToolResult.

        ``session_id`` is the subagent's session_id; when provided, it is
        injected into every result envelope — successful results carry it
        inline alongside the payload, and failure envelopes carry it under
        ``result`` so the parent LLM can resume the partial session
        (e.g. after a MaxTurnsExceeded interruption) by passing it back
        to a later task() call.
        """

        def _failure(error_msg: str) -> FuncToolResult:
            # Carry session_id under `result` so the parent can resume the
            # partial subagent session. ``result`` is None when no session
            # was created (e.g. errors raised before node construction).
            result_payload = {"session_id": session_id} if session_id else None
            return FuncToolResult(success=0, error=error_msg, result=result_payload)

        if not output or not isinstance(output, dict):
            return _failure("No result from subagent")

        # Check for explicit failure from subagent
        if output.get("success") is False:
            return _failure(output.get("error") or output.get("response") or output.get("content", "Subagent failed"))

        response = output.get("response", "")
        tokens = output.get("tokens_used", 0)

        def _wrap(d: Dict[str, Any]) -> FuncToolResult:
            if session_id:
                d["session_id"] = session_id
            return FuncToolResult(result=d)

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
            return _wrap(result_dict)

        # Inline SQL result: has 'sql' key
        sql = output.get("sql")
        if sql is not None:
            return _wrap(
                {
                    "sql": sql,
                    "response": response,
                    "tokens_used": tokens,
                }
            )

        # Semantic model result: has 'semantic_models' key
        semantic_models = output.get("semantic_models")
        if semantic_models is not None:
            return _wrap(
                {
                    "response": response,
                    "semantic_models": semantic_models,
                    "tokens_used": tokens,
                }
            )

        # SQL summary result: has 'sql_summary_file' key
        sql_summary_file = output.get("sql_summary_file")
        if sql_summary_file is not None:
            return _wrap(
                {
                    "response": response,
                    "sql_summary_file": sql_summary_file,
                    "tokens_used": tokens,
                }
            )

        # Report result: has 'report_result' key
        report_result = output.get("report_result")
        if report_result is not None:
            return _wrap(
                {
                    "response": response,
                    "report_result": report_result,
                    "tokens_used": tokens,
                }
            )

        if "markdown_report" in output:
            return _wrap(
                {
                    "response": response,
                    "markdown_report": output.get("markdown_report", response),
                    "tokens_used": tokens,
                }
            )

        # Skill creator result: has 'skill_path' key
        skill_path = output.get("skill_path")
        if skill_path is not None:
            return _wrap(
                {
                    "response": response,
                    "skill_name": output.get("skill_name", ""),
                    "skill_path": skill_path,
                    "tokens_used": tokens,
                }
            )

        # Dashboard result: has 'dashboard_result' key
        dashboard_result = output.get("dashboard_result")
        if dashboard_result is not None:
            return _wrap(
                {
                    "response": response,
                    "dashboard_result": dashboard_result,
                    "tokens_used": tokens,
                }
            )

        # Visual dashboard result (new artifact-based subagent). The
        # legacy ``dashboard_result`` envelope above is from
        # ``gen_dashboard_agentic_node.py``; the new
        # ``GenVisualDashboardNodeResult`` carries the documented fields
        # (``dashboard_slug``, ``app_jsx_path``, ``render_file_count``,
        # ``template_count``) flat at the top level. Without this branch
        # the conversion falls through to the generic envelope and drops
        # everything the parent LLM was told (via
        # ``BUILTIN_SUBAGENT_DESCRIPTIONS["gen_visual_dashboard"]``) to
        # expect. Match on the slug key's presence rather than tool name
        # so the branch also fires for legitimate model_dump output
        # where ``dashboard_slug`` is None (run failed before binding) —
        # preserving the explicit None is more honest than silently
        # collapsing into the generic envelope.
        if "dashboard_slug" in output:
            return _wrap(
                {
                    "response": response,
                    "dashboard_slug": output.get("dashboard_slug"),
                    "app_jsx_path": output.get("app_jsx_path"),
                    "render_file_count": output.get("render_file_count", 0),
                    "template_count": output.get("template_count", 0),
                    "tokens_used": tokens,
                }
            )

        # Scheduler result: has 'scheduler_result' key
        scheduler_result = output.get("scheduler_result")
        if scheduler_result is not None:
            return _wrap(
                {
                    "response": response,
                    "scheduler_result": scheduler_result,
                    "tokens_used": tokens,
                }
            )

        # Feedback result: has 'items_saved' key
        items_saved = output.get("items_saved")
        if items_saved is not None:
            return _wrap(
                {
                    "response": response,
                    "items_saved": items_saved,
                    "storage_summary": output.get("storage_summary"),
                    "tokens_used": tokens,
                }
            )

        # Generic format
        return _wrap(
            {
                "response": response or output.get("content", ""),
                "tokens_used": tokens,
            }
        )

    # ── description builder ────────────────────────────────────────────

    def _build_task_description(self) -> str:
        """Build a dynamic description for the task tool."""
        available = self._get_available_types()

        lines = [
            "Delegate work to a specialized subagent when the requested deliverable belongs to that "
            "subagent's owning workflow or platform. Task complexity is not the deciding factor: "
            "a simple scheduled job, dashboard, persisted table, semantic model, metric definition, "
            "or skill should still be handled by its specialized subagent. Use your own tools "
            "(list_tables, describe_table, execute_sql, etc.) for read-only answers, explanations, "
            "or lightweight investigations that do not create or update an artifact owned by another "
            "platform/workflow.",
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
                "- First classify the deliverable and its owning workflow/platform; delegate when it matches a subagent",
                "- For read-only answers, explanations, and lightweight investigations, handle directly with your own tools",
                '- For complex questions requiring deep exploration, call multiple task(type="explore") '
                "in PARALLEL, each with a direction-specific prompt (schema+sample, knowledge, file)",
                '- For quick single-direction lookups, call one task(type="explore") with a focused prompt',
                "- Each task() result — successful OR failed — includes a 'session_id' once the "
                "subagent has started running. On success the id sits at top level of 'result'; "
                "on failure (including MaxTurnsExceeded mid-run) it sits at 'result.session_id' "
                "alongside the error. To CONTINUE refining a prior answer or to RESUME a "
                "partial run with full prior context (its previous SQL, schema discoveries, "
                "reasoning), pass that session_id back as the task's 'session_id' argument and "
                "put ONLY the diff/clarification in 'prompt' — do not re-state the original "
                "problem. The session_id MUST be reused with the SAME 'type'.",
                "- Iterate-on-gen_sql example:",
                '    Turn 1: task(type="gen_sql", prompt="Top 10 customers by revenue last quarter",',
                '              description="customer revenue ranking")',
                '          → returns {sql, response, tokens_used, session_id: "gen_sql_session_ab12cd34"}',
                '    Turn 2: task(type="gen_sql", session_id="gen_sql_session_ab12cd34",',
                "              prompt=\"Exclude internal test accounts (account_type='test') and group",
                '                      monthly instead of the quarterly total",',
                '              description="refine: exclude tests, monthly granularity")',
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

                # If scoped_context is configured, datasource must match current datasource
                try:
                    sub_config = SubAgentConfig.model_validate(config)
                    if sub_config.has_scoped_context() and not sub_config.is_in_datasource(current_datasource):
                        continue
                except Exception as e:
                    logger.debug(f"Skipping invalid subagent config '{name}': {e}")
                    continue

                types.append(name)

        return types
