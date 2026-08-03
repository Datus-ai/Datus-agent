# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenMetricsAgenticNode implementation for metrics generation.

This module provides a specialized implementation of AgenticNode focused on
metrics generation with support for filesystem tools, generation tools,
hooks, and metricflow MCP server integration.
"""

import re
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.semantic_agentic_node_models import GenMetricsNodeResult, SemanticNodeInput
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.generation_tools import GenerationTools, MetricOutputBinding
from datus.tools.func_tool.metric_queryability import (
    query_backed_queryability_contracts,
    summarize_queryability_contracts,
)
from datus.tools.func_tool.sql_modeling_planner import (
    SqlModelingPlan,
    SqlModelingPlanTools,
    inspect_planned_semantic_sources,
)
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenMetricsAgenticNode(AgenticNode):
    """
    Metrics generation agentic node.

    This node provides specialized metrics generation capabilities with:
    - Enhanced system prompt with template variables
    - Filesystem tools for file operations
    - Generation tools for metrics generation
    - Hooks support for custom behavior
    - Metricflow MCP server integration
    - Session-based conversation management
    - Subject tree management (predefined or learning mode)
    """

    NODE_NAME = "gen_metrics"
    result_class = GenMetricsNodeResult

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        subject_tree: Optional[list] = None,
        scope: Optional[str] = None,
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the GenMetricsAgenticNode.

        Args:
            agent_config: Agent configuration
            execution_mode: Execution mode - "interactive" (default) or "workflow"
            subject_tree: Optional predefined subject tree categories
        """
        self.execution_mode = execution_mode
        self.subject_tree = subject_tree

        # Get max_turns from agentic_nodes configuration, default to 50
        self.max_turns = 50
        if agent_config and hasattr(agent_config, "agentic_nodes") and self.NODE_NAME in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[self.NODE_NAME]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 50)

        self.metrics_dir = str(agent_config.path_manager.semantic_model_path(agent_config.current_datasource))
        self.knowledge_base_dir = str(agent_config.path_manager.subject_dir)

        from datus.configuration.node_type import NodeType

        node_type = NodeType.TYPE_SEMANTIC

        # Call parent constructor first to set up node_config
        super().__init__(
            node_id=f"{self.NODE_NAME}_node",
            description=f"Metrics generation node: {self.NODE_NAME}",
            node_type=node_type,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
            scope=scope,
            is_subagent=is_subagent,
            session_id=session_id,
        )

        # Initialize metrics storage for context queries
        from datus.storage.metric.store import MetricRAG

        self.metrics_rag = MetricRAG(agent_config)

        # Setup tools
        self.db_func_tool = None
        self.semantic_discovery_tools = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.generation_tools: Optional[GenerationTools] = None
        from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState

        self.osi_target_state = OsiSemanticModelTargetState()
        self.osi_target_tools = None
        self.ask_user_tool = None
        self.hooks = None
        self.generation_evidence = GenerationEvidence()
        self.sql_modeling_plan: Optional[SqlModelingPlan] = None
        self.sql_modeling_tools: Optional[SqlModelingPlanTools] = None
        self.response_metric_output_bindings: Optional[List[Dict[str, str]]] = None
        self.setup_tools()

    def get_node_name(self) -> str:
        """
        Get the configured node name for this metrics generation node.

        Returns:
            The configured node name
        """
        return self.NODE_NAME

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Serialize metric writes with semantic-model authoring for this datasource."""
        from datus.agent.node.semantic_authoring import semantic_authoring_guard

        async with semantic_authoring_guard(self.agent_config):
            try:
                async for action in super().execute_stream(action_history_manager):
                    yield action
            finally:
                result = getattr(self, "result", None)
                if result is not None and not getattr(result, "success", False):
                    filesystem_tool = getattr(self, "filesystem_func_tool", None)
                    try:
                        if filesystem_tool is not None and filesystem_tool.rollback_failed_metric_authoring():
                            logger.info("Rolled back metric artifact after terminal generation failure")
                    except Exception:
                        logger.exception("Failed to roll back metric artifact after terminal generation failure")

    def setup_tools(self):
        """Setup tools for metrics generation."""
        if not self.agent_config:
            return

        self.tools = []

        self._setup_sql_modeling_tools()
        self._setup_osi_target_tools()
        from datus.agent.node.semantic_authoring import is_osi_authoring

        self._setup_db_tools(expose_tools=not is_osi_authoring(self.agent_config))
        if not is_osi_authoring(self.agent_config):
            self._setup_semantic_discovery_tools()
        self._setup_generation_tools()
        self._setup_filesystem_tools()
        self._setup_semantic_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.info(f"Setup {len(self.tools)} tools for {self.NODE_NAME}: {[tool.name for tool in self.tools]}")

    async def _before_stream(self, ctx: StreamRunContext) -> None:
        """Reset request-local authoring state before the first model turn."""
        await super()._before_stream(ctx)
        self.result = None
        self.generation_evidence.reset()
        self.osi_target_state.reset()
        self.sql_modeling_plan = None
        self.response_metric_output_bindings = None
        if self.sql_modeling_tools is not None:
            self.sql_modeling_tools.reset()

    def _setup_sql_modeling_tools(self) -> None:
        """Expose the single shared SQL preflight entry point."""
        from datus.agent.node.semantic_authoring import is_osi_authoring
        from datus.tools.func_tool import trans_to_function_tool

        self.sql_modeling_tools = SqlModelingPlanTools(
            agent_config=self.agent_config,
            sub_agent_name=self.get_node_name(),
            user_message_provider=lambda: str(getattr(self.input, "user_message", "") or ""),
            generation_evidence=self.generation_evidence,
            plan_consumer=self._accept_sql_modeling_plan,
            semantic_source_inspector=(
                None if is_osi_authoring(self.agent_config) else self._inspect_planned_semantic_sources
            ),
        )
        self.tools.append(trans_to_function_tool(self.sql_modeling_tools.prepare_sql_modeling_plan))

    def _accept_sql_modeling_plan(self, plan: Optional[SqlModelingPlan]) -> None:
        self.sql_modeling_plan = plan
        self._set_metric_queryability_contracts_from_input(getattr(self, "input", None))

    def _inspect_planned_semantic_sources(self, plan: SqlModelingPlan) -> Dict[str, Any]:
        """Batch-inspect physical SQL sources for MetricFlow authoring."""
        self.sql_modeling_plan = plan
        return inspect_planned_semantic_sources(plan, self.semantic_discovery_tools)

    def _ensure_bash_tool_in_tools(self) -> None:
        """Keep OSI metric authoring on its metrics-only filesystem surface."""
        from datus.agent.node.semantic_authoring import is_osi_authoring

        if is_osi_authoring(self.agent_config):
            return
        super()._ensure_bash_tool_in_tools()

    def _make_filesystem_tool(self, **kwargs):
        from datus.agent.node.semantic_authoring import resolve_authoring_format
        from datus.configuration.inherited_memory_overrides import get_inherited_memory
        from datus.tools.func_tool.metric_filesystem_tools import MetricFilesystemFuncTool

        root_path = kwargs.pop("root_path", None) or self._resolve_workspace_root()
        datus_home = kwargs.pop("datus_home", None)
        if datus_home is None and self.agent_config is not None:
            path_manager = getattr(self.agent_config, "path_manager", None)
            if path_manager is not None:
                try:
                    datus_home = str(path_manager.datus_home)
                except Exception:
                    datus_home = None
        strict = kwargs.pop("strict", None)
        if strict is None:
            strict = self._resolve_filesystem_strict()
        current_node = kwargs.pop("current_node", None) or self.get_node_name()
        inherited_memory_node = kwargs.pop("inherited_memory_node", None)
        if inherited_memory_node is None:
            inherited_memory_node = get_inherited_memory(current_node)
        session_data_dir = kwargs.pop("session_data_dir", None) or self._resolve_session_data_dir()
        mutation_callback = kwargs.pop(
            "mutation_callback",
            self.generation_evidence.record_artifact_mutation,
        )
        return MetricFilesystemFuncTool(
            root_path=root_path,
            current_node=current_node,
            datus_home=datus_home,
            strict=strict,
            inherited_memory_node=inherited_memory_node,
            session_data_dir=session_data_dir,
            mutation_callback=mutation_callback,
            authoring_format=resolve_authoring_format(self.agent_config),
            osi_target_state=self.osi_target_state,
            generation_evidence=self.generation_evidence,
            **kwargs,
        )

    def _setup_filesystem_tools(self):
        """Setup filesystem tools."""
        try:
            self.filesystem_func_tool = self._make_filesystem_tool()

            filesystem_tools = [
                tool for tool in self.filesystem_func_tool.available_tools() if tool.name != "delete_file"
            ]
            self.tools.extend(filesystem_tools)
            logger.debug("Added filesystem tools: %s", [tool.name for tool in filesystem_tools])
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _setup_generation_tools(self):
        """Setup generation tools."""
        try:
            from datus.agent.node.semantic_authoring import resolve_authoring_format
            from datus.tools.func_tool import trans_to_function_tool

            authoring_format = resolve_authoring_format(self.agent_config)
            self.generation_tools = GenerationTools(
                self.agent_config,
                generation_evidence=self.generation_evidence,
                authoring_format=authoring_format,
                osi_target_state=self.osi_target_state,
                require_bound_osi_target=authoring_format == "osi",
                sql_modeling_plan_required=self.sql_modeling_tools.request_contains_sql,
            )

            if authoring_format != "osi":
                self.tools.append(trans_to_function_tool(self.generation_tools.check_semantic_object_exists))
            self.tools.append(trans_to_function_tool(self.publish_metrics))
            if authoring_format != "osi":
                self.tools.append(trans_to_function_tool(self.generation_tools.publish_semantic_model))
            logger.debug("Added metric generation tools for authoring_format=%s", authoring_format)

        except Exception as e:
            logger.error(f"Failed to setup generation tools: {e}")

    def _setup_osi_target_tools(self) -> None:
        """Expose live inventory and exact binding without touching DB or RAG."""
        from datus.agent.node.semantic_authoring import is_osi_authoring

        if not is_osi_authoring(self.agent_config):
            return
        from datus.tools.func_tool import OsiSemanticModelTargetTools, trans_to_function_tool

        self.osi_target_tools = OsiSemanticModelTargetTools(
            self.agent_config,
            target_state=self.osi_target_state,
            generation_evidence=self.generation_evidence,
        )
        self.tools.extend(
            [
                trans_to_function_tool(self.osi_target_tools.list_existing_osi_semantic_models),
                trans_to_function_tool(self.osi_target_tools.bind_osi_semantic_model_target),
            ]
        )

    def _setup_skill_func_tools(self) -> None:
        """Default the optional skill set from the active authoring format."""
        from datus.agent.node.semantic_authoring import default_optional_skills

        if self.node_config.get("skills") is None:
            self.node_config["skills"] = default_optional_skills(self.agent_config, self.NODE_NAME)
        super()._setup_skill_func_tools()

    def _get_required_skills(self) -> list:
        """Host-inject the metric authoring specification skill."""
        from datus.agent.node.semantic_authoring import required_authoring_skills

        patterns = required_authoring_skills(self.agent_config, self.NODE_NAME)
        return [pattern.strip() for pattern in patterns.split(",") if pattern.strip()]

    def _setup_semantic_tools(self):
        """Setup semantic tools for metrics querying and exploration."""
        try:
            from datus.agent.node.semantic_authoring import resolve_semantic_adapter_type
            from datus.tools.func_tool.semantic_tools import SemanticTools

            adapter_type = resolve_semantic_adapter_type(self.agent_config)

            # Initialize semantic func tool
            self.semantic_tools = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name=self.NODE_NAME,
                adapter_type=adapter_type,
                generation_evidence=self.generation_evidence,
                runtime_db_context_provider=self._semantic_runtime_db_context,
                warehouse_dry_run_provider=self._warehouse_dry_run_compiled_sql,
            )

            # Add all available tools from semantic func tool
            semantic_tools = [
                tool
                for tool in self.semantic_tools.available_tools()
                if tool.name in {"get_dimensions", "query_metrics", "validate_semantic"}
            ]
            self.tools.extend(semantic_tools)

            tool_names = [tool.name for tool in semantic_tools]
            logger.info(f"Added semantic tools (adapter: {adapter_type}): {', '.join(tool_names)}")

        except Exception as e:
            logger.error(f"Failed to setup semantic tools: {e}")

    def _setup_db_tools(self, *, expose_tools: bool = True):
        """Setup the database helper, optionally without exposing LLM tools."""
        try:
            from datus.tools.func_tool import DBFuncTool

            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=self.NODE_NAME,
                read_only=not expose_tools,
            )
            if expose_tools:
                self.tools.extend(self.db_func_tool.available_tools())
                logger.debug("Added database tools from DBFuncTool")
            else:
                logger.debug("Initialized internal read-only database helper")
        except Exception as e:
            logger.error(f"Failed to setup database tools: {e}")

    def _warehouse_dry_run_compiled_sql(self, sql: str) -> Dict[str, Any]:
        """Validate adapter-compiled SQL with a read-only warehouse EXPLAIN."""
        if self.db_func_tool is None:
            return {"status": "failed", "error": "Database connection is unavailable."}
        runtime_context = self._semantic_runtime_db_context()
        result = self.db_func_tool.read_query(
            f"EXPLAIN {sql.rstrip(';')}",
            datasource=runtime_context.get("datasource", ""),
            database=runtime_context.get("database", ""),
        )
        if not getattr(result, "success", False):
            return {
                "status": "failed",
                "error": str(getattr(result, "error", None) or "Warehouse EXPLAIN failed."),
            }
        return {
            "status": "success",
            "datasource": runtime_context.get("datasource", ""),
            "database": runtime_context.get("database", ""),
        }

    def _setup_semantic_discovery_tools(self):
        """Setup read-only semantic discovery tools."""
        try:
            from datus.tools.func_tool.semantic_discovery_tools import SemanticDiscoveryTools

            self.semantic_discovery_tools = SemanticDiscoveryTools(
                self.db_func_tool,
                agent_config=self.agent_config,
                sub_agent_name=self.NODE_NAME,
                source_sql_provider=self._semantic_discovery_source_sql,
            )
            discovery_tools = self.semantic_discovery_tools.available_tools()
            self.tools.extend(discovery_tools)
            logger.debug(
                "Added semantic discovery tools: %s",
                [tool.name for tool in discovery_tools],
            )
        except Exception as e:
            logger.error(f"Failed to setup semantic discovery tools: {e}")

    def _semantic_discovery_source_sql(self) -> List[Dict[str, Any]]:
        """Expose exact request SQL captured by the shared preflight."""
        if self.sql_modeling_plan is None:
            return []
        return [
            {
                "name": source.source_sql_name,
                "question": source.question,
                "sql": source.sql,
            }
            for source in self.sql_modeling_plan.source_queries
        ]

    def _get_existing_subject_trees(self) -> list:
        """
        Query existing subject_tree values from metrics storage.

        Returns:
            List of unique subject_path values as strings (e.g., ["Finance/Revenue/Q1", ...])
        """
        try:
            # Check if storage is available
            if not getattr(self.metrics_rag, "storage", None):
                return []

            # Get all subject paths using the flat tree structure
            subject_paths = sorted(self.metrics_rag.storage.get_subject_tree_flat())
            logger.debug(f"Found {len(subject_paths)} unique metric subject_paths")
            return subject_paths

        except Exception as e:
            logger.error(f"Error getting existing metric subject_trees: {e}")
            return []

    def _prepare_template_context(self, user_input: SemanticNodeInput) -> dict:
        """
        Prepare template context variables for the metrics generation template.

        Args:
            user_input: User input

        Returns:
            Dictionary of template variables
        """
        from datus.utils.node_utils import build_datasource_prompt_context

        context = {}

        # Tool name lists for template display
        context["native_tools"] = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"
        context["mcp_tools"] = ", ".join(list(self.mcp_servers.keys())) if self.mcp_servers else "None"
        context["semantic_model_dir"] = self.metrics_dir
        context["knowledge_base_dir"] = self.knowledge_base_dir
        # Filesystem tool is rooted at project_root; full path required.
        context["kind_subdir"] = f"subject/semantic_models/{self.agent_config.current_datasource}"
        context["current_datasource"] = self.agent_config.current_datasource
        context["has_ask_user_tool"] = self.ask_user_tool is not None
        context.update(build_datasource_prompt_context(self.agent_config))

        from datus.agent.node.semantic_authoring import resolve_authoring_format

        context["authoring_format"] = resolve_authoring_format(self.agent_config)

        # Handle subject_tree context based on whether predefined or query from storage
        if self.subject_tree:
            # Predefined mode: use provided subject_tree
            context["has_subject_tree"] = True
            context["subject_tree"] = self.subject_tree
        else:
            # Learning mode: query existing subject_trees from vector store
            context["has_subject_tree"] = False
            context["existing_subject_trees"] = self._get_existing_subject_trees()

        logger.debug(f"Prepared template context: {context}")
        return context

    def _build_enhanced_message(
        self,
        user_input: SemanticNodeInput,
        extra_enhanced_parts: Optional[List[str]] = None,
    ) -> str:
        """Expose a structured caller hint without resolving it on the host."""
        from datus.agent.node.semantic_authoring import is_osi_authoring

        parts = list(extra_enhanced_parts or [])
        semantic_model_name = str(getattr(user_input, "semantic_model_name", "") or "").strip()
        semantic_model_file = str(getattr(user_input, "semantic_model_file", "") or "").strip()
        if is_osi_authoring(self.agent_config) and (semantic_model_name or semantic_model_file):
            hints = ["## OSI Semantic Model Selection Hint for This Turn"]
            if semantic_model_file:
                hints.append(f"- Requested semantic model file: `{semantic_model_file}`")
            if semantic_model_name:
                hints.append(f"- Requested semantic model name: `{semantic_model_name}`")
            hints.append(
                "Treat these as unverified hints: call `bind_osi_semantic_model_target` with the exact "
                "selectors, then inspect the live inventory if binding fails."
            )
            parts.append("\n".join(hints))
        return super()._build_enhanced_message(user_input, parts)

    def _system_prompt_snapshot_meta(self, prompt_version: Optional[str]) -> Dict[str, str]:
        """Invalidate snapshots created before semantic targets became request-scoped."""
        meta = super()._system_prompt_snapshot_meta(prompt_version)
        meta["semantic_target_scope"] = "agent_bound_v3"
        return meta

    def _get_system_prompt(
        self,
        prompt_version: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        """
        Get the system prompt for metrics generation using enhanced template context.

        Args:
            prompt_version: Optional prompt version override (ignored when the
                ``node_config`` / ``self.input`` already pin a version)
            template_context: Optional template context variables

        Returns:
            System prompt string loaded from the template
        """
        # Both authoring formats share one template; the format-specific spec
        # is injected as a required skill.
        template_name = f"{self.NODE_NAME}_system"
        version = (
            prompt_version or getattr(self.input, "prompt_version", None) or self.node_config.get("prompt_version")
        )

        try:
            # Prepare template variables
            template_vars = {
                "agent_config": self.agent_config,
            }

            # Add template context if provided
            if template_context:
                template_vars.update(template_context)

            # Use prompt manager to render the template
            from datus.prompts.prompt_manager import get_prompt_manager

            base_prompt = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name=template_name, version=version, **template_vars
            )
            return self._finalize_system_prompt(base_prompt)

        except FileNotFoundError as e:
            # Template not found - throw DatusException
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_TEMPLATE_NOT_FOUND,
                message_args={"template_name": template_name, "version": version or "latest"},
            ) from e
        except Exception as e:
            # Other template errors - wrap in DatusException
            logger.error(f"Template loading error for '{template_name}': {e}")
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Template loading failed for '{template_name}': {str(e)}"},
            ) from e

    def _build_template_context(self, ctx: StreamRunContext) -> Optional[dict]:
        self._set_metric_queryability_contracts_from_input(ctx.user_input)
        return self._prepare_template_context(ctx.user_input)

    def _build_success_result(self, ctx: StreamRunContext) -> GenMetricsNodeResult:
        sql_request = (
            self.sql_modeling_tools.require_plan_for_sql_request() if self.sql_modeling_tools is not None else False
        )
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            raw_output = ctx.last_successful_output.get("raw_output", "")
            if isinstance(raw_output, dict) or raw_output:
                response_content = raw_output
            else:
                response_content = str(ctx.last_successful_output)

        semantic_model_files, metric_file, status, blocker_code, skip_reason, extracted_output = (
            self._extract_metric_and_output_from_response({"content": response_content})
        )
        metric_output_bindings = getattr(self, "response_metric_output_bindings", None)
        if extracted_output:
            response_content = extracted_output

        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        blocker_code = blocker_code.strip().lower() if isinstance(blocker_code, str) else blocker_code
        skip_reason = skip_reason.strip().lower() if isinstance(skip_reason, str) else skip_reason
        if sql_request and status is None:
            raise RuntimeError(
                "SQL-backed metric generation must return a structured final response "
                "with status and the generated metric_file or an explicit supported skip."
            )
        if status is not None and status not in {"generated", "skipped", "blocked"}:
            raise RuntimeError(f"Unsupported metric generation status: {status!r}")
        if skip_reason is not None and skip_reason != "not_a_metric":
            raise RuntimeError(f"Unsupported metric generation skip_reason: {skip_reason!r}")
        if skip_reason is not None and status != "skipped":
            raise RuntimeError("skip_reason is only valid when status='skipped'.")
        if status == "skipped" and self.generation_evidence.required_metric_output_ids:
            raise RuntimeError(
                "SQL planning found required metric outputs, so generation cannot return status='skipped'."
            )
        normalized_status = status
        if normalized_status == "blocked":
            if metric_file or self.osi_target_state.touched_metric_names:
                raise RuntimeError(
                    "status='blocked' is only valid before metric authoring and with metric_file set to null."
                )
            if blocker_code not in {
                "semantic_model_required",
                "semantic_model_selection_required",
                "semantic_model_target_invalid",
            }:
                raise RuntimeError("status='blocked' requires a supported semantic-model blocker_code.")
            if self.osi_target_tools is None:
                raise RuntimeError("status='blocked' is only supported for OSI metric generation.")
            final_metric_file = None
        else:
            self._finalize_metric_generation(
                semantic_model_files=semantic_model_files,
                metric_file=metric_file,
                metric_output_bindings=metric_output_bindings,
                status=normalized_status,
                skip_reason=skip_reason,
            )
            target_state = getattr(self, "osi_target_state", None)
            bound = target_state.bound if target_state is not None else None
            final_metric_file = (
                str(bound["semantic_model_file"])
                if bound is not None and target_state.touched_metric_names
                else metric_file
            )
            if normalized_status is None:
                normalized_status = "generated" if final_metric_file else "skipped"

        tokens_used = 0
        if self.execution_mode == "interactive":
            tokens_used = self._extract_total_tokens(ctx.action_history_manager.get_actions())

        return GenMetricsNodeResult(
            success=normalized_status != "blocked",
            error=response_content if normalized_status == "blocked" else None,
            response=response_content,
            semantic_models=[final_metric_file] if final_metric_file else [],
            tokens_used=int(tokens_used),
            status=normalized_status,
            blocker_code=blocker_code if normalized_status == "blocked" else None,
            skip_reason=skip_reason if normalized_status == "skipped" else None,
        )

    @staticmethod
    def _tool_succeeded(result: Any) -> bool:
        if isinstance(result, dict):
            return result.get("success", 1) in (1, True)
        if hasattr(result, "success"):
            return result.success in (1, True)
        return False

    @staticmethod
    def _tool_error(result: Any) -> str:
        if isinstance(result, dict):
            return str(result.get("error") or result.get("result") or "unknown error")
        return str(getattr(result, "error", None) or getattr(result, "result", None) or "unknown error")

    def _resolve_metric_artifact_path(self, path: Optional[str], kind: str) -> str:
        if not path:
            return ""

        from datus.cli.generation_hooks import resolve_kb_sandbox_path

        resolved_path = resolve_kb_sandbox_path(path, kind, self.knowledge_base_dir)
        if not resolved_path:
            raise RuntimeError(f"Metric generation reported {kind}_file outside Knowledge Base sandbox: {path!r}")
        return resolved_path

    @staticmethod
    def _tool_result_payload(result: Any) -> Dict[str, Any]:
        payload = result.get("result") if isinstance(result, dict) else getattr(result, "result", None)
        return payload if isinstance(payload, dict) else {}

    @classmethod
    def _warehouse_dry_run_succeeded(cls, result: Any) -> bool:
        metadata = cls._tool_result_payload(result).get("metadata")
        warehouse = metadata.get("warehouse_dry_run") if isinstance(metadata, dict) else None
        return isinstance(warehouse, dict) and warehouse.get("status") == "success"

    @staticmethod
    def _normalized_query_name(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")

    @classmethod
    def _matching_dimension_name(cls, available: List[str], candidates: List[str]) -> Optional[str]:
        normalized_candidates = {
            cls._normalized_query_name(candidate) for candidate in candidates if cls._normalized_query_name(candidate)
        }
        for name in available:
            normalized = cls._normalized_query_name(name)
            leaf = cls._normalized_query_name(str(name).split("__")[-1])
            if normalized in normalized_candidates or leaf in normalized_candidates:
                return name
        return None

    def _metric_dimension_capabilities(self, metric_name: str) -> tuple[List[str], Optional[str]]:
        result = self.semantic_tools.get_dimensions(metric_name=metric_name)
        if not self._tool_succeeded(result):
            raise RuntimeError(
                f"get_dimensions failed for generated metric `{metric_name}`: {self._tool_error(result)}"
            )
        payload = self._tool_result_payload(result)
        names = [
            str(item.get("name")).strip()
            for item in payload.get("items") or []
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        time_dimension = str(extra.get("time_dimension") or "").strip() or None
        return names, time_dimension

    def _queryability_dry_run_args(
        self,
        contract: Dict[str, Any],
        metric_names: List[str],
        dimension_cache: Dict[str, tuple[List[str], Optional[str]]],
    ) -> tuple[List[str], List[str], Optional[str]]:
        metric_hints = {str(name).strip() for name in contract.get("metric_hints") or [] if str(name).strip()}
        contract_metrics = [name for name in metric_names if not metric_hints or name in metric_hints]
        if not contract_metrics:
            return [], [], None

        capabilities = []
        for name in contract_metrics:
            if name not in dimension_cache:
                dimension_cache[name] = self._metric_dimension_capabilities(name)
            capabilities.append(dimension_cache[name])
        common_dimensions = [
            name for name in capabilities[0][0] if all(name in available for available, _ in capabilities[1:])
        ]
        time_dimensions = [time_dimension for _, time_dimension in capabilities]
        common_time_dimensions = {time_dimension for time_dimension in time_dimensions if time_dimension}
        primary_time_dimension = (
            next(iter(common_time_dimensions)) if all(time_dimensions) and len(common_time_dimensions) == 1 else None
        )

        resolved_dimensions: List[str] = []
        time_granularity: Optional[str] = None
        expr_hints = [hint for hint in contract.get("dimension_expr_hints") or [] if isinstance(hint, dict)]
        time_hints = [hint for hint in contract.get("time_group_hints") or [] if isinstance(hint, dict)]
        for dimension_hint in contract.get("dimension_hints") or []:
            if not isinstance(dimension_hint, str) or not dimension_hint.strip():
                continue
            matching_time_hint = next(
                (
                    hint
                    for hint in time_hints
                    if self._normalized_query_name(dimension_hint)
                    in {
                        self._normalized_query_name(hint.get("alias")),
                        self._normalized_query_name(hint.get("base_expr")),
                    }
                ),
                None,
            )
            candidates = [dimension_hint]
            for expr_hint in expr_hints:
                if self._normalized_query_name(dimension_hint) not in {
                    self._normalized_query_name(expr_hint.get("alias")),
                    self._normalized_query_name(expr_hint.get("expr")),
                }:
                    continue
                candidates.extend([expr_hint.get("expr", ""), expr_hint.get("column", "")])
            if matching_time_hint:
                candidates.append(str(matching_time_hint.get("base_expr") or ""))
                time_granularity = str(matching_time_hint.get("grain") or "").strip() or time_granularity
            resolved = self._matching_dimension_name(common_dimensions, candidates)
            if resolved is None and matching_time_hint:
                resolved = primary_time_dimension
            if resolved is None:
                raise DatusException(
                    ErrorCode.COMMON_VALIDATION_FAILED,
                    "Cannot map source SQL GROUP BY dimension "
                    f"`{dimension_hint}` to a queryable semantic dimension for "
                    f"{', '.join(contract_metrics)}.",
                )
            if resolved not in resolved_dimensions:
                resolved_dimensions.append(resolved)
        return contract_metrics, resolved_dimensions, time_granularity

    def _ensure_metric_dry_runs(self, metric_names: List[str]) -> None:
        """Run fresh contract and warehouse dry-runs before every publication."""
        query_metrics = getattr(getattr(self, "semantic_tools", None), "query_metrics", None)
        if not callable(query_metrics):
            raise RuntimeError("Metric generation produced metrics, but query_metrics is unavailable.")

        dimension_cache: Dict[str, tuple[List[str], Optional[str]]] = {}
        covered_metrics = set()
        for contract in self.generation_evidence.metric_queryability_contracts:
            contract_metrics, dimensions, time_granularity = self._queryability_dry_run_args(
                contract,
                metric_names,
                dimension_cache,
            )
            if not contract_metrics:
                continue
            kwargs: Dict[str, Any] = {
                "metrics": contract_metrics,
                "dimensions": dimensions,
                "dry_run": True,
            }
            if time_granularity:
                kwargs["time_granularity"] = time_granularity
            result = query_metrics(**kwargs)
            self.generation_evidence.record_metric_dry_run(
                contract_metrics,
                result,
                dimensions=dimensions,
                time_granularity=time_granularity,
            )
            if not self._tool_succeeded(result):
                raise RuntimeError(
                    "query_metrics(dry_run=True) failed for source SQL GROUP BY contract "
                    f"{contract.get('source') or 'unknown'}: {self._tool_error(result)}"
                )
            if not self._warehouse_dry_run_succeeded(result):
                raise RuntimeError(
                    "query_metrics(dry_run=True) compiled the source SQL GROUP BY contract "
                    f"{contract.get('source') or 'unknown'}, but did not complete a warehouse dry-run."
                )
            covered_metrics.update(contract_metrics)

        uncovered = [name for name in metric_names if name not in covered_metrics]
        if uncovered:
            result = query_metrics(metrics=uncovered, dry_run=True)
            self.generation_evidence.record_metric_dry_run(uncovered, result)
            if not self._tool_succeeded(result):
                raise RuntimeError(
                    "query_metrics(dry_run=True) failed for generated metric(s) "
                    f"{', '.join(uncovered)}: {self._tool_error(result)}"
                )
            if not self._warehouse_dry_run_succeeded(result):
                raise RuntimeError(
                    "query_metrics(dry_run=True) compiled generated metric(s) "
                    f"{', '.join(uncovered)}, but did not complete a warehouse dry-run."
                )

        missing_contracts = self.generation_evidence.missing_queryability_contracts(metric_names)
        if missing_contracts:
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                "query_metrics(dry_run=True) must pass with every source SQL GROUP BY dimension "
                "before publishing metrics. "
                f"Missing: {summarize_queryability_contracts(missing_contracts)}",
            )

    def publish_metrics(
        self,
        metric_file: str,
        metric_output_bindings: Optional[List[MetricOutputBinding]] = None,
    ):
        """Run deterministic queryability checks, then sync metrics to storage."""
        if not self.generation_tools:
            raise RuntimeError("Metric generation tools are unavailable.")
        self._set_metric_queryability_contracts_from_input(getattr(self, "input", None))
        bindings = [
            binding.model_dump() if isinstance(binding, MetricOutputBinding) else dict(binding)
            for binding in metric_output_bindings or []
        ]
        self.generation_evidence.bind_metric_output_names(bindings)

        abs_metric_file = self._resolve_metric_artifact_path(metric_file, "metric")
        if self.osi_target_state.touched_metric_names:
            current_names = self.generation_tools.extract_osi_metric_names(abs_metric_file)
            metric_names, _ = self.osi_target_state.partition_touched_metrics(current_names)
        else:
            metric_names = self.generation_tools._extract_metric_names_from_file(abs_metric_file)
            metric_definitions = self.generation_tools._extract_metric_definitions_from_file(abs_metric_file)
            metric_names = self.generation_tools._metric_names_requiring_dry_run(
                metric_names,
                metric_definitions,
                self.generation_evidence.metric_sqls,
            )
        if metric_names:
            try:
                self._ensure_metric_dry_runs(metric_names)
            except Exception as exc:
                error_message = str(exc) or type(exc).__name__
                logger.exception(f"publish_metrics preflight failed for metrics={metric_names}: {error_message}")
                return FuncToolResult(
                    success=0,
                    error=error_message,
                    result={
                        "code": "metric_publish_preflight_failed",
                        "stage": "query_metrics_dry_run",
                        "metric_file": metric_file,
                        "metrics": metric_names,
                    },
                )

        publish_kwargs: Dict[str, Any] = {"metric_file": abs_metric_file}
        if metric_output_bindings is not None:
            publish_kwargs["metric_output_bindings"] = metric_output_bindings
        return self.generation_tools.publish_metrics(**publish_kwargs)

    def _set_metric_queryability_contracts_from_input(self, user_input: Optional[SemanticNodeInput]) -> None:
        if not user_input:
            return
        candidate_plan = self.sql_modeling_plan.candidate_plan if self.sql_modeling_plan is not None else {}
        contracts = [
            dict(contract)
            for contract in candidate_plan.get("queryability_contracts") or []
            if isinstance(contract, dict)
        ]
        if not contracts:
            # Compatibility for request plans created before queryability
            # contracts became a first-class planner output.
            contracts = query_backed_queryability_contracts(candidate_plan)
        metric_aliases = self._metric_aliases_from_candidate_plan(candidate_plan)
        self.generation_evidence.set_metric_queryability_contracts(contracts, metric_aliases=metric_aliases)

    @staticmethod
    def _metric_aliases_from_candidate_plan(candidate_plan: Dict[str, Any]) -> Dict[str, str]:
        aliases: Dict[str, str] = {}

        def add(alias: Any, canonical: Any) -> None:
            if not isinstance(alias, str) or not isinstance(canonical, str):
                return
            alias = alias.strip()
            canonical = canonical.strip()
            if alias and canonical and alias != canonical:
                aliases[alias] = canonical

        for item in candidate_plan.get("metric_aliases") or []:
            if not isinstance(item, dict):
                continue
            canonical = item.get("canonical_name") or item.get("name")
            add(item.get("source_alias"), canonical)
            add(item.get("candidate_name"), canonical)

        for key in (
            "direct_metric_candidates",
            "derived_metric_candidates",
            "metric_candidates",
            "identity_metric_references",
        ):
            for candidate in candidate_plan.get(key) or []:
                if not isinstance(candidate, dict):
                    continue
                canonical = candidate.get("name")
                add(candidate.get("source_alias"), canonical)
                for source_alias in candidate.get("source_aliases") or []:
                    add(source_alias, canonical)
                for candidate_name in candidate.get("candidate_names") or []:
                    add(candidate_name, canonical)
                for mapping in candidate.get("source_alias_mappings") or []:
                    if not isinstance(mapping, dict):
                        continue
                    add(mapping.get("source_alias"), canonical)
                    add(mapping.get("candidate_name"), canonical)
        return aliases

    def _finalize_metric_generation(
        self,
        semantic_model_files: Optional[List[str] | str],
        metric_file: Optional[str],
        status: Optional[str],
        skip_reason: Optional[str] = None,
        metric_output_bindings: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Ensure generated metric artifacts are published without relying on one LLM tool call."""
        from datus.agent.node.semantic_authoring import is_osi_authoring

        if is_osi_authoring(self.agent_config):
            self._finalize_osi_metric_generation(
                semantic_model_files=semantic_model_files,
                metric_file=metric_file,
                metric_output_bindings=metric_output_bindings,
                status=status,
                skip_reason=skip_reason,
            )
            return
        del semantic_model_files

        normalized_status = status.strip().lower() if isinstance(status, str) else status

        if normalized_status == "skipped":
            if metric_file:
                raise RuntimeError(
                    "Metric generation returned status='skipped' with a non-null metric_file. "
                    "Skipped responses must set metric_file to null; generated metric files must be published."
                )
            return

        if self.generation_evidence.metric_kb_sync_passed:
            return

        if normalized_status and not metric_file:
            raise RuntimeError(
                f"Metric generation returned status='{normalized_status}' without a metric_file. "
                "Non-skipped metric responses must include metric_file or call publish_metrics."
            )

        if not metric_file:
            return

        if not self.generation_tools:
            raise RuntimeError("Metric generation produced a metric_file, but generation tools are unavailable.")
        self._set_metric_queryability_contracts_from_input(getattr(self, "input", None))
        self.generation_evidence.bind_metric_output_names(metric_output_bindings)

        if not self.generation_evidence.validation_passed:
            if not getattr(self, "semantic_tools", None):
                raise RuntimeError("Metric generation produced a metric_file, but validate_semantic is unavailable.")
            validation_result = self.semantic_tools.validate_semantic()
            self.generation_evidence.record_validation_result(validation_result)
            if not self._tool_succeeded(validation_result):
                raise RuntimeError(
                    f"validate_semantic failed before publishing metrics: {self._tool_error(validation_result)}"
                )

        abs_metric_file = self._resolve_metric_artifact_path(metric_file, "metric")
        preflight_error = self.generation_tools._validate_metric_file_has_blocks(abs_metric_file)
        if preflight_error:
            raise RuntimeError(preflight_error)

        publish_kwargs: Dict[str, Any] = {"metric_file": abs_metric_file}
        if self.generation_evidence.required_metric_output_ids:
            publish_kwargs["metric_output_bindings"] = metric_output_bindings
        publish_result = self.publish_metrics(**publish_kwargs)
        if not self._tool_succeeded(publish_result):
            raise RuntimeError(f"Metric KB sync failed: {self._tool_error(publish_result)}")

    def _finalize_osi_metric_generation(
        self,
        semantic_model_files: Optional[List[str] | str],
        metric_file: Optional[str],
        status: Optional[str],
        skip_reason: Optional[str],
        metric_output_bindings: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Validate and publish only the exact target bound during this request."""
        del semantic_model_files
        normalized_status = status.strip().lower() if isinstance(status, str) else status
        bound = self.osi_target_state.bound
        touched_metric_names = list(self.osi_target_state.touched_metric_names)

        if normalized_status == "skipped":
            if metric_file or touched_metric_names:
                raise RuntimeError(
                    "Metric generation cannot return status='skipped' after changing metrics "
                    "or with a non-null metric_file."
                )
            if skip_reason != "not_a_metric":
                raise RuntimeError(
                    "OSI status='skipped' is only valid for a non-metric request with skip_reason='not_a_metric'. "
                    "Existing requested metrics must be idempotently upserted and published."
                )
            return
        if self.osi_target_state.last_error_code:
            raise RuntimeError(
                "The OSI target selection is unresolved after a failed bind. "
                "Bind a valid target again or return a matching blocked result."
            )
        if bound is None:
            raise RuntimeError(
                "OSI metric generation must bind an existing semantic model or return a supported blocked result."
            )
        if not touched_metric_names:
            raise RuntimeError(
                "No metrics were changed or scheduled for idempotent publish in the bound OSI "
                "semantic model. Pass requested changes through upsert_osi_metrics or delete_osi_metrics."
            )

        abs_metric_file = str(bound["absolute_path"])
        try:
            self.osi_target_state.require_current_revision(abs_metric_file)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        artifact_validated = self.generation_evidence.semantic_artifact_validation_passed(
            bound["semantic_model_name"],
            abs_metric_file,
        )
        if self.generation_evidence.has_metric_kb_sync(touched_metric_names) and artifact_validated:
            return

        if metric_file:
            try:
                reported = self._resolve_metric_artifact_path(metric_file, "metric")
            except RuntimeError:
                reported = ""
            if reported and reported != abs_metric_file:
                logger.warning(
                    "Ignoring LLM-reported metric_file %s; publishing bound target %s",
                    metric_file,
                    bound["semantic_model_file"],
                )

        if not self.generation_tools:
            raise RuntimeError("Metrics were changed in a bound OSI target, but generation tools are unavailable.")
        self._set_metric_queryability_contracts_from_input(getattr(self, "input", None))
        self.generation_evidence.bind_metric_output_names(metric_output_bindings)

        if not getattr(self, "semantic_tools", None):
            raise RuntimeError("Metrics were changed in a bound OSI target, but validate_semantic is unavailable.")
        model_names = self.generation_tools.extract_osi_model_names(abs_metric_file)
        if model_names != [bound["semantic_model_name"]]:
            raise RuntimeError("The bound OSI artifact no longer declares the selected semantic model.")
        _, absent_metric_names = self.osi_target_state.partition_touched_metrics(
            self.generation_tools.extract_osi_metric_names(abs_metric_file)
        )
        has_absent_touched_metric = bool(absent_metric_names)
        if not artifact_validated:
            validation_kwargs = {"semantic_model_name": bound["semantic_model_name"]}
            if has_absent_touched_metric:
                validation_kwargs["scope"] = "semantic_model"
            validation_result = self.semantic_tools.validate_semantic(**validation_kwargs)
            self.generation_evidence.record_validation_result(validation_result)
            if not self._tool_succeeded(validation_result):
                raise RuntimeError(
                    f"validate_semantic failed before publishing OSI metrics: {self._tool_error(validation_result)}"
                )

        publish_kwargs: Dict[str, Any] = {"metric_file": abs_metric_file}
        if self.generation_evidence.required_metric_output_ids:
            publish_kwargs["metric_output_bindings"] = metric_output_bindings
        publish_result = self.publish_metrics(**publish_kwargs)
        if not self._tool_succeeded(publish_result):
            raise RuntimeError(f"OSI metric KB sync failed: {self._tool_error(publish_result)}")

    def _extract_metric_and_output_from_response(
        self, output: dict
    ) -> tuple[
        Optional[List[str]],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
    ]:
        """
        Extract semantic model files, metric file, status and formatted output from model response.

        Per prompt template requirements, LLM should return JSON format:
        {"semantic_model_files": ["path.yml"], "metric_file": "path.yml",
         "metric_output_bindings": [{"output_id": "...", "metric_name": "..."}],
         "status": "generated" | "skipped" | "blocked", "blocker_code": null,
         "skip_reason": null | "not_a_metric",
         "output": "markdown text"}

        ``status`` is optional for backward compatibility; absent values are treated as ``"generated"``.

        Args:
            output: Output dictionary from model generation

        Returns:
            Tuple of (semantic_model_files, metric_file, status, blocker_code,
            skip_reason, output_string). Parsed bindings are retained on the
            request-scoped node for the host publish fallback.
        """
        self.response_metric_output_bindings = None
        try:
            from datus.utils.json_utils import strip_json_str

            content = output.get("content", "")
            logger.info(f"extract_metric_and_output_from_response: {content} (type: {type(content)})")

            # Case 1: content is already a dict (most common)
            if isinstance(content, dict):
                output_text = content.get("output")
                semantic_model_files = self._normalize_semantic_model_files(content)
                metric_file = content.get("metric_file")
                metric_output_bindings = self._normalize_metric_output_bindings(content)
                self.response_metric_output_bindings = metric_output_bindings
                status = content.get("status")
                normalized_status = status.strip().lower() if isinstance(status, str) else None
                blocker_code = content.get("blocker_code")
                skip_reason = content.get("skip_reason")

                if (metric_file and isinstance(metric_file, str)) or normalized_status:
                    logger.debug(
                        f"Extracted from dict: semantic_model_files={semantic_model_files}, "
                        f"metric_file={metric_file}, status={normalized_status}"
                    )
                    return (
                        semantic_model_files,
                        metric_file,
                        normalized_status,
                        blocker_code,
                        skip_reason,
                        output_text,
                    )

                logger.warning(f"Dict format but missing expected keys or invalid format: {content.keys()}")

            # Case 2: content is a JSON string (possibly wrapped in markdown code blocks)
            elif isinstance(content, str) and content.strip():
                # Use strip_json_str to handle markdown code blocks and extract JSON
                cleaned_json = strip_json_str(content)
                if cleaned_json:
                    try:
                        import json_repair

                        parsed = json_repair.loads(cleaned_json)
                        if isinstance(parsed, dict):
                            output_text = parsed.get("output")
                            semantic_model_files = self._normalize_semantic_model_files(parsed)
                            metric_file = parsed.get("metric_file")
                            metric_output_bindings = self._normalize_metric_output_bindings(parsed)
                            self.response_metric_output_bindings = metric_output_bindings
                            status = parsed.get("status")
                            normalized_status = status.strip().lower() if isinstance(status, str) else None
                            blocker_code = parsed.get("blocker_code")
                            skip_reason = parsed.get("skip_reason")

                            if (metric_file and isinstance(metric_file, str)) or normalized_status:
                                logger.debug(
                                    f"Extracted from JSON string: "
                                    f"semantic_model_files={semantic_model_files}, "
                                    f"metric_file={metric_file}, status={normalized_status}"
                                )
                                return (
                                    semantic_model_files,
                                    metric_file,
                                    normalized_status,
                                    blocker_code,
                                    skip_reason,
                                    output_text,
                                )

                            logger.warning(f"Parsed JSON but missing expected keys or invalid format: {parsed.keys()}")
                    except Exception as e:
                        logger.warning(f"Failed to parse cleaned JSON: {e}. Cleaned content: {cleaned_json[:200]}")

            logger.warning(f"Could not extract metric_file from response. Content type: {type(content)}")
            return None, None, None, None, None, None

        except Exception as e:
            logger.error(f"Unexpected error extracting metric_file: {e}", exc_info=True)
            return None, None, None, None, None, None

    @staticmethod
    def _normalize_metric_output_bindings(content: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        raw_bindings = content.get("metric_output_bindings")
        if not isinstance(raw_bindings, list):
            return None
        return raw_bindings

    @staticmethod
    def _normalize_semantic_model_files(content: Dict[str, Any]) -> List[str]:
        raw_files = content.get("semantic_model_files")
        if raw_files is None:
            raw_files = content.get("semantic_model_file")
        if isinstance(raw_files, str):
            candidates = [raw_files]
        elif isinstance(raw_files, list):
            candidates = raw_files
        else:
            candidates = []
        result: List[str] = []
        for item in candidates:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if value and value not in result:
                result.append(value)
        return result
