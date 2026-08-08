# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenSemanticModelAgenticNode implementation for semantic model generation.

This module provides a specialized implementation of AgenticNode focused on
semantic model generation with support for filesystem tools, generation tools,
database tools, hooks, and metricflow MCP server integration.
"""

from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput, SemanticNodeResult
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.tools.func_tool.semantic_discovery_tools import SemanticDiscoveryTools
from datus.tools.func_tool.sql_modeling_planner import (
    SqlModelingPlan,
    SqlModelingPlanTools,
    inspect_planned_semantic_sources,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenSemanticModelAgenticNode(AgenticNode):
    """
    Semantic model generation agentic node.

    This node provides specialized semantic model generation capabilities with:
    - Enhanced system prompt with template variables
    - Database tools for schema exploration
    - Filesystem tools for file operations
    - Generation tools for model generation
    - Hooks support for custom behavior
    - Metricflow MCP server integration
    - Session-based conversation management
    """

    NODE_NAME = "gen_semantic_model"
    result_class = SemanticNodeResult

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        scope: Optional[str] = None,
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the GenSemanticModelAgenticNode.

        Args:
            agent_config: Agent configuration
            execution_mode: Execution mode - "interactive" (default) or "workflow"
        """
        self.execution_mode = execution_mode

        # Get max_turns from agentic_nodes configuration, default to 50
        self.max_turns = 50
        if agent_config and hasattr(agent_config, "agentic_nodes") and self.NODE_NAME in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[self.NODE_NAME]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 50)

        self.semantic_model_dir = str(agent_config.path_manager.semantic_model_path(agent_config.current_datasource))
        # ``knowledge_base_dir`` is the sandbox root for FilesystemFuncTool. It
        # now points at the project-scoped ``subject/`` directory so tools can
        # browse all three KB subfolders but not escape the project.
        self.knowledge_base_dir = str(agent_config.path_manager.subject_dir)

        from datus.configuration.node_type import NodeType

        node_type = NodeType.TYPE_SEMANTIC

        # Call parent constructor first to set up node_config
        super().__init__(
            node_id=f"{self.NODE_NAME}_node",
            description=f"Semantic model generation node: {self.NODE_NAME}",
            node_type=node_type,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
            scope=scope,
            is_subagent=is_subagent,
            session_id=session_id,
        )

        # Setup tools
        self.db_func_tool: Optional[DBFuncTool] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.generation_tools: Optional[GenerationTools] = None
        self.semantic_discovery_tools: Optional[SemanticDiscoveryTools] = None
        from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState

        self.osi_target_state = OsiSemanticModelTargetState()
        self.osi_target_tools = None
        self.ask_user_tool = None
        self.hooks = None
        self.generation_evidence = GenerationEvidence()
        self.sql_modeling_plan: Optional[SqlModelingPlan] = None
        self.sql_modeling_tools: Optional[SqlModelingPlanTools] = None
        self.setup_tools()

        # Debug: log hooks status after setup
        logger.debug(f"Hooks after setup: {self.hooks} (type: {type(self.hooks)})")

    def get_node_name(self) -> str:
        """
        Get the configured node name for this semantic model generation node.

        Returns:
            The configured node name
        """
        return self.NODE_NAME

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Serialize semantic-model writes with metric authoring for this datasource."""
        from datus.agent.node.semantic_authoring import semantic_authoring_guard

        async with semantic_authoring_guard(self.agent_config):
            async for action in super().execute_stream(action_history_manager):
                yield action

    def setup_tools(self):
        """Setup tools for semantic model generation."""
        if not self.agent_config:
            return

        self.tools = []

        self._setup_sql_modeling_tools()
        self._setup_osi_target_tools()
        self._setup_db_tools()
        self._setup_semantic_discovery_tools()
        self._setup_semantic_tools()
        self._setup_generation_tools()
        self._setup_filesystem_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.debug(f"Setup {len(self.tools)} tools for {self.NODE_NAME}: {[tool.name for tool in self.tools]}")

        # Setup hooks (only in interactive mode)
        if self.execution_mode == "interactive":
            self._setup_hooks()

    async def _before_stream(self, ctx: StreamRunContext) -> None:
        """Reset request-local authoring state before the first model turn."""
        await super()._before_stream(ctx)
        self.generation_evidence.reset()
        self.osi_target_state.reset()
        self.sql_modeling_plan = None
        if self.sql_modeling_tools is not None:
            self.sql_modeling_tools.reset()

    def _setup_sql_modeling_tools(self) -> None:
        """Expose the single shared SQL preflight entry point."""
        self.sql_modeling_tools = SqlModelingPlanTools(
            agent_config=self.agent_config,
            sub_agent_name=self.get_node_name(),
            generation_evidence=self.generation_evidence,
            plan_consumer=self._accept_sql_modeling_plan,
            semantic_source_inspector=self._inspect_planned_semantic_sources,
        )
        self.tools.extend(self.sql_modeling_tools.available_tools())

    def _accept_sql_modeling_plan(self, plan: Optional[SqlModelingPlan]) -> None:
        self.sql_modeling_plan = plan

    def _inspect_planned_semantic_sources(self, plan: SqlModelingPlan) -> Dict[str, Any]:
        """Batch-inspect physical SQL sources during preflight when possible."""
        self.sql_modeling_plan = plan
        return inspect_planned_semantic_sources(plan, self.semantic_discovery_tools)

    def _setup_db_tools(self):
        """Setup database tools."""
        try:
            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=self.get_node_name(),
            )
            # Add standard database tools
            self.tools.extend(self.db_func_tool.available_tools())
            logger.debug("Added database tools from DBFuncTool")
        except Exception as e:
            logger.error(f"Failed to setup database tools: {e}")

    def _setup_semantic_discovery_tools(self):
        """Setup read-only semantic discovery tools."""
        try:
            if not self.db_func_tool:
                logger.warning("DBFuncTool not initialized, skipping semantic discovery tools setup")
                return

            self.semantic_discovery_tools = SemanticDiscoveryTools(
                self.db_func_tool,
                enable_semantic_model_profiler=self._semantic_sql_history_profiler_enabled(),
                source_sql_provider=self._semantic_discovery_source_sql,
            )
            self.tools.extend(self.semantic_discovery_tools.available_tools())
            logger.debug("Added semantic discovery tools from SemanticDiscoveryTools")
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

    def _semantic_sql_history_profiler_enabled(self) -> bool:
        """Return true when the optional profiler skill is visible to this node."""
        if not self.skill_manager:
            return False
        skill_patterns_str = self.node_config.get("skills", "")
        if not skill_patterns_str:
            return False
        skill_patterns = self.skill_manager.parse_skill_patterns(skill_patterns_str)
        skills = self.skill_manager.get_available_skills(
            self.get_node_name(),
            patterns=skill_patterns,
            node_class=self.get_node_class_name(),
        )
        return any(skill.name == "semantic-sql-history-profiler" for skill in skills)

    def _setup_semantic_tools(self):
        """Setup semantic function tools (for querying metrics via adapters)."""
        try:
            from datus.agent.node.semantic_authoring import resolve_semantic_adapter_type
            from datus.tools.func_tool.semantic_tools import SemanticTools

            adapter_type = resolve_semantic_adapter_type(self.agent_config)

            # Initialize semantic func tool
            self.semantic_func_tool = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name=self.NODE_NAME,
                adapter_type=adapter_type,
                generation_evidence=self.generation_evidence,
                runtime_db_context_provider=self._semantic_runtime_db_context,
            )

            # Add all available tools from semantic func tool
            semantic_tools = [
                tool for tool in self.semantic_func_tool.available_tools() if tool.name == "validate_semantic"
            ]
            self.tools.extend(semantic_tools)

            tool_names = [tool.name for tool in semantic_tools]
            logger.info(f"Added semantic func tools (adapter: {adapter_type}): {', '.join(tool_names)}")

        except Exception as e:
            logger.error(f"Failed to setup semantic func tools: {e}")

    def _ensure_bash_tool_in_tools(self) -> None:
        """Keep Ossie authoring on the explicit filesystem-tool surface."""
        from datus.agent.node.semantic_authoring import is_osi_authoring

        if is_osi_authoring(self.agent_config):
            return
        super()._ensure_bash_tool_in_tools()

    def _make_filesystem_tool(self, **kwargs):
        """Use a structure-preserving dataset upsert surface for OSI authoring."""
        from datus.agent.node.semantic_authoring import is_osi_authoring

        if not is_osi_authoring(self.agent_config):
            return super()._make_filesystem_tool(**kwargs)

        from datus.configuration.inherited_memory_overrides import get_inherited_memory
        from datus.tools.func_tool.metric_filesystem_tools import OsiSemanticModelFilesystemFuncTool

        root_path = kwargs.pop("root_path", None) or self._resolve_workspace_root()
        datus_home = kwargs.pop("datus_home", None)
        if datus_home is None:
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
        from datus.agent.node.semantic_authoring import resolve_semantic_adapter_type

        return OsiSemanticModelFilesystemFuncTool(
            root_path=root_path,
            current_node=current_node,
            datus_home=datus_home,
            strict=strict,
            inherited_memory_node=inherited_memory_node,
            session_data_dir=session_data_dir,
            mutation_callback=mutation_callback,
            semantic_adapter=resolve_semantic_adapter_type(self.agent_config),
            generation_evidence=self.generation_evidence,
            osi_target_state=self.osi_target_state,
            **kwargs,
        )

    def _setup_filesystem_tools(self):
        """Setup filesystem tools."""
        try:
            from datus.agent.node.semantic_authoring import is_osi_authoring

            filesystem_kwargs = {
                "mutation_callback": self._record_semantic_model_mutation,
            }
            if is_osi_authoring(self.agent_config):
                filesystem_kwargs["mutation_guard"] = self.osi_target_state.require_planned_path
            self.filesystem_func_tool = self._make_filesystem_tool(**filesystem_kwargs)
            filesystem_tools = self.filesystem_func_tool.available_tools()
            filesystem_tools = [tool for tool in filesystem_tools if tool.name != "delete_file"]
            self.tools.extend(filesystem_tools)
            logger.debug("Added filesystem tools: %s", [tool.name for tool in filesystem_tools])
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _record_semantic_model_mutation(self, path=None) -> None:
        """Invalidate publish evidence and freeze an authored OSI plan."""
        self.generation_evidence.record_artifact_mutation(path)
        from datus.agent.node.semantic_authoring import is_osi_authoring

        if is_osi_authoring(self.agent_config):
            self.osi_target_state.record_planned_write()

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
            )

            if authoring_format != "osi":
                self.tools.append(trans_to_function_tool(self.generation_tools.check_semantic_object_exists))
            self.tools.append(trans_to_function_tool(self.generation_tools.publish_semantic_model))
            logger.debug("Added semantic-model generation tools for authoring_format=%s", authoring_format)

        except Exception as e:
            logger.error(f"Failed to setup generation tools: {e}")

    def _setup_osi_target_tools(self) -> None:
        """Expose live target discovery and semantic-model target planning."""
        from datus.agent.node.semantic_authoring import is_osi_authoring

        if not is_osi_authoring(self.agent_config):
            return
        from datus.tools.func_tool import OsiSemanticModelTargetTools, trans_to_function_tool

        self.osi_target_tools = OsiSemanticModelTargetTools(
            self.agent_config,
            target_state=self.osi_target_state,
            generation_evidence=self.generation_evidence,
        )
        self.tools.append(trans_to_function_tool(self.osi_target_tools.list_existing_osi_semantic_models))
        self.tools.append(trans_to_function_tool(self.osi_target_tools.plan_osi_semantic_model_target))

    def _setup_skill_func_tools(self) -> None:
        """Default the optional skill set from the active authoring format."""
        from datus.agent.node.semantic_authoring import default_optional_skills

        if self.node_config.get("skills") is None:
            self.node_config["skills"] = default_optional_skills(self.agent_config, self.NODE_NAME)
        super()._setup_skill_func_tools()

    def _get_required_skills(self) -> list:
        """Host-inject the authoring format specification skill."""
        from datus.agent.node.semantic_authoring import required_authoring_skills

        patterns = required_authoring_skills(self.agent_config, self.NODE_NAME)
        return [pattern.strip() for pattern in patterns.split(",") if pattern.strip()]

    def _setup_hooks(self):
        """Setup hooks for interactive mode."""
        try:
            broker = self._get_or_create_broker()
            self.hooks = GenerationHooks(
                broker=broker,
                agent_config=self.agent_config,
                generation_evidence=self.generation_evidence,
            )
            logger.info("Setup hooks: generation_hooks")
        except Exception as e:
            logger.error(f"Failed to setup generation_hooks: {e}")

    def _get_existing_subject_trees(self) -> list:
        """
        Query existing subject_tree values from metrics storage.

        Returns:
            List of unique subject_path values as List[str]
        """
        try:
            # Get all metrics with subject_path field
            subject_paths = sorted(self.metrics_rag.storage.get_subject_tree_flat())
            logger.debug(f"Found {len(subject_paths)} unique metric subject_paths")
            return subject_paths

        except Exception as e:
            logger.error(f"Error getting existing metric subject_trees: {e}")
            return []

    def _prepare_template_context(self, user_input: SemanticNodeInput) -> dict:
        """
        Prepare template context variables for the semantic model generation template.

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
        context["semantic_model_dir"] = self.semantic_model_dir
        context["knowledge_base_dir"] = self.knowledge_base_dir
        # Filesystem tool is now rooted at project_root (not subject/), so the
        # LLM must pass the full ``subject/<kind>/…`` relative path.
        context["kind_subdir"] = f"subject/semantic_models/{self.agent_config.current_datasource}"
        context["current_datasource"] = self.agent_config.current_datasource
        context["has_ask_user_tool"] = self.ask_user_tool is not None
        context.update(build_datasource_prompt_context(self.agent_config))

        from datus.agent.node.semantic_authoring import (
            resolve_authoring_format,
            resolve_semantic_adapter_type,
        )

        context["authoring_format"] = resolve_authoring_format(self.agent_config)
        context["osi_authoring_spec"] = ""
        if context["authoring_format"] == "osi":
            try:
                if resolve_semantic_adapter_type(self.agent_config) == "dosi":
                    from datus_semantic_dosi.authoring_spec import (
                        authoring_spec_text,
                    )
                else:
                    from datus_semantic_osi.authoring_spec import (
                        authoring_spec_text,
                    )

                context["osi_authoring_spec"] = authoring_spec_text("__OSI_DIALECT__")
            except ImportError:
                logger.debug("Semantic adapter authoring spec unavailable; skipping spec injection")

        logger.debug(f"Prepared template context: {context}")
        return context

    def _build_enhanced_message(
        self,
        user_input: SemanticNodeInput,
        extra_enhanced_parts: Optional[List[str]] = None,
    ) -> str:
        """Add Ossie naming intent to this turn instead of the cached system prompt."""
        from datus.agent.node.semantic_authoring import (
            osi_semantic_model_turn_context,
        )

        parts = list(extra_enhanced_parts or [])
        target_context = osi_semantic_model_turn_context(self.agent_config, user_input)
        if target_context:
            parts.append(target_context)
        return super()._build_enhanced_message(user_input, parts)

    def _system_prompt_snapshot_meta(self, prompt_version: Optional[str]) -> Dict[str, str]:
        """Invalidate snapshots created before semantic targets became request-scoped."""
        meta = super()._system_prompt_snapshot_meta(prompt_version)
        meta["semantic_target_scope"] = "agent_bound_v2"
        return meta

    def _get_system_prompt(
        self,
        prompt_version: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        """
        Get the system prompt for semantic model generation using enhanced template context.

        Args:
            prompt_version: Optional prompt version override (falls back to
                ``node_config`` setting when not supplied)
            template_context: Optional template context variables

        Returns:
            System prompt string loaded from the template
        """
        # ``prompt_version`` kwarg wins over the config default; preserves the
        # template's signature parity with the other nodes. Both authoring
        # formats share one template; the format-specific spec is injected as a
        # required skill.
        template_name = f"{self.NODE_NAME}_system"
        version = prompt_version or self.node_config.get("prompt_version")

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
                message_args={"template_name": template_name, "version": version},
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
        return self._prepare_template_context(ctx.user_input)

    def _build_success_result(self, ctx: StreamRunContext) -> SemanticNodeResult:
        sql_plan_ready = self.generation_evidence.ensure_sql_modeling_plan_resolved()
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            raw_output = ctx.last_successful_output.get("raw_output", "")
            if isinstance(raw_output, dict) or raw_output:
                response_content = raw_output
            else:
                response_content = str(ctx.last_successful_output)

        semantic_model_files, extracted_output = self._extract_semantic_model_and_output_from_response(
            {"content": response_content}
        )
        target_state = getattr(self, "osi_target_state", None)
        planned = target_state.planned if target_state is not None else None
        if planned is not None:
            semantic_model_files = [str(planned["semantic_model_file"])]
        if sql_plan_ready and not semantic_model_files:
            raise RuntimeError(
                "SQL-backed semantic model generation must return the generated or reused semantic_model_files."
            )
        if extracted_output:
            response_content = extracted_output

        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        tokens_used = 0
        if self.execution_mode == "interactive":
            tokens_used = self._extract_total_tokens(ctx.action_history_manager.get_actions())

        user_input = ctx.user_input
        self._finalize_semantic_model_generation(
            semantic_model_files=semantic_model_files,
            catalog=user_input.catalog,
            database=user_input.database,
            db_schema=user_input.db_schema,
        )

        return SemanticNodeResult(
            success=True,
            response=response_content,
            semantic_models=semantic_model_files,
            tokens_used=int(tokens_used),
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

    def _finalize_semantic_model_generation(
        self,
        semantic_model_files: list[str],
        catalog=None,
        database=None,
        db_schema=None,
    ) -> None:
        """Validate and publish semantic model artifacts without relying on one LLM tool call."""
        from datus.agent.node.semantic_authoring import is_osi_authoring

        if is_osi_authoring(self.agent_config):
            self._finalize_osi_semantic_model_generation()
            return

        if not semantic_model_files or self.generation_evidence.semantic_kb_sync_passed:
            return

        if not self.generation_evidence.validation_passed:
            if not getattr(self, "semantic_func_tool", None):
                raise RuntimeError(
                    "Semantic model generation produced semantic_model_files, but validate_semantic is unavailable."
                )
            validation_result = self.semantic_func_tool.validate_semantic(scope="semantic_model")
            self.generation_evidence.record_validation_result(validation_result)
            if not self._tool_succeeded(validation_result):
                raise RuntimeError(
                    f"validate_semantic failed before publishing semantic models: {self._tool_error(validation_result)}"
                )

        del catalog, database, db_schema
        publish_result = self.generation_tools.publish_semantic_model(semantic_model_files)
        if not self._tool_succeeded(publish_result):
            raise RuntimeError(f"Semantic model KB sync failed: {self._tool_error(publish_result)}")

    def _finalize_osi_semantic_model_generation(self) -> None:
        """Run the same exact-target gate whether or not the LLM called the end tool."""
        if not getattr(self, "generation_tools", None):
            raise RuntimeError("OSI semantic model generation tools are unavailable.")

        try:
            semantic_model_file, resolved, model_name = self.generation_tools.resolve_planned_osi_semantic_target()
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        if self.generation_evidence.semantic_kb_sync_passed:
            if self.generation_evidence.semantic_artifact_validation_passed(model_name, resolved):
                return
            self.generation_evidence.invalidate_artifact_evidence()

        if not self.generation_evidence.semantic_artifact_validation_passed(model_name, resolved):
            if not getattr(self, "semantic_func_tool", None):
                raise RuntimeError(
                    "Semantic model generation produced a planned OSI target, but validate_semantic is unavailable."
                )
            validation_result = self.semantic_func_tool.validate_semantic(
                scope="semantic_model",
                semantic_model_name=model_name,
            )
            self.generation_evidence.record_validation_result(validation_result)
            if not self._tool_succeeded(validation_result):
                raise RuntimeError(
                    f"validate_semantic failed before publishing semantic models: {self._tool_error(validation_result)}"
                )
            if not self.generation_evidence.record_semantic_artifact_validation(model_name, resolved):
                raise RuntimeError("Cannot bind semantic validation evidence to the planned OSI artifact.")

        publish_result = self.generation_tools.publish_semantic_model([semantic_model_file])
        if not self._tool_succeeded(publish_result):
            raise RuntimeError(f"OSI semantic model KB sync failed: {self._tool_error(publish_result)}")

    def _extract_semantic_model_and_output_from_response(self, output: dict) -> tuple[list[str], Optional[str]]:
        """
        Extract semantic_model_files and formatted output from model response.

        Per prompt template requirements, LLM should return JSON format:
        {"semantic_model_files": ["path1.yml", "path2.yml"], "output": "markdown text"}

        Args:
            output: Output dictionary from model generation

        Returns:
            Tuple of (semantic_model_files: List[str], output_string: Optional[str])
        """
        try:
            from datus.utils.json_utils import strip_json_str

            content = output.get("content", "")
            logger.info(f"extract_semantic_model_and_output_from_final_resp: {content} (type: {type(content)})")

            # Case 1: content is already a dict (most common)
            if isinstance(content, dict):
                semantic_model_files = content.get("semantic_model_files")
                output_text = content.get("output")
                if semantic_model_files and isinstance(semantic_model_files, list):
                    logger.debug(f"Extracted from dict: semantic_model_files={semantic_model_files}")
                    return semantic_model_files, output_text
                else:
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
                            semantic_model_files = parsed.get("semantic_model_files")
                            output_text = parsed.get("output")
                            if semantic_model_files and isinstance(semantic_model_files, list):
                                logger.debug(f"Extracted from JSON string: semantic_model_files={semantic_model_files}")
                                return semantic_model_files, output_text
                            else:
                                logger.warning(
                                    f"Parsed JSON but missing expected keys or invalid format: {parsed.keys()}"
                                )
                    except Exception as e:
                        logger.warning(f"Failed to parse cleaned JSON: {e}. Cleaned content: {cleaned_json[:200]}")

            logger.warning(f"Could not extract semantic_model_files from response. Content type: {type(content)}")
            return [], None

        except Exception as e:
            logger.error(f"Unexpected error extracting semantic_model_files: {e}", exc_info=True)
            return [], None
