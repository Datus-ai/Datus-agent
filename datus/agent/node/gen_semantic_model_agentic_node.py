# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenSemanticModelAgenticNode implementation for semantic model generation.

This module provides a specialized implementation of AgenticNode focused on
semantic model generation with support for filesystem tools, generation tools,
database tools, hooks, and metricflow MCP server integration.
"""

from typing import Any, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput, SemanticNodeResult
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.tools.func_tool.semantic_discovery_tools import SemanticDiscoveryTools
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
    DEFAULT_SKILLS = "gen-semantic-model"

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
        self.ask_user_tool = None
        self.hooks = None
        self.generation_evidence = GenerationEvidence()
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

    def setup_tools(self):
        """Setup tools for semantic model generation."""
        if not self.agent_config:
            return

        self.tools = []

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

            self.semantic_discovery_tools = SemanticDiscoveryTools(self.db_func_tool)
            self.tools.extend(self.semantic_discovery_tools.available_tools())
            logger.debug("Added semantic discovery tools from SemanticDiscoveryTools")
        except Exception as e:
            logger.error(f"Failed to setup semantic discovery tools: {e}")

    def _setup_semantic_tools(self):
        """Setup semantic function tools (for querying metrics via adapters)."""
        try:
            from datus.tools.func_tool.semantic_tools import SemanticTools

            adapter_type = None
            if hasattr(self.agent_config, "agentic_nodes") and self.NODE_NAME in self.agent_config.agentic_nodes:
                node_config = self.agent_config.agentic_nodes[self.NODE_NAME]
                if isinstance(node_config, dict) and node_config.get("semantic_adapter"):
                    adapter_type = node_config.get("semantic_adapter")
            adapter_type = self.agent_config.resolve_semantic_adapter(adapter_type)

            # Initialize semantic func tool
            self.semantic_func_tool = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name=self.NODE_NAME,
                adapter_type=adapter_type,
                generation_evidence=self.generation_evidence,
                runtime_db_context_provider=self._semantic_runtime_db_context,
            )

            # Add all available tools from semantic func tool
            semantic_tools = self.semantic_func_tool.available_tools()
            self.tools.extend(semantic_tools)

            tool_names = [tool.name for tool in semantic_tools]
            logger.info(f"Added semantic func tools (adapter: {adapter_type}): {', '.join(tool_names)}")

        except Exception as e:
            logger.error(f"Failed to setup semantic func tools: {e}")

    def _setup_filesystem_tools(self):
        """Setup filesystem tools."""
        try:
            self.filesystem_func_tool = self._make_filesystem_tool()

            self.tools.extend(self.filesystem_func_tool.available_tools())
            logger.debug("Added filesystem tools: read_file, write_file, edit_file, glob, grep")
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _setup_generation_tools(self):
        """Setup generation tools."""
        try:
            from datus.agent.node.semantic_authoring import resolve_authoring_format
            from datus.tools.func_tool import trans_to_function_tool

            self.generation_tools = GenerationTools(
                self.agent_config,
                generation_evidence=self.generation_evidence,
                authoring_format=resolve_authoring_format(self.agent_config, self.node_config),
            )

            self.tools.append(trans_to_function_tool(self.generation_tools.check_semantic_object_exists))
            self.tools.append(trans_to_function_tool(self.generation_tools.end_semantic_model_generation))
            logger.debug("Added tools: check_semantic_object_exists, end_semantic_model_generation")

        except Exception as e:
            logger.error(f"Failed to setup generation tools: {e}")

    def _setup_skill_func_tools(self) -> None:
        """Avoid injecting MetricFlow authoring skills into OSI workflows."""
        from datus.agent.node.semantic_authoring import is_osi_authoring

        node_config = getattr(self, "node_config", None) or {}
        if is_osi_authoring(self.agent_config, node_config) and "skills" not in node_config:
            logger.info("Skipping default MetricFlow skills for OSI semantic-model authoring")
            return
        super()._setup_skill_func_tools()

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
            default_osi_semantic_model_file,
            default_osi_semantic_model_name,
        )

        context["default_osi_semantic_model_name"] = default_osi_semantic_model_name(self.agent_config)
        context["default_osi_semantic_model_file"] = default_osi_semantic_model_file(self.agent_config)

        logger.debug(f"Prepared template context: {context}")
        return context

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
        from datus.agent.node.semantic_authoring import (
            AUTHORING_FORMAT_OSI,
            osi_prompt_version,
            osi_template_name,
            resolve_authoring_format,
        )

        # ``prompt_version`` kwarg wins over the config default; preserves the
        # template's signature parity with the other nodes.
        # OSI mode uses a separate template name so the default metricflow
        # template and its latest-version scan are left untouched.
        authoring_format = resolve_authoring_format(self.agent_config, self.node_config)
        if authoring_format == AUTHORING_FORMAT_OSI:
            template_name = osi_template_name(self.NODE_NAME)
            requested = prompt_version or self.node_config.get("prompt_version")
            version = osi_prompt_version(self.agent_config, self.NODE_NAME, requested)
        else:
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

        synced_files = []
        failed_files = []
        for semantic_model_file in semantic_model_files:
            if self._save_to_db(
                semantic_model_file,
                catalog=catalog,
                database=database,
                db_schema=db_schema,
            ):
                synced_files.append(semantic_model_file)
            else:
                failed_files.append(semantic_model_file)

        if failed_files:
            raise RuntimeError(
                "Semantic model generation produced file(s), but failed to sync to Knowledge Base: "
                f"{', '.join(failed_files)}"
            )

        logger.info(f"Auto-saved {len(synced_files)} semantic models to database")

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

    def _save_to_db(self, semantic_model_file: str, catalog=None, database=None, db_schema=None) -> bool:
        """
        Save generated semantic model to database (synchronous).

        Args:
            semantic_model_file: Name of the semantic model file (e.g., "orders.yaml")
            catalog: Optional catalog override
            database: Optional database override
            db_schema: Optional schema override
        """
        try:
            import os

            from datus.cli.generation_hooks import resolve_kb_sandbox_path

            full_path = resolve_kb_sandbox_path(semantic_model_file, "semantic", self.knowledge_base_dir)
            if not full_path:
                logger.warning(f"Semantic model file rejected by sandbox check: {semantic_model_file!r}")
                return False

            if not os.path.exists(full_path):
                logger.warning(f"Semantic model file not found: {full_path}")
                return False

            from datus.agent.node.semantic_authoring import is_osi_authoring

            if is_osi_authoring(self.agent_config, self.node_config):
                if not self.generation_tools:
                    logger.error("Generation tools unavailable for OSI semantic sync")
                    return False
                result = self.generation_tools.sync_osi_semantic_to_db(full_path)
                if result.get("success"):
                    self.generation_evidence.mark_kb_sync("semantic")
                    logger.info(f"Successfully saved OSI semantic model to database: {result.get('message')}")
                    return True
                logger.error(f"Failed to save OSI semantic model to database: {result.get('error', 'unknown error')}")
                return False

            # Call static method to save to database
            # Deduplication is handled inside _sync_semantic_to_db
            result = GenerationHooks._sync_semantic_to_db(
                full_path, self.agent_config, catalog=catalog, database=database, schema=db_schema
            )

            if result.get("success"):
                self.generation_evidence.mark_kb_sync("semantic")
                logger.info(f"Successfully saved to database: {result.get('message')}")
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Failed to save to database: {error}")
                return False

        except Exception as e:
            logger.error(f"Error saving to database: {e}", exc_info=True)
            raise
