# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenMetricsAgenticNode implementation for metrics generation.

This module provides a specialized implementation of AgenticNode focused on
metrics generation with support for filesystem tools, generation tools,
hooks, and metricflow MCP server integration.
"""

from typing import Any, Dict, List, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.configuration.agent_config import AgentConfig
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput, SemanticNodeResult
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.tools.func_tool.metric_queryability import (
    extract_metric_queryability_contracts,
    summarize_queryability_contracts,
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
    result_class = SemanticNodeResult

    # Define-metric workflow scoped to ``gen_metrics`` via SKILL.md allowed_agents.
    DEFAULT_SKILLS = "gen-metrics, gen-semantic-model"

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

        # Get max_turns from agentic_nodes configuration, default to 30
        self.max_turns = 40
        if agent_config and hasattr(agent_config, "agentic_nodes") and self.NODE_NAME in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[self.NODE_NAME]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 40)

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
        self.ask_user_tool = None
        self.hooks = None
        self.generation_evidence = GenerationEvidence()
        self.setup_tools()

    def get_node_name(self) -> str:
        """
        Get the configured node name for this metrics generation node.

        Returns:
            The configured node name
        """
        return self.NODE_NAME

    def setup_tools(self):
        """Setup tools for metrics generation."""
        if not self.agent_config:
            return

        self.tools = []

        # Setup db_tools.*, semantic_discovery_tools.*, generation_tools.*, filesystem_tools.*, semantic_tools.*
        self._setup_db_tools()
        self._setup_semantic_discovery_tools()
        self._setup_generation_tools()
        self._setup_filesystem_tools()
        self._setup_semantic_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.info(f"Setup {len(self.tools)} tools for {self.NODE_NAME}: {[tool.name for tool in self.tools]}")

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
            from datus.tools.func_tool import trans_to_function_tool

            self.generation_tools = GenerationTools(self.agent_config, generation_evidence=self.generation_evidence)

            self.tools.append(trans_to_function_tool(self.generation_tools.check_semantic_object_exists))
            self.tools.append(trans_to_function_tool(self.generation_tools.end_metric_generation))
            self.tools.append(trans_to_function_tool(self.generation_tools.end_semantic_model_generation))
            logger.debug(
                "Added tools: check_semantic_object_exists, end_metric_generation, end_semantic_model_generation"
            )

        except Exception as e:
            logger.error(f"Failed to setup generation tools: {e}")

    def _setup_semantic_tools(self):
        """Setup semantic tools for metrics querying and exploration."""
        try:
            from datus.tools.func_tool.semantic_tools import SemanticTools

            adapter_type = None
            if hasattr(self.agent_config, "agentic_nodes") and self.NODE_NAME in self.agent_config.agentic_nodes:
                node_config = self.agent_config.agentic_nodes[self.NODE_NAME]
                if isinstance(node_config, dict) and node_config.get("semantic_adapter"):
                    adapter_type = node_config.get("semantic_adapter")
            adapter_type = self.agent_config.resolve_semantic_adapter(adapter_type)

            # Initialize semantic func tool
            self.semantic_tools = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name=self.NODE_NAME,
                adapter_type=adapter_type,
                generation_evidence=self.generation_evidence,
            )

            # Add all available tools from semantic func tool
            semantic_tools = self.semantic_tools.available_tools()
            self.tools.extend(semantic_tools)

            tool_names = [tool.name for tool in semantic_tools]
            logger.info(f"Added semantic tools (adapter: {adapter_type}): {', '.join(tool_names)}")

        except Exception as e:
            logger.error(f"Failed to setup semantic tools: {e}")

    def _setup_db_tools(self):
        """Setup database tools for schema introspection."""
        try:
            from datus.tools.func_tool import DBFuncTool

            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=self.NODE_NAME,
            )
            self.tools.extend(self.db_func_tool.available_tools())
            logger.debug("Added database tools from DBFuncTool")
        except Exception as e:
            logger.error(f"Failed to setup database tools: {e}")

    def _setup_semantic_discovery_tools(self):
        """Setup read-only semantic discovery tools."""
        try:
            if not self.db_func_tool:
                logger.warning("DBFuncTool not initialized, skipping semantic_discovery_tools setup")
                return

            from datus.tools.func_tool.semantic_discovery_tools import SemanticDiscoveryTools

            self.semantic_discovery_tools = SemanticDiscoveryTools(self.db_func_tool)
            self.tools.extend(self.semantic_discovery_tools.available_tools())
            logger.debug(
                "Added semantic discovery tools: analyze_table_relationships, get_multiple_tables_ddl, "
                "analyze_column_usage_patterns, analyze_metric_candidates_from_history"
            )
        except Exception as e:
            logger.error(f"Failed to setup semantic discovery tools: {e}")

    def _tool_category_map(self) -> Dict[str, List[Any]]:
        """Map tools to permission categories so profile rules apply.

        ``generation_tools`` / ``semantic_discovery_tools`` expose semantic
        layer operations (``end_metric_generation`` etc.) so they ride the
        ``semantic_tools`` category for the sake of profile rules.
        """
        mapping = super()._tool_category_map()
        if self.db_func_tool:
            mapping["db_tools"] = list(self.db_func_tool.available_tools())
        semantic_bucket: List[Any] = []
        if getattr(self, "semantic_tools", None):
            semantic_bucket.extend(self.semantic_tools.available_tools())
        if self.generation_tools:
            from datus.tools.func_tool import trans_to_function_tool

            semantic_bucket.extend(
                [
                    trans_to_function_tool(self.generation_tools.check_semantic_object_exists),
                    trans_to_function_tool(self.generation_tools.end_metric_generation),
                    trans_to_function_tool(self.generation_tools.end_semantic_model_generation),
                ]
            )
        if self.semantic_discovery_tools:
            semantic_bucket.extend(self.semantic_discovery_tools.available_tools())
        if semantic_bucket:
            mapping["semantic_tools"] = semantic_bucket
        if self.filesystem_func_tool:
            mapping["filesystem_tools"] = list(self.filesystem_func_tool.available_tools())
        if self.ask_user_tool:
            mapping.setdefault("tools", []).extend(self.ask_user_tool.available_tools())
        return mapping

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
        version = (
            prompt_version or getattr(self.input, "prompt_version", None) or self.node_config.get("prompt_version")
        )

        # Hardcoded system_prompt template name
        template_name = f"{self.NODE_NAME}_system"

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

    def _build_success_result(self, ctx: StreamRunContext) -> SemanticNodeResult:
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            raw_output = ctx.last_successful_output.get("raw_output", "")
            if isinstance(raw_output, dict) or raw_output:
                response_content = raw_output
            else:
                response_content = str(ctx.last_successful_output)

        semantic_model_file, metric_file, status, extracted_output = self._extract_metric_and_output_from_response(
            {"content": response_content}
        )
        if extracted_output:
            response_content = extracted_output

        if not isinstance(response_content, str):
            response_content = str(response_content) if response_content else ""

        self._finalize_metric_generation(
            semantic_model_file=semantic_model_file,
            metric_file=metric_file,
            status=status,
        )

        tokens_used = 0
        if self.execution_mode == "interactive":
            tokens_used = self._extract_total_tokens(ctx.action_history_manager.get_actions())

        return SemanticNodeResult(
            success=True,
            response=response_content,
            semantic_models=[metric_file] if metric_file else [],
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

    def _resolve_metric_artifact_path(self, path: Optional[str], kind: str) -> str:
        if not path:
            return ""

        from datus.cli.generation_hooks import resolve_kb_sandbox_path

        resolved_path = resolve_kb_sandbox_path(path, kind, self.knowledge_base_dir)
        if not resolved_path:
            raise RuntimeError(f"Metric generation reported {kind}_file outside Knowledge Base sandbox: {path!r}")
        return resolved_path

    def _set_metric_queryability_contracts_from_input(self, user_input: Optional[SemanticNodeInput]) -> None:
        if not user_input:
            return
        contracts = extract_metric_queryability_contracts(getattr(user_input, "user_message", "") or "")
        self.generation_evidence.set_metric_queryability_contracts(contracts)

    def _finalize_metric_generation(
        self,
        semantic_model_file: Optional[str],
        metric_file: Optional[str],
        status: Optional[str],
    ) -> None:
        """Ensure generated metric artifacts are published without relying on one LLM tool call."""
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
                "Non-skipped metric responses must include metric_file or call end_metric_generation."
            )

        if not metric_file:
            return

        if not self.generation_tools:
            raise RuntimeError("Metric generation produced a metric_file, but generation tools are unavailable.")
        self._set_metric_queryability_contracts_from_input(getattr(self, "input", None))

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
        abs_semantic_model_file = (
            self._resolve_metric_artifact_path(semantic_model_file, "semantic") if semantic_model_file else ""
        )
        preflight_error = self.generation_tools._validate_metric_file_has_blocks(abs_metric_file)
        if preflight_error:
            raise RuntimeError(preflight_error)

        metric_names = self.generation_tools._extract_metric_names_from_file(abs_metric_file)
        if metric_names and not self.generation_evidence.has_metric_dry_run(metric_names):
            if not getattr(self, "semantic_tools", None):
                raise RuntimeError("Metric generation produced a metric_file, but query_metrics is unavailable.")
            dry_run_result = self.semantic_tools.query_metrics(metrics=metric_names, dry_run=True)
            self.generation_evidence.record_metric_dry_run(metric_names, dry_run_result)
            if not self._tool_succeeded(dry_run_result):
                raise RuntimeError(
                    "query_metrics(dry_run=True) failed for generated metric(s) "
                    f"{', '.join(metric_names)}: {self._tool_error(dry_run_result)}"
                )
        if metric_names and not self.generation_evidence.has_required_queryability_dry_runs(metric_names):
            missing_contracts = self.generation_evidence.missing_queryability_contracts(metric_names)
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                "query_metrics(dry_run=True) must pass with the source SQL group-by dimensions before "
                "publishing metrics. Run a dry-run query for the generated metric names with the matching "
                "dimensions/time grain, fix semantic model join or dimension issues, and retry. "
                f"Missing: {summarize_queryability_contracts(missing_contracts)}",
            )

        publish_result = self.generation_tools.end_metric_generation(
            metric_file=abs_metric_file,
            semantic_model_file=abs_semantic_model_file,
        )
        if not self._tool_succeeded(publish_result):
            raise RuntimeError(f"Metric KB sync failed: {self._tool_error(publish_result)}")
        self.generation_evidence.mark_kb_sync("metric")

    def _extract_metric_and_output_from_response(
        self, output: dict
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Extract semantic model file, metric file, status and formatted output from model response.

        Per prompt template requirements, LLM should return JSON format:
        {"semantic_model_file": "path.yml", "metric_file": "path.yml",
         "status": "generated" | "skipped", "output": "markdown text"}

        ``status`` is optional for backward compatibility; absent values are treated as ``"generated"``.

        Args:
            output: Output dictionary from model generation

        Returns:
            Tuple of (semantic_model_file, metric_file, status, output_string), each Optional[str].
        """
        try:
            from datus.utils.json_utils import strip_json_str

            content = output.get("content", "")
            logger.info(f"extract_metric_and_output_from_response: {content} (type: {type(content)})")

            # Case 1: content is already a dict (most common)
            if isinstance(content, dict):
                output_text = content.get("output")
                semantic_model_file = content.get("semantic_model_file")
                metric_file = content.get("metric_file")
                status = content.get("status")
                normalized_status = status.strip().lower() if isinstance(status, str) else None

                if (metric_file and isinstance(metric_file, str)) or normalized_status:
                    logger.debug(
                        f"Extracted from dict: semantic_model_file={semantic_model_file}, "
                        f"metric_file={metric_file}, status={normalized_status}"
                    )
                    return semantic_model_file, metric_file, normalized_status, output_text

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
                            semantic_model_file = parsed.get("semantic_model_file")
                            metric_file = parsed.get("metric_file")
                            status = parsed.get("status")
                            normalized_status = status.strip().lower() if isinstance(status, str) else None

                            if (metric_file and isinstance(metric_file, str)) or normalized_status:
                                logger.debug(
                                    f"Extracted from JSON string: "
                                    f"semantic_model_file={semantic_model_file}, "
                                    f"metric_file={metric_file}, status={normalized_status}"
                                )
                                return semantic_model_file, metric_file, normalized_status, output_text

                            logger.warning(f"Parsed JSON but missing expected keys or invalid format: {parsed.keys()}")
                    except Exception as e:
                        logger.warning(f"Failed to parse cleaned JSON: {e}. Cleaned content: {cleaned_json[:200]}")

            logger.warning(f"Could not extract metric_file from response. Content type: {type(content)}")
            return None, None, None, None

        except Exception as e:
            logger.error(f"Unexpected error extracting metric_file: {e}", exc_info=True)
            return None, None, None, None
