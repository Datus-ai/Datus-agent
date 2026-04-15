# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenMetricsAgenticNode implementation for metrics generation.

This module provides a specialized implementation of AgenticNode focused on
metrics generation with support for filesystem tools, generation tools,
hooks, and metricflow MCP server integration.
"""

from typing import AsyncGenerator, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.cli.execution_state import ExecutionInterrupted
from datus.cli.generation_hooks import make_kb_path_normalizer
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput, SemanticNodeResult
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.utils.loggings import get_logger
from datus.utils.message_utils import MessagePart, build_structured_content

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

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        subject_tree: Optional[list] = None,
        scope: Optional[str] = None,
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

        self.metrics_dir = str(agent_config.path_manager.semantic_model_path(agent_config.current_database))
        self.knowledge_base_dir = str(agent_config.path_manager.knowledge_base_home)

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
        )

        # Initialize metrics storage for context queries
        from datus.storage.metric.store import MetricRAG

        self.metrics_rag = MetricRAG(agent_config)

        # Setup tools
        self.db_func_tool = None
        self.gen_semantic_model_tools = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.generation_tools: Optional[GenerationTools] = None
        self.ask_user_tool = None
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

        # Setup db_tools.*, gen_semantic_model_tools.*, generation_tools.*, filesystem_tools.*, semantic_tools.*
        self._setup_db_tools()
        self._setup_gen_semantic_model_tools()
        self._setup_generation_tools()
        self._setup_filesystem_tools()
        self._setup_semantic_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.info(f"Setup {len(self.tools)} tools for {self.NODE_NAME}: {[tool.name for tool in self.tools]}")

    def _setup_filesystem_tools(self):
        """Setup filesystem tools."""
        try:
            self.filesystem_func_tool = FilesystemFuncTool(
                root_path=self.knowledge_base_dir,
                path_normalizer=make_kb_path_normalizer(self.agent_config, default_kind="metric"),
            )

            self.tools.extend(self.filesystem_func_tool.available_tools())
            logger.debug("Added filesystem tools: read_file, write_file, edit_file, glob, grep")
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _setup_generation_tools(self):
        """Setup generation tools."""
        try:
            from datus.tools.func_tool import trans_to_function_tool

            self.generation_tools = GenerationTools(self.agent_config)

            self.tools.append(trans_to_function_tool(self.generation_tools.check_semantic_object_exists))
            self.tools.append(trans_to_function_tool(self.generation_tools.end_metric_generation))
            logger.debug("Added tools: check_semantic_object_exists, end_metric_generation")

        except Exception as e:
            logger.error(f"Failed to setup generation tools: {e}")

    def _setup_semantic_tools(self):
        """Setup semantic tools for metrics querying and exploration."""
        try:
            from datus.tools.func_tool.semantic_tools import SemanticTools

            # Default to "metricflow", override from config if specified
            adapter_type = "metricflow"
            if hasattr(self.agent_config, "agentic_nodes") and self.NODE_NAME in self.agent_config.agentic_nodes:
                node_config = self.agent_config.agentic_nodes[self.NODE_NAME]
                if isinstance(node_config, dict) and node_config.get("semantic_adapter"):
                    adapter_type = node_config.get("semantic_adapter")

            # Initialize semantic func tool
            self.semantic_tools = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name=self.NODE_NAME,
                adapter_type=adapter_type,
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

            self.db_func_tool = DBFuncTool.create_dynamic(
                self.agent_config,
                sub_agent_name=self.NODE_NAME,
            )
            self.tools.extend(self.db_func_tool.available_tools())
            logger.debug("Added database tools from DBFuncTool")
        except Exception as e:
            logger.error(f"Failed to setup database tools: {e}")

    def _setup_gen_semantic_model_tools(self):
        """Setup semantic model generation tools (table relationships, column usage patterns)."""
        try:
            if not self.db_func_tool:
                logger.warning("DBFuncTool not initialized, skipping gen_semantic_model_tools setup")
                return

            from datus.tools.func_tool.gen_semantic_model_tools import GenSemanticModelTools

            self.gen_semantic_model_tools = GenSemanticModelTools(self.db_func_tool)
            self.tools.extend(self.gen_semantic_model_tools.available_tools())
            logger.debug(
                "Added gen_semantic_model tools: analyze_table_relationships, get_multiple_tables_ddl, analyze_column_usage_patterns"
            )
        except Exception as e:
            logger.error(f"Failed to setup gen_semantic_model tools: {e}")

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
        context = {}

        # Tool name lists for template display
        context["native_tools"] = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"
        context["mcp_tools"] = ", ".join(list(self.mcp_servers.keys())) if self.mcp_servers else "None"
        context["semantic_model_dir"] = self.metrics_dir
        context["knowledge_base_dir"] = self.knowledge_base_dir
        context["kind_subdir"] = "semantic_models"
        context["current_database"] = self.agent_config.current_database
        context["has_ask_user_tool"] = self.ask_user_tool is not None

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
        conversation_summary: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        """
        Get the system prompt for metrics generation using enhanced template context.

        Args:
            conversation_summary: Optional summary from previous conversation compact
            template_context: Optional template context variables

        Returns:
            System prompt string loaded from the template
        """
        version = getattr(self.input, "prompt_version", None) or self.node_config.get("prompt_version")

        # Hardcoded system_prompt template name
        template_name = f"{self.NODE_NAME}_system"

        try:
            # Prepare template variables
            template_vars = {
                "agent_config": self.agent_config,
                "conversation_summary": conversation_summary,
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

    async def execute_stream(
        self,
        action_history_manager: Optional[ActionHistoryManager] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the metrics generation with streaming support.

        Args:
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        # Get input from self.input
        if self.input is None:
            raise ValueError("Metrics input not set. Set self.input before calling execute_stream.")

        user_input = self.input

        # Create initial action
        action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type=self.get_node_name(),
            messages=f"User: {user_input.user_message}",
            input_data=user_input.model_dump(),
            status=ActionStatus.PROCESSING,
        )
        action_history_manager.add_action(action)
        yield action

        try:
            # Session management (only in interactive mode)
            session = None
            conversation_summary = None
            if self.execution_mode == "interactive":
                # Check for auto-compact before session creation to ensure fresh context
                await self._auto_compact()

                # Get or create session and any available summary
                session, conversation_summary = self._get_or_create_session()

            # Prepare enhanced template context
            template_context = self._prepare_template_context(user_input)

            # Get system instruction from template with enhanced context
            system_instruction = self._get_system_prompt(conversation_summary, template_context)

            # Add context to user message if provided
            enhanced_message = user_input.user_message
            enhanced_parts = []

            if user_input.catalog or user_input.database or user_input.db_schema:
                context_parts = []
                if user_input.catalog:
                    context_parts.append(f"catalog: {user_input.catalog}")
                if user_input.database:
                    context_parts.append(f"database: {user_input.database}")
                if user_input.db_schema:
                    context_parts.append(f"schema: {user_input.db_schema}")
                context_part_str = f"Context: {', '.join(context_parts)}"
                enhanced_parts.append(context_part_str)

            if enhanced_parts:
                enhanced_context = "\n\n".join(enhanced_parts)
                enhanced_message = build_structured_content(
                    [
                        MessagePart(type="enhanced", content=enhanced_context),
                        MessagePart(type="user", content=user_input.user_message),
                    ]
                )

            logger.debug(f"Tools available : {len(self.tools)} tools - {[tool.name for tool in self.tools]}")
            logger.debug(f"MCP servers available : {len(self.mcp_servers)} servers - {list(self.mcp_servers.keys())}")
            logger.debug(f"Tools: {len(self.tools)}, MCP: {len(self.mcp_servers)}")

            # Initialize response collection variables
            response_content = ""
            metric_file = None
            tokens_used = 0
            last_successful_output = None

            # Stream response using the model's generate_with_tools_stream
            async for stream_action in self.model.generate_with_tools_stream(
                prompt=enhanced_message,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
                instruction=system_instruction,
                max_turns=self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=None,
                agent_name=self.get_node_name(),
                interrupt_controller=self.interrupt_controller,
            ):
                yield stream_action

                # Collect response content from successful actions
                if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        last_successful_output = stream_action.output
                        # Look for content in various possible fields
                        raw_output = stream_action.output.get("raw_output", "")
                        # Handle case where raw_output is already a dict
                        if isinstance(raw_output, dict):
                            response_content = raw_output
                        elif raw_output:
                            response_content = raw_output

            # If we still don't have response_content, check the last successful output
            if not response_content and last_successful_output:
                # Try different fields that might contain the response
                raw_output = last_successful_output.get("raw_output", "")
                if isinstance(raw_output, dict):
                    response_content = raw_output
                elif raw_output:
                    response_content = raw_output
                else:
                    response_content = str(last_successful_output)  # Fallback to string representation

            # Extract semantic_model_file, metric_file and output from the final response_content
            semantic_model_file, metric_file, extracted_output = self._extract_metric_and_output_from_response(
                {"content": response_content}
            )
            if extracted_output:
                response_content = extracted_output

            # Extract token usage (only in interactive mode with session)
            tokens_used = 0
            if self.execution_mode == "interactive":
                # With our streaming token fix, only the final assistant action will have accurate usage
                final_actions = action_history_manager.get_actions()

                # Find the final assistant action with token usage
                for action in reversed(final_actions):
                    if action.role == "assistant":
                        if action.output and isinstance(action.output, dict):
                            usage_info = action.output.get("usage", {})
                            if usage_info and isinstance(usage_info, dict) and usage_info.get("total_tokens"):
                                tokens_used = usage_info.get("total_tokens", 0)
                                if tokens_used > 0:
                                    break
                                else:
                                    logger.warning(f"no usage token found in this action {action.messages}")

            # Create final result
            result = SemanticNodeResult(
                success=True,
                response=response_content,
                semantic_models=[metric_file] if metric_file else [],  # Note: field name kept for compatibility
                tokens_used=int(tokens_used),
            )

            # Add to internal actions list
            self.actions.extend(action_history_manager.get_actions())

            # Create final action
            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="metrics_response",
                messages=f"{self.get_node_name()} interaction completed successfully",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except ExecutionInterrupted:
            raise

        except Exception as e:
            logger.error(f"{self.get_node_name()} execution error: {e}")

            # Create error result
            error_result = SemanticNodeResult(
                success=False,
                error=str(e),
                response="Sorry, I encountered an error while processing your request.",
                tokens_used=0,
            )

            # Update action with error
            action_history_manager.update_current_action(
                status=ActionStatus.FAILED,
                output=error_result.model_dump(),
                messages=f"Error: {str(e)}",
            )

            # Create error action
            error_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="error",
                messages=f"{self.get_node_name()} interaction failed: {str(e)}",
                input_data=user_input.model_dump(),
                output_data=error_result.model_dump(),
                status=ActionStatus.FAILED,
            )
            action_history_manager.add_action(error_action)
            yield error_action

    def _extract_metric_and_output_from_response(
        self, output: dict
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract semantic model file, metric file and formatted output from model response.

        Per prompt template requirements, LLM should return JSON format:
        {"semantic_model_file": "path.yml", "metric_file": "path.yml", "output": "markdown text"}

        Args:
            output: Output dictionary from model generation

        Returns:
            Tuple of (semantic_model_file: Optional[str], metric_file: Optional[str], output_string: Optional[str])
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

                if metric_file and isinstance(metric_file, str):
                    logger.debug(
                        f"Extracted from dict: semantic_model_file={semantic_model_file}, metric_file={metric_file}"
                    )
                    return semantic_model_file, metric_file, output_text

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

                            if metric_file and isinstance(metric_file, str):
                                logger.debug(
                                    f"Extracted from JSON string: "
                                    f"semantic_model_file={semantic_model_file}, metric_file={metric_file}"
                                )
                                return semantic_model_file, metric_file, output_text

                            logger.warning(f"Parsed JSON but missing expected keys or invalid format: {parsed.keys()}")
                    except Exception as e:
                        logger.warning(f"Failed to parse cleaned JSON: {e}. Cleaned content: {cleaned_json[:200]}")

            logger.warning(f"Could not extract metric_file from response. Content type: {type(content)}")
            return None, None, None

        except Exception as e:
            logger.error(f"Unexpected error extracting metric_file: {e}", exc_info=True)
            return None, None, None
