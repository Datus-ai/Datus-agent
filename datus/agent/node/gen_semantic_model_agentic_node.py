# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenSemanticModelAgenticNode implementation for semantic model generation.

This module provides a specialized implementation of AgenticNode focused on
semantic model generation with support for filesystem tools, generation tools,
database tools, hooks, and metricflow MCP server integration.
"""

from typing import Any, AsyncGenerator, Dict, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput, SemanticNodeResult
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.tools.func_tool.semantic_tools import SemanticModelTools
from datus.tools.mcp_tools.mcp_server import MCPServer
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

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

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
    ):
        """
        Initialize the GenSemanticModelAgenticNode.

        Args:
            agent_config: Agent configuration
            execution_mode: Execution mode - "interactive" (default) or "workflow"
        """
        self.execution_mode = execution_mode

        # Get max_turns from agentic_nodes configuration, default to 30
        self.max_turns = 30
        if agent_config and hasattr(agent_config, "agentic_nodes") and self.NODE_NAME in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[self.NODE_NAME]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 30)

        path_manager = get_path_manager()
        self.semantic_model_dir = str(path_manager.semantic_model_path(agent_config.current_namespace))

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
            mcp_servers={},  # Initialize empty, will setup after parent init
        )

        # Initialize MCP servers
        self.mcp_servers = self._setup_mcp_servers()

        logger.debug(
            f"GenSemanticModelAgenticNode final mcp_servers: "
            f"{len(self.mcp_servers)} servers - {list(self.mcp_servers.keys())}"
        )

        # Setup tools
        self.db_func_tool: Optional[DBFuncTool] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.generation_tools: Optional[GenerationTools] = None
        self.semantic_model_tools: Optional[SemanticModelTools] = None
        self.hooks = None
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
        self._setup_semantic_tools()
        self._setup_generation_tools()
        self._setup_filesystem_tools()

        logger.debug(f"Setup {len(self.tools)} tools for {self.NODE_NAME}: {[tool.name for tool in self.tools]}")

        # Setup hooks (only in interactive mode)
        if self.execution_mode == "interactive":
            self._setup_hooks()

    def _setup_db_tools(self):
        """Setup database tools."""
        try:
            db_manager = db_manager_instance(self.agent_config.namespaces)
            conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
            self.db_func_tool = DBFuncTool(
                conn,
                agent_config=self.agent_config,
            )
            # Add standard database tools
            self.tools.extend(self.db_func_tool.available_tools())
            logger.debug("Added database tools from DBFuncTool")
        except Exception as e:
            logger.error(f"Failed to setup database tools: {e}")

    def _setup_semantic_tools(self):
        """Setup semantic model specific tools."""
        try:
            if not self.db_func_tool:
                logger.warning("DBFuncTool not initialized, skipping semantic tools setup")
                return

            self.semantic_model_tools = SemanticModelTools(self.db_func_tool)
            self.tools.extend(self.semantic_model_tools.available_tools())
            logger.debug("Added semantic model tools from SemanticModelTools")
        except Exception as e:
            logger.error(f"Failed to setup semantic tools: {e}")

    def _setup_filesystem_tools(self):
        """Setup filesystem tools."""
        try:
            from datus.tools.func_tool import trans_to_function_tool

            self.filesystem_func_tool = FilesystemFuncTool(root_path=self.semantic_model_dir)

            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.read_file))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.read_multiple_files))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.write_file))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.edit_file))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.list_directory))
            logger.debug(
                "Added filesystem tools: read_file, read_multiple_files, write_file, edit_file, list_directory"
            )
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _setup_generation_tools(self):
        """Setup generation tools."""
        try:
            from datus.tools.func_tool import trans_to_function_tool

            self.generation_tools = GenerationTools(self.agent_config)

            self.tools.append(trans_to_function_tool(self.generation_tools.check_semantic_object_exists))
            self.tools.append(trans_to_function_tool(self.generation_tools.end_semantic_model_generation))
            logger.debug("Added tools: check_semantic_object_exists, end_semantic_model_generation")

        except Exception as e:
            logger.error(f"Failed to setup generation tools: {e}")

    def _setup_hooks(self):
        """Setup hooks for interactive mode."""
        try:
            from rich.console import Console

            console = Console()
            self.hooks = GenerationHooks(console=console, agent_config=self.agent_config)
            logger.info("Setup hooks: generation_hooks")
        except Exception as e:
            logger.error(f"Failed to setup generation_hooks: {e}")

    def _setup_mcp_servers(self) -> Dict[str, Any]:
        """Set up MCP servers (metricflow_mcp)."""
        mcp_servers = {}

        try:
            if not self.agent_config:
                logger.warning("Agent config not available for metricflow MCP setup")
                return mcp_servers

            # Get current database config
            db_config = self.agent_config.current_db_config()
            if not db_config:
                logger.warning("Database config not found")
                return mcp_servers

            metricflow_server = MCPServer.get_metricflow_mcp_server(namespace=self.agent_config.current_namespace)
            if metricflow_server:
                mcp_servers["metricflow_mcp"] = metricflow_server
                logger.info(f"Setup metricflow_mcp MCP server for database: {db_config.database}")
            else:
                logger.warning(f"Failed to create metricflow MCP server for db_config: {db_config}")
        except Exception as e:
            logger.error(f"Failed to setup metricflow_mcp: {e}")

        logger.info(f"Setup {len(mcp_servers)} MCP servers: {list(mcp_servers.keys())}")

        return mcp_servers

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
        context = {}

        # Tool name lists for template display
        context["native_tools"] = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"
        context["mcp_tools"] = ", ".join(list(self.mcp_servers.keys())) if self.mcp_servers else "None"
        context["semantic_model_dir"] = self.semantic_model_dir

        logger.debug(f"Prepared template context: {context}")
        return context

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        """
        Get the system prompt for semantic model generation using enhanced template context.

        Args:
            conversation_summary: Optional summary from previous conversation compact
            template_context: Optional template context variables

        Returns:
            System prompt string loaded from the template
        """
        # Hardcoded prompt version
        version = "1.0"

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
            from datus.prompts.prompt_manager import prompt_manager

            return prompt_manager.render_template(template_name=template_name, version=version, **template_vars)

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
        Execute the semantic model generation with streaming support.

        Args:
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        # Get input from self.input
        if self.input is None:
            raise ValueError("Semantic input not set. Set self.input before calling execute_stream.")

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
                context_part_str = f'Context: {", ".join(context_parts)}'
                enhanced_parts.append(context_part_str)

            if enhanced_parts:
                enhanced_message = f"{'\\n\\n'.join(enhanced_parts)}\\n\\nUser question: {user_input.user_message}"

            # Create assistant action for processing
            assistant_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="llm_generation",
                messages="Generating response with tools...",
                input_data={"prompt": enhanced_message, "system": system_instruction},
                status=ActionStatus.PROCESSING,
            )
            action_history_manager.add_action(assistant_action)
            yield assistant_action

            logger.debug(f"Tools available : {len(self.tools)} tools - {[tool.name for tool in self.tools]}")
            logger.debug(f"MCP servers available : {len(self.mcp_servers)} servers - {list(self.mcp_servers.keys())}")
            logger.debug(f"Passing hooks to model: {self.hooks} (type: {type(self.hooks)})")

            # Initialize response collection variables
            response_content = ""
            semantic_model_files = []
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
                hooks=self.hooks if self.execution_mode == "interactive" else None,
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
                logger.debug(f"Trying to extract response from last_successful_output: {last_successful_output}")
                # Try different fields that might contain the response
                raw_output = last_successful_output.get("raw_output", "")
                if isinstance(raw_output, dict):
                    response_content = raw_output
                elif raw_output:
                    response_content = raw_output
                else:
                    response_content = str(last_successful_output)  # Fallback to string representation

            # Extract semantic_model_files and output from the final response_content
            semantic_model_files, extracted_output = self._extract_semantic_model_and_output_from_response(
                {"content": response_content}
            )
            if extracted_output:
                response_content = extracted_output

            logger.debug(f"Final response_content: '{response_content}' (length: {len(response_content)})")

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
                                conversation_tokens = usage_info.get("total_tokens", 0)
                                if conversation_tokens > 0:
                                    # Add this conversation's tokens to the session
                                    self._add_session_tokens(conversation_tokens)
                                    tokens_used = conversation_tokens
                                    logger.info(f"Added {conversation_tokens} tokens to session")
                                    break
                                else:
                                    logger.warning(f"no usage token found in this action {action.messages}")

            # Auto-save to database in workflow mode (support multiple files)
            if self.execution_mode == "workflow" and semantic_model_files:
                try:
                    for semantic_model_file in semantic_model_files:
                        self._save_to_db(
                            semantic_model_file,
                            catalog=user_input.catalog,
                            database=user_input.database,
                            db_schema=user_input.db_schema,
                        )
                    logger.info(f"Auto-saved {len(semantic_model_files)} semantic models to database")
                except Exception as e:
                    logger.error(f"Failed to auto-save to database: {e}")

            # Create final result
            result = SemanticNodeResult(
                success=True,
                response=response_content,
                semantic_models=semantic_model_files,
                tokens_used=int(tokens_used),
            )

            # Add to internal actions list
            self.actions.extend(action_history_manager.get_actions())

            # Create final action
            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="semantic_response",
                messages=f"{self.get_node_name()} interaction completed successfully",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

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

    def _save_to_db(self, semantic_model_file: str, catalog=None, database=None, db_schema=None):
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

            # Construct full path
            full_path = os.path.join(self.semantic_model_dir, semantic_model_file)

            if not os.path.exists(full_path):
                logger.warning(f"Semantic model file not found: {full_path}")
                return

            # Call static method to save to database
            # Deduplication is handled inside _sync_semantic_to_db
            result = GenerationHooks._sync_semantic_to_db(
                full_path, self.agent_config, catalog=catalog, database=database, schema=db_schema
            )

            if result.get("success"):
                logger.info(f"Successfully saved to database: {result.get('message')}")
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Failed to save to database: {error}")

        except Exception as e:
            logger.error(f"Error saving to database: {e}", exc_info=True)
            raise
