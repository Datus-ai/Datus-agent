# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SqlSummaryAgenticNode implementation for SQL summary generation workflow.

This module provides a specialized implementation of AgenticNode focused on
SQL query summarization and classification with support for filesystem tools,
generation tools, and hooks.
"""

from typing import AsyncGenerator, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput, SqlSummaryNodeResult
from datus.tools.filesystem_tools.filesystem_tool import FilesystemFuncTool
from datus.tools.generation_tools import GenerationTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SqlSummaryAgenticNode(AgenticNode):
    """
    SQL summary generation agentic node with enhanced configuration.

    This node provides specialized SQL query summarization and classification with:
    - Enhanced system prompt with template variables
    - Filesystem tools for file operations
    - Generation tools for SQL summary context preparation
    - Hooks support for custom behavior
    - Configurable tool sets
    - Session-based conversation management
    """

    def __init__(
        self,
        node_name: str,
        agent_config: Optional[AgentConfig] = None,
        max_turns: int = 30,
    ):
        """
        Initialize the SqlSummaryAgenticNode.

        Args:
            node_name: Name of the node configuration in agent.yml
            agent_config: Agent configuration
            max_turns: Maximum conversation turns per interaction
        """
        self.configured_node_name = node_name
        self.max_turns = max_turns

        # Call parent constructor first to set up node_config
        super().__init__(
            tools=[],
            mcp_servers={},
            agent_config=agent_config,
        )

        # Setup tools based on configuration
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.generation_tools: Optional[GenerationTools] = None
        self.hooks = None
        self.setup_tools()

        # Debug: log hooks status after setup
        logger.info(f"Hooks after setup: {self.hooks} (type: {type(self.hooks)})")

    def get_node_name(self) -> str:
        """
        Get the configured node name for this SQL summary agentic node.

        Returns:
            The configured node name from agent.yml
        """
        return self.configured_node_name

    def setup_tools(self):
        """Setup tools based on configuration."""
        if not self.agent_config:
            return

        self.tools = []
        config_value = self.node_config.get("tools", "")
        if not config_value:
            return

        tool_patterns = [p.strip() for p in config_value.split(",") if p.strip()]
        for pattern in tool_patterns:
            self._setup_tool_pattern(pattern)

        logger.info(f"Setup {len(self.tools)} tools: {[tool.name for tool in self.tools]}")

        # Setup hooks after tools are configured
        self._setup_hooks()

    def _setup_filesystem_tools(self):
        """Setup filesystem tools."""
        try:
            root_path = self._resolve_workspace_root()
            self.filesystem_func_tool = FilesystemFuncTool(root_path=root_path)
            self.tools.extend(self.filesystem_func_tool.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup filesystem tools: {e}")

    def _setup_generation_tools(self):
        """Setup generation tools."""
        try:
            self.generation_tools = GenerationTools(self.agent_config)
            self.tools.extend(self.generation_tools.available_tools())
        except Exception as e:
            logger.error(f"Failed to setup generation tools: {e}")

    def _setup_hooks(self):
        """Setup hooks if configured."""
        hooks_config = self.node_config.get("hooks", "")
        logger.info(f"Hooks config: {hooks_config}, node_config: {self.node_config}")
        if not hooks_config:
            return

        try:
            # Import hooks module
            if hooks_config == "generation_hooks":
                from rich.console import Console

                from datus.cli.generation_hooks import GenerationHooks

                console = Console()
                self.hooks = GenerationHooks(console=console, agent_config=self.agent_config)
                logger.info(f"Setup hooks: {hooks_config}")
            else:
                logger.warning(f"Unknown hooks configuration: {hooks_config}")

        except Exception as e:
            logger.error(f"Failed to setup hooks '{hooks_config}': {e}")

    def _setup_tool_pattern(self, pattern: str):
        """Setup tools based on pattern."""
        try:
            # Handle wildcard patterns (e.g., "generation_tools.*")
            if pattern.endswith(".*"):
                base_type = pattern[:-2]  # Remove ".*"
                if base_type == "filesystem_tools":
                    self._setup_filesystem_tools()
                elif base_type == "generation_tools":
                    self._setup_generation_tools()
                else:
                    logger.warning(f"Unknown tool type: {base_type}")

            # Handle exact type patterns
            elif pattern == "filesystem_tools":
                self._setup_filesystem_tools()
            elif pattern == "generation_tools":
                self._setup_generation_tools()

            # Handle specific method patterns (e.g., "generation_tools.prepare_sql_summary_context")
            elif "." in pattern:
                tool_type, method_name = pattern.split(".", 1)
                self._setup_specific_tool_method(tool_type, method_name)

            else:
                logger.warning(f"Unknown tool pattern: {pattern}")

        except Exception as e:
            logger.error(f"Failed to setup tool pattern '{pattern}': {e}")

    def _setup_specific_tool_method(self, tool_type: str, method_name: str):
        """Setup a specific tool method."""
        try:
            if tool_type == "generation_tools":
                if not hasattr(self, "generation_tools") or not self.generation_tools:
                    self.generation_tools = GenerationTools(self.agent_config)
                tool_instance = self.generation_tools
            elif tool_type == "filesystem_tools":
                if not hasattr(self, "filesystem_func_tool") or not self.filesystem_func_tool:
                    root_path = self._resolve_workspace_root()
                    self.filesystem_func_tool = FilesystemFuncTool(root_path=root_path)
                tool_instance = self.filesystem_func_tool
            else:
                logger.warning(f"Unknown tool type: {tool_type}")
                return

            if hasattr(tool_instance, method_name):
                method = getattr(tool_instance, method_name)
                from datus.tools.tools import trans_to_function_tool

                self.tools.append(trans_to_function_tool(method))
                logger.debug(f"Added specific tool method: {tool_type}.{method_name}")
            else:
                logger.warning(f"Method '{method_name}' not found in {tool_type}")
        except Exception as e:
            logger.error(f"Failed to setup {tool_type}.{method_name}: {e}")

    def _prepare_template_context(self, user_input: SqlSummaryNodeInput) -> dict:
        """
        Prepare template context variables for the SQL summary generation template.

        Args:
            user_input: User input

        Returns:
            Dictionary of template variables
        """
        context = {}

        # Tool detection flags
        context["has_filesystem_tools"] = bool(self.filesystem_func_tool)
        context["has_generation_tools"] = bool(self.generation_tools)

        # Tool name lists for template display
        context["native_tools"] = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"

        # Add rules from configuration
        context["rules"] = self.node_config.get("rules", [])

        # Add agent description from configuration or input
        context["agent_description"] = user_input.agent_description or self.node_config.get("agent_description", "")

        # Add namespace and workspace info
        if self.agent_config:
            context["namespace"] = getattr(self.agent_config, "current_namespace", None)
            context["workspace_root"] = self._resolve_workspace_root()

        logger.debug(f"Prepared template context: {context}")
        return context

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        prompt_version: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        """
        Get the system prompt for this SQL summary node using enhanced template context.

        Args:
            conversation_summary: Optional summary from previous conversation compact
            prompt_version: Optional prompt version to use, overrides agent config version
            template_context: Optional template context variables

        Returns:
            System prompt string loaded from the template
        """
        # Get prompt version from parameter, fallback to node config, then agent config
        version = prompt_version
        if version is None:
            version = self.node_config.get("prompt_version")
        if version is None and self.agent_config and hasattr(self.agent_config, "prompt_version"):
            version = self.agent_config.prompt_version

        # Use shared workspace_root resolution logic
        root_path = self._resolve_workspace_root()

        # Construct template name: {system_prompt}_system or fallback to {node_name}_system
        system_prompt_name = self.node_config.get("system_prompt")
        if system_prompt_name:
            template_name = f"{system_prompt_name}_system"
        else:
            template_name = f"{self.get_node_name()}_system"

        try:
            # Prepare template variables
            template_vars = {
                "agent_config": self.agent_config,
                "namespace": getattr(self.agent_config, "current_namespace", None) if self.agent_config else None,
                "workspace_root": root_path,
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
        self, user_input: SqlSummaryNodeInput, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the SQL summary node interaction with streaming support.

        Args:
            user_input: Customized input containing user message and SQL context
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

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
            # Check for auto-compact before session creation to ensure fresh context
            await self._auto_compact()

            # Get or create session and any available summary
            session, conversation_summary = self._get_or_create_session()

            # Prepare enhanced template context
            template_context = self._prepare_template_context(user_input)

            # Get system instruction from template with enhanced context
            prompt_version = user_input.prompt_version or self.node_config.get("prompt_version")
            system_instruction = self._get_system_prompt(conversation_summary, prompt_version, template_context)

            # Add context to user message if provided
            enhanced_message = user_input.user_message
            enhanced_parts = []

            # Add SQL query context if provided
            if user_input.sql_query:
                enhanced_parts.append(f"SQL Query:\n```sql\n{user_input.sql_query}\n```")

            if user_input.comment:
                enhanced_parts.append(f"Comment: {user_input.comment}")

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
                messages="Generating SQL summary with tools...",
                input_data={"prompt": enhanced_message, "system": system_instruction},
                status=ActionStatus.PROCESSING,
            )
            action_history_manager.add_action(assistant_action)
            yield assistant_action

            logger.debug(f"Tools available: {len(self.tools)} tools - {[tool.name for tool in self.tools]}")
            logger.info(f"Passing hooks to model: {self.hooks} (type: {type(self.hooks)})")

            # Initialize response collection variables
            response_content = ""
            sql_summary_file = None
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
                hooks=self.hooks,
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

            # Extract sql_summary_file and output from the final response_content
            sql_summary_file, extracted_output = self._extract_sql_summary_and_output_from_response(
                {"content": response_content}
            )
            if extracted_output:
                response_content = extracted_output

            logger.debug(f"Final response_content: '{response_content}' (length: {len(response_content)})")

            # Extract token usage from final actions
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

            # Create final result
            result = SqlSummaryNodeResult(
                success=True,
                response=response_content,
                sql_summary_file=sql_summary_file,
                tokens_used=int(tokens_used),
            )

            # Add to internal actions list
            self.actions.extend(action_history_manager.get_actions())

            # Create final action
            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="sql_summary_response",
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
            error_result = SqlSummaryNodeResult(
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

    def _extract_sql_summary_and_output_from_response(self, output: dict) -> tuple[Optional[str], Optional[str]]:
        """
        Extract sql_summary_file and formatted output from model response.

        Per prompt template requirements, LLM should return JSON format:
        {"sql_summary_file": "path", "output": "markdown text"}

        Args:
            output: Output dictionary from model generation

        Returns:
            Tuple of (sql_summary_file, output_string) - both can be None if not found
        """
        try:
            from datus.utils.json_utils import strip_json_str

            content = output.get("content", "")
            logger.info(f"extract_sql_summary_and_output_from_final_resp: {content} (type: {type(content)})")

            # Case 1: content is already a dict (most common)
            if isinstance(content, dict):
                sql_summary_file = content.get("sql_summary_file")
                output_text = content.get("output")
                if sql_summary_file or output_text:
                    logger.debug(f"Extracted from dict: sql_summary_file={sql_summary_file}")
                    return sql_summary_file, output_text
                else:
                    logger.warning(f"Dict format but missing expected keys: {content.keys()}")

            # Case 2: content is a JSON string (possibly wrapped in markdown code blocks)
            elif isinstance(content, str) and content.strip():
                # Use strip_json_str to handle markdown code blocks and extract JSON
                cleaned_json = strip_json_str(content)
                if cleaned_json:
                    try:
                        import json_repair

                        parsed = json_repair.loads(cleaned_json)
                        if isinstance(parsed, dict):
                            sql_summary_file = parsed.get("sql_summary_file")
                            output_text = parsed.get("output")
                            if sql_summary_file or output_text:
                                logger.debug(f"Extracted from JSON string: sql_summary_file={sql_summary_file}")
                                return sql_summary_file, output_text
                            else:
                                logger.warning(f"Parsed JSON but missing expected keys: {parsed.keys()}")
                    except Exception as e:
                        logger.warning(f"Failed to parse cleaned JSON: {e}. Cleaned content: {cleaned_json[:200]}")

            logger.warning(f"Could not extract sql_summary_file from response. Content type: {type(content)}")
            return None, None

        except Exception as e:
            logger.error(f"Unexpected error extracting sql_summary_file: {e}", exc_info=True)
            return None, None
