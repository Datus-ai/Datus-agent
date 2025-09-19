"""
CustomizedAgenticNode implementation for flexible configuration-based interactions.

This module provides a concrete implementation of AgenticNode that supports
flexible configuration through agent.yml for system prompts, tools, MCP servers,
and custom rules.
"""

import os
from typing import AsyncGenerator, Dict, Optional

from agents.mcp import MCPServerStdio

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.customized_agentic_node_models import CustomizedNodeInput, CustomizedNodeResult
from datus.tools.context_search import ContextSearchTools
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.mcp_server import MCPServer
from datus.tools.tools import DBFuncTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class CustomizedAgenticNode(AgenticNode):
    """
    Customized agentic node with flexible configuration support.

    This node provides configurable capabilities with:
    - Custom system prompts and prompt versions
    - Configurable tool sets (db_tools, filesystem_tools, etc.)
    - MCP server integration (MetricFlow, etc.)
    - Custom rules and language support
    - Session-based conversation management
    """

    def __init__(
        self,
        node_name: str,
        agent_config: Optional[AgentConfig] = None,
        max_turns: int = 30,
    ):
        """
        Initialize the CustomizedAgenticNode.

        Args:
            node_name: Name of the node configuration in agent.yml
            agent_config: Agent configuration
            max_turns: Maximum conversation turns per interaction
        """
        self.configured_node_name = node_name
        self.max_turns = max_turns

        # Parse node configuration from agent.yml
        self.node_config = self._parse_node_config(agent_config, node_name)

        # Initialize MCP servers based on configuration
        self.mcp_servers = self._setup_mcp_servers(agent_config)

        super().__init__(
            tools=[],
            mcp_servers=self.mcp_servers,
            agent_config=agent_config,
        )

        # Setup tools based on configuration
        self.db_func_tool: Optional[DBFuncTool] = None
        self.context_search_tools: Optional[ContextSearchTools] = None
        self.setup_tools()

    def _parse_node_config(self, agent_config: Optional[AgentConfig], node_name: str) -> dict:
        """
        Parse node configuration from agent.yml.

        Args:
            agent_config: Agent configuration
            node_name: Name of the node configuration

        Returns:
            Dictionary containing node configuration
        """
        if not agent_config or not hasattr(agent_config, "nodes"):
            return {}

        nodes_config = agent_config.nodes
        if node_name not in nodes_config:
            logger.warning(f"Node configuration '{node_name}' not found in agent.yml")
            return {}

        node_config = nodes_config[node_name]

        # Extract configuration attributes
        config = {}

        # Basic node config attributes
        if hasattr(node_config, "model"):
            config["model"] = node_config.model
        if hasattr(node_config, "system_prompt"):
            config["system_prompt"] = node_config.system_prompt
        if hasattr(node_config, "prompt_version"):
            config["prompt_version"] = node_config.prompt_version
        if hasattr(node_config, "prompt_language"):
            config["prompt_language"] = node_config.prompt_language
        if hasattr(node_config, "tools"):
            config["tools"] = node_config.tools
        if hasattr(node_config, "mcp"):
            config["mcp"] = node_config.mcp
        if hasattr(node_config, "rules"):
            config["rules"] = node_config.rules
        if hasattr(node_config, "max_turns"):
            config["max_turns"] = node_config.max_turns
            self.max_turns = node_config.max_turns

        logger.info(f"Parsed node configuration for '{node_name}': {config}")
        return config

    def get_node_name(self) -> str:
        """
        Get the configured node name for this customized agentic node.

        Returns:
            The configured node name (e.g., "gen_metrics")
        """
        return self.configured_node_name

    def setup_tools(self):
        """Setup tools based on configuration."""
        if not self.agent_config:
            logger.warning("No agent config available, skipping tool setup")
            return

        tools_config = self.node_config.get("tools", "")
        if not tools_config:
            logger.info("No tools configured for this node")
            self.tools = []
            return

        self.tools = []

        # Parse comma-separated tool names
        tool_names = [name.strip() for name in tools_config.split(",") if name.strip()]

        for tool_name in tool_names:
            if tool_name == "db_tools":
                self._setup_db_tools()
            elif tool_name == "filesystem_tools":
                # Filesystem tools are handled via MCP servers
                logger.debug("Filesystem tools configured via MCP servers")
            elif tool_name == "context_search_tools":
                self._setup_context_search_tools()
            else:
                logger.warning(f"Unknown tool name: {tool_name}")

        logger.info(f"Setup {len(self.tools)} tools: {[tool.name for tool in self.tools]}")

    def _setup_db_tools(self):
        """Setup database tools."""
        try:
            db_manager = db_manager_instance(self.agent_config.namespaces)
            conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
            self.db_func_tool = DBFuncTool(conn)
            self.tools.extend(self.db_func_tool.available_tools())
            logger.debug(f"Added {len(self.db_func_tool.available_tools())} database tools")
        except Exception as e:
            logger.error(f"Failed to setup database tools: {e}")

    def _setup_context_search_tools(self):
        """Setup context search tools."""
        try:
            self.context_search_tools = ContextSearchTools(self.agent_config)
            self.tools.extend(self.context_search_tools.available_tools())
            logger.debug(f"Added {len(self.context_search_tools.available_tools())} context search tools")
        except Exception as e:
            logger.error(f"Failed to setup context search tools: {e}")

    def _setup_mcp_servers(self, agent_config: Optional[AgentConfig] = None) -> Dict[str, MCPServerStdio]:
        """
        Set up MCP servers based on configuration.

        Args:
            agent_config: Agent configuration

        Returns:
            Dictionary of MCP servers
        """
        mcp_servers = {}

        try:
            # Add filesystem MCP server (always available)
            root_path = "."
            if agent_config and hasattr(agent_config, "workspace_root"):
                workspace_root = agent_config.workspace_root
                if workspace_root is not None:
                    root_path = workspace_root

            # Handle relative vs absolute paths
            if root_path and os.path.isabs(root_path):
                filesystem_path = root_path
            else:
                filesystem_path = os.path.join(os.getcwd(), root_path)

            filesystem_server = MCPServer.get_filesystem_mcp_server(path=filesystem_path)
            if filesystem_server:
                mcp_servers["filesystem"] = filesystem_server
                logger.info(f"Added filesystem MCP server at path: {filesystem_path}")
            else:
                logger.warning(f"Failed to create filesystem MCP server for path: {filesystem_path}")

            # Add configured MCP servers
            mcp_config = self.node_config.get("mcp", "")
            if mcp_config:
                mcp_names = [name.strip() for name in mcp_config.split(",") if name.strip()]
                for mcp_name in mcp_names:
                    if mcp_name.startswith("metricflow_mcp"):
                        # Handle MetricFlow MCP server
                        try:
                            metricflow_server = MCPServer.get_metricflow_mcp_server()
                            if metricflow_server:
                                mcp_servers[mcp_name] = metricflow_server
                                logger.info(f"Added MetricFlow MCP server: {mcp_name}")
                            else:
                                logger.warning(f"Failed to create MetricFlow MCP server: {mcp_name}")
                        except Exception as e:
                            logger.error(f"Error setting up MetricFlow MCP server {mcp_name}: {e}")
                    else:
                        logger.warning(f"Unknown MCP server type: {mcp_name}")

        except Exception as e:
            logger.error(f"Error setting up MCP servers: {e}")

        return mcp_servers

    async def execute_stream(
        self, user_input: CustomizedNodeInput, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the customized node interaction with streaming support.

        Args:
            user_input: Customized input containing user message and context
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        # Create initial action
        action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type="customized_interaction",
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

            # Get system instruction from template, passing summary and prompt version if available
            prompt_version = user_input.prompt_version or self.node_config.get("prompt_version")
            system_instruction = self._get_system_prompt(conversation_summary, prompt_version)

            # Enhance system prompt with custom rules if configured
            enhanced_system_instruction = self._enhance_system_prompt(system_instruction)

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
                enhanced_message = f"{'\n\n'.join(enhanced_parts)}\n\nUser question: {user_input.user_message}"

            # Execute with streaming
            response_content = ""
            sql_content = None
            tokens_used = 0
            last_successful_output = None

            # Create assistant action for processing
            assistant_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="llm_generation",
                messages="Generating response with tools...",
                input_data={"prompt": enhanced_message, "system": enhanced_system_instruction},
                status=ActionStatus.PROCESSING,
            )
            action_history_manager.add_action(assistant_action)
            yield assistant_action

            # Stream response using the model's generate_with_tools_stream
            async for stream_action in self.model.generate_with_tools_stream(
                prompt=enhanced_message,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
                instruction=enhanced_system_instruction,
                max_turns=self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
            ):
                yield stream_action

                # Collect response content from successful actions
                if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        last_successful_output = stream_action.output
                        # Look for content in various possible fields
                        response_content = (
                            stream_action.output.get("content", "")
                            or stream_action.output.get("response", "")
                            or response_content
                        )

            # If we still don't have response_content, check the last successful output
            if not response_content and last_successful_output:
                logger.debug(f"Trying to extract response from last_successful_output: {last_successful_output}")
                # Try different fields that might contain the response
                response_content = (
                    last_successful_output.get("content", "")
                    or last_successful_output.get("text", "")
                    or last_successful_output.get("response", "")
                    or str(last_successful_output)  # Fallback to string representation
                )

            # Extract SQL and output from the final response_content
            sql_content, extracted_output = self._extract_sql_and_output_from_response({"content": response_content})
            if extracted_output:
                response_content = extracted_output

            logger.debug(f"Final response_content: '{response_content}' (length: {len(response_content)})")

            # Extract token usage from final actions
            final_actions = action_history_manager.get_actions()
            tokens_used = 0

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
            result = CustomizedNodeResult(
                success=True,
                response=response_content,
                sql=sql_content,
                tokens_used=int(tokens_used),
            )

            # Add to internal actions list
            self.actions.extend(action_history_manager.get_actions())

            # Create final action
            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="customized_response",
                messages="Customized interaction completed successfully",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except Exception as e:
            logger.error(f"Customized execution error: {e}")

            # Create error result
            error_result = CustomizedNodeResult(
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
                messages=f"Customized interaction failed: {str(e)}",
                input_data=user_input.model_dump(),
                output_data=error_result.model_dump(),
                status=ActionStatus.FAILED,
            )
            action_history_manager.add_action(error_action)
            yield error_action

    def _enhance_system_prompt(self, base_prompt: str) -> str:
        """
        Enhance the system prompt with configured rules and context.

        Args:
            base_prompt: Base system prompt from template

        Returns:
            Enhanced system prompt with additional context
        """
        enhanced_prompt = base_prompt

        # Add agent description if configured
        agent_description = self.node_config.get("agent_description", "")
        if agent_description:
            enhanced_prompt = enhanced_prompt.replace("{{ agent_description }}", agent_description)

        # Add rules if configured
        rules = self.node_config.get("rules", [])
        if rules:
            rules_text = "\n".join([f"   * {rule}" for rule in rules])
            enhanced_prompt = enhanced_prompt.replace(
                "{% for rule in rules %} \n   * {{rule}}\n  {% endfor %}", rules_text
            )

        return enhanced_prompt

    def _extract_sql_and_output_from_response(self, output: dict) -> tuple[Optional[str], Optional[str]]:
        """
        Extract SQL content and formatted output from model response.

        Args:
            output: Output dictionary from model generation

        Returns:
            Tuple of (sql_string, output_string) - both can be None if not found
        """
        try:
            import ast
            import json

            from datus.utils.json_utils import strip_json_str

            content = output.get("content", "")
            logger.info(f"extract_sql_and_output_from_final_resp: {content}")

            # Handle string representation of dictionary with raw_output
            if isinstance(content, str) and content.strip().startswith("{'"):
                parsed_dict = None

                # Try ast.literal_eval first (most reliable for proper Python dict strings)
                try:
                    parsed_dict = ast.literal_eval(content)
                except (ValueError, SyntaxError) as e:
                    logger.debug(f"ast.literal_eval failed: {e}, trying alternative parsing")

                    # Alternative approach: manually extract raw_output using regex
                    # This handles cases where the dict contains values that can't be parsed by ast.literal_eval
                    import re

                    # More robust pattern that handles the actual structure in the content
                    # Look for 'raw_output': ' and then capture everything until the final '} pattern
                    raw_output_pattern = r"'raw_output':\s*'(.+?)'(?:\s*})?$"
                    match = re.search(raw_output_pattern, content, re.DOTALL)

                    if match:
                        raw_output_value = match.group(1)
                        # Unescape the extracted value
                        raw_output_value = raw_output_value.replace("\\'", "'").replace("\\\\", "\\")
                        parsed_dict = {"raw_output": raw_output_value}
                        logger.debug("Extracted raw_output using regex pattern")
                    else:
                        logger.debug("Could not extract raw_output using regex")

                if isinstance(parsed_dict, dict) and "raw_output" in parsed_dict:
                    try:
                        # Use strip_json_str to clean raw_output before parsing JSON
                        cleaned_raw_output = strip_json_str(parsed_dict["raw_output"])

                        # Try with json_repair for better handling of malformed JSON
                        import json_repair

                        try:
                            json_content = json_repair.loads(cleaned_raw_output)
                        except Exception:
                            # Last resort: try regular json.loads
                            json_content = json.loads(cleaned_raw_output)

                        # Ensure json_content is a dict before calling get()
                        if isinstance(json_content, dict):
                            sql = json_content.get("sql")
                            output_text = json_content.get("output")
                        else:
                            return None, None

                        # Unescape output content
                        if output_text:
                            output_text = output_text.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")

                        return sql, output_text
                    except (ValueError, SyntaxError, json.JSONDecodeError) as e:
                        logger.debug(f"Failed to parse raw_output JSON: {e}")

            return None, None

        except Exception as e:
            logger.warning(f"Failed to extract SQL and output from response: {e}")
            return None, None

    def _extract_sql_from_response(self, output: dict) -> Optional[str]:
        """
        Extract SQL content from model response (backward compatibility).

        Args:
            output: Output dictionary from model generation

        Returns:
            SQL string if found, None otherwise
        """
        sql_content, _ = self._extract_sql_and_output_from_response(output)
        return sql_content
