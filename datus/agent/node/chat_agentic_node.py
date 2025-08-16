"""
ChatAgenticNode implementation for flexible CLI chat interactions.

This module provides a concrete implementation of AgenticNode specifically
designed for chat interactions with database and filesystem tool support.
"""

from typing import AsyncGenerator, Dict, Optional

from agents.mcp import MCPServerStdio

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput, ChatNodeResult
from datus.tools.mcp_server import MCPServer
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ChatAgenticNode(AgenticNode):
    """
    Chat-focused agentic node with database and filesystem tool support.

    This node provides flexible chat capabilities with:
    - Namespace-based database MCP server selection
    - Default filesystem MCP server
    - Streaming response generation
    - Session-based conversation management
    """

    def __init__(
        self,
        namespace: Optional[str] = None,
        agent_config: Optional[AgentConfig] = None,
        max_turns: int = 30,
    ):
        """
        Initialize the ChatAgenticNode.

        Args:
            namespace: Database namespace for MCP server selection
            agent_config: Agent configuration
            max_turns: Maximum conversation turns per interaction
        """
        self.namespace = namespace
        self.max_turns = max_turns

        # Initialize MCP servers based on namespace
        mcp_servers = self._setup_mcp_servers(namespace, agent_config)

        super().__init__(
            tools=[],  # Empty tools list initially
            mcp_servers=mcp_servers,
            agent_config=agent_config,
        )

    def _setup_mcp_servers(
        self, namespace: Optional[str], agent_config: Optional[AgentConfig]
    ) -> Dict[str, MCPServerStdio]:
        """
        Set up MCP servers based on namespace and configuration.

        Args:
            namespace: Database namespace for server selection
            agent_config: Agent configuration containing namespace definitions

        Returns:
            Dictionary of MCP servers
        """
        mcp_servers = {}

        try:
            # Add filesystem MCP server as default
            import os

            sqls_path = os.path.join(os.getcwd(), "sqls")
            filesystem_server = MCPServer.get_filesystem_mcp_server(path=sqls_path)
            if filesystem_server:
                mcp_servers["filesystem"] = filesystem_server
                logger.debug(f"Added filesystem MCP server with path: {sqls_path}")
            else:
                logger.warning(f"Failed to create filesystem MCP server for path: {sqls_path}")

            # Add database MCP server based on namespace
            if namespace and agent_config and namespace in agent_config.namespaces:
                namespace_config = agent_config.namespaces[namespace]

                # Get the first database config from the namespace
                if isinstance(namespace_config, dict):
                    for db_name, db_config in namespace_config.items():
                        try:
                            db_server = MCPServer.get_db_mcp_server(db_config)
                            if db_server:
                                mcp_servers[f"database_{db_name}"] = db_server
                                logger.debug(f"Added database MCP server for {db_name}")
                                break  # Use the first available database
                        except Exception as e:
                            logger.warning(f"Failed to initialize database MCP server for {db_name}: {e}")
                else:
                    # Single database configuration
                    try:
                        db_server = MCPServer.get_db_mcp_server(namespace_config)
                        if db_server:
                            mcp_servers[f"database_{namespace}"] = db_server
                            logger.debug(f"Added database MCP server for namespace {namespace}")
                    except Exception as e:
                        logger.warning(f"Failed to initialize database MCP server for namespace {namespace}: {e}")

        except Exception as e:
            logger.error(f"Error setting up MCP servers: {e}")

        return mcp_servers

    async def execute_stream(
        self, user_input: ChatNodeInput, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the chat interaction with streaming support.

        Args:
            user_input: Chat input containing user message and context
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        # Create initial action
        action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type="chat_interaction",
            messages=f"User: {user_input.user_message}",
            input_data=user_input.model_dump(),
            status=ActionStatus.PROCESSING,
        )
        action_history_manager.add_action(action)
        yield action

        try:
            # Get or create session
            session = self._get_or_create_session()

            # Get system instruction from template
            system_instruction = self.system_prompt

            # Add database context to user message if provided
            enhanced_message = user_input.user_message
            if user_input.catalog or user_input.database or user_input.db_schema:
                context_parts = []
                if user_input.catalog:
                    context_parts.append(f"catalog: {user_input.catalog}")
                if user_input.database:
                    context_parts.append(f"database: {user_input.database}")
                if user_input.db_schema:
                    context_parts.append(f"schema: {user_input.db_schema}")

                enhanced_message = f"Context: {', '.join(context_parts)}\n\nUser question: {user_input.user_message}"

            # Check for auto-compact
            await self._auto_compact()

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
                input_data={"prompt": enhanced_message, "system": system_instruction},
                status=ActionStatus.PROCESSING,
            )
            action_history_manager.add_action(assistant_action)
            yield assistant_action

            # Stream response using the model's generate_with_tools_stream
            async for stream_action in self.model.generate_with_tools_stream(
                prompt=enhanced_message,
                mcp_servers=self.mcp_servers,
                instruction=system_instruction,
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

            # Count tokens (simplified - would need actual implementation)
            if response_content:
                tokens_used = len(response_content.split()) * 1.3  # Rough estimation

            # Create final result
            result = ChatNodeResult(
                success=True,
                response=response_content,
                sql=sql_content,
                tokens_used=int(tokens_used),
            )

            # Update assistant action with success
            action_history_manager.update_action_by_id(
                assistant_action.action_id,
                status=ActionStatus.SUCCESS,
                output=result.model_dump(),
                messages=(
                    f"Generated response: {response_content[:100]}..."
                    if len(response_content) > 100
                    else response_content
                ),
            )

            # Add to internal actions list
            self.actions.extend(action_history_manager.get_actions())

            # Create final action
            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="chat_response",
                messages="Chat interaction completed successfully",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except Exception as e:
            logger.error(f"Chat execution error: {e}")

            # Create error result
            error_result = ChatNodeResult(
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
                messages=f"Chat interaction failed: {str(e)}",
                input_data=user_input.model_dump(),
                output_data=error_result.model_dump(),
                status=ActionStatus.FAILED,
            )
            action_history_manager.add_action(error_action)
            yield error_action

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
