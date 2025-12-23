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
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput, SqlSummaryNodeResult
from datus.tools.func_tool.filesystem_tool import FilesystemFuncTool
from datus.tools.func_tool.generation_tools import GenerationTools
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

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
        execution_mode: str = "interactive",
        build_mode: str = "incremental",
        subject_tree: Optional[list] = None,
    ):
        """
        Initialize the SqlSummaryAgenticNode.

        Args:
            node_name: Name of the node configuration in agent.yml (should be "gen_sql_summary")
            agent_config: Agent configuration
            execution_mode: Execution mode - "interactive" (default) or "workflow"
            build_mode: "overwrite" or "incremental" (default: "incremental")
            subject_tree: Optional predefined subject tree categories
        """
        self.configured_node_name = node_name
        self.execution_mode = execution_mode
        self.build_mode = build_mode
        self.subject_tree = subject_tree

        # Get max_turns from agentic_nodes configuration
        self.max_turns = 30
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[node_name]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 30)

        path_manager = get_path_manager()
        self.sql_summary_dir = str(path_manager.sql_summary_path(agent_config.current_namespace))

        from datus.configuration.node_type import NodeType

        node_type = NodeType.TYPE_SQL_SUMMARY

        # Call parent constructor first to set up node_config
        super().__init__(
            node_id="sql_summary_node",
            description="SQL summary generation node",
            node_type=node_type,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
        )

        # Initialize reference SQL storage for context queries
        from datus.storage.reference_sql.store import ReferenceSqlRAG

        self.reference_sql_rag = ReferenceSqlRAG(agent_config)

        # Setup tools based on hardcoded configuration
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.generation_tools: Optional[GenerationTools] = None
        self.hooks = None
        self.setup_tools()

        logger.info(f"Hooks after setup: {self.hooks} (type: {type(self.hooks)})")

    def get_node_name(self) -> str:
        """
        Get the configured node name for this SQL summary agentic node.

        Returns:
            The configured node name from agent.yml
        """
        return self.configured_node_name

    def setup_tools(self):
        """Setup tools based on hardcoded configuration."""
        if not self.agent_config:
            return

        self.tools = []

        # Hardcoded tool configuration: specific methods from generation_tools and filesystem_tools
        # tools: generation_tools.generate_sql_summary_id,
        # filesystem_tools.read_file, filesystem_tools.read_multiple_files, filesystem_tools.write_file,
        # filesystem_tools.edit_file, filesystem_tools.list_directory
        self._setup_specific_generation_tools()
        self._setup_specific_filesystem_tool()

        logger.info(
            f"Setup {len(self.tools)} tools for {self.configured_node_name}: {[tool.name for tool in self.tools]}"
        )

        # Setup hooks (only in interactive mode)
        if self.execution_mode == "interactive":
            self._setup_hooks()

    def _setup_specific_generation_tools(self):
        """Setup specific generation tools: generate_sql_summary_id."""
        try:
            from datus.tools.func_tool import trans_to_function_tool

            self.generation_tools = GenerationTools(self.agent_config)
            self.tools.append(trans_to_function_tool(self.generation_tools.generate_sql_summary_id))
        except Exception as e:
            logger.error(f"Failed to setup specific generation tools: {e}")

    def _setup_specific_filesystem_tool(self):
        """Setup specific filesystem tools"""
        try:
            from datus.tools.func_tool import trans_to_function_tool

            self.filesystem_func_tool = FilesystemFuncTool(root_path=self.sql_summary_dir)

            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.read_file))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.read_multiple_files))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.write_file))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.edit_file))
            self.tools.append(trans_to_function_tool(self.filesystem_func_tool.list_directory))
        except Exception as e:
            logger.error(f"Failed to setup specific filesystem tool: {e}")

    def _setup_hooks(self):
        """Setup hooks (hardcoded to generation_hooks)."""
        try:
            from rich.console import Console

            console = Console()
            self.hooks = GenerationHooks(console=console, agent_config=self.agent_config)
            logger.info("Setup hooks: generation_hooks")
        except Exception as e:
            logger.error(f"Failed to setup generation_hooks: {e}")

    def _get_existing_subject_trees(self) -> list:
        """
        Query existing subject_path values from reference SQL storage.

        Returns:
            List of unique subject_path values as List[str]
        """
        try:
            # Get all metrics with subject_path field
            subject_paths = sorted(self.reference_sql_rag.reference_sql_storage.get_subject_tree_flat())
            logger.debug(f"Found {len(subject_paths)} unique reference SQL subject_paths")
            return subject_paths
        except Exception as e:
            logger.error(f"Error getting existing subject_paths: {e}")
            return []

    def _get_similar_sqls(self, query_text: str, top_n: int = 5) -> list:
        """
        Find similar reference SQLs based on query text.

        Args:
            query_text: Text to use for similarity search (comment or SQL)
            top_n: Number of similar results to return

        Returns:
            List of similar reference SQLs with fields: name, subject_tree, tags, comment, summary
        """
        try:
            if not query_text:
                return []

            # Search using vector similarity on summary field
            similar_items = self.reference_sql_rag.search_reference_sql(query_text=query_text, top_n=top_n)

            # Extract relevant fields and format results
            results = []
            for item in similar_items:
                # Get subject_path from item
                subject_path = item.get("subject_path", [])
                # Format as string for display
                subject_tree = "/".join(subject_path) if subject_path else ""

                results.append(
                    {
                        "name": item.get("name", ""),
                        "subject_tree": subject_tree,
                        "tags": item.get("tags", ""),
                        "comment": item.get("comment", ""),
                        "summary": item.get("summary", ""),
                    }
                )

            logger.debug(f"Found {len(results)} similar reference SQLs")
            return results

        except Exception as e:
            logger.error(f"Error getting similar reference SQLs: {e}")
            return []

    def _prepare_template_context(self, user_input: SqlSummaryNodeInput) -> dict:
        """
        Prepare template context variables for the SQL summary generation template.

        Args:
            user_input: User input

        Returns:
            Dictionary of template variables
        """
        context = {}

        context["native_tools"] = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"
        context["sql_summary_dir"] = self.sql_summary_dir

        # Handle subject_tree context based on whether predefined or query from storage
        if self.subject_tree:
            # Predefined mode: use provided subject_tree
            context["has_subject_tree"] = True
            context["subject_tree"] = self.subject_tree
        else:
            # Learning mode: query existing subject_trees from LanceDB
            context["has_subject_tree"] = False
            existing_trees = self._get_existing_subject_trees()
            context["existing_subject_trees"] = existing_trees
            if existing_trees:
                logger.info(f"Found {len(existing_trees)} existing reference SQL subject_trees for context")

        # Query similar reference SQLs for classification reference
        # Use comment as primary query text, fallback to first 200 chars of SQL
        query_text = user_input.comment if user_input.comment else ""
        if not query_text and user_input.sql_query:
            query_text = user_input.sql_query[:200]

        similar_items = self._get_similar_sqls(query_text, top_n=5)
        context["similar_items"] = similar_items
        if similar_items:
            logger.info(f"Found {len(similar_items)} similar reference SQLs for context")

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
            prompt_version: Optional prompt version to use (ignored, hardcoded to "1.0")
            template_context: Optional template context variables

        Returns:
            System prompt string loaded from the template
        """
        # Hardcoded prompt version
        version = "1.0"

        # Hardcoded system_prompt based on node name
        template_name = f"{self.configured_node_name}_system"

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
        Execute the SQL summary node interaction with streaming support.

        Args:
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        # Get input from self.input
        if self.input is None:
            raise ValueError("SQL summary input not set. Set self.input before calling execute_stream.")

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
            # prompt_version is now hardcoded to "1.0" in _get_system_prompt
            system_instruction = self._get_system_prompt(conversation_summary, None, template_context)

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

            # Extract sql_summary_file and output from the final response_content
            sql_summary_file, extracted_output = self._extract_sql_summary_and_output_from_response(
                {"content": response_content}
            )
            if extracted_output:
                response_content = extracted_output

            logger.debug(f"Final response_content: '{response_content}' (length: {len(response_content)})")

            # Extract token usage (only in interactive mode with session)
            tokens_used = 0
            if self.execution_mode == "interactive":
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

            # Auto-save to database in workflow mode
            if self.execution_mode == "workflow" and sql_summary_file:
                try:
                    self._save_to_db(sql_summary_file)
                    logger.info(f"Auto-saved to database: {sql_summary_file}")
                except Exception as e:
                    logger.error(f"Failed to auto-save to database: {e}")

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

    def _save_to_db(self, sql_summary_file: str):
        """
        Save generated SQL summary to database (synchronous).

        Args:
            sql_summary_file: Name of the SQL summary file (e.g., "query_001.yaml")
        """
        try:
            import os

            # Construct full path
            full_path = os.path.join(self.sql_summary_dir, sql_summary_file)

            if not os.path.exists(full_path):
                logger.warning(f"SQL summary file not found: {full_path}")
                return

            # Call static method to save to database with build_mode
            result = GenerationHooks._sync_reference_sql_to_db(full_path, self.agent_config, self.build_mode)

            if result.get("success"):
                logger.info(f"Successfully saved to database: {result.get('message')}")
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Failed to save to database: {error}")

        except Exception as e:
            logger.error(f"Error saving to database: {e}", exc_info=True)
            raise
