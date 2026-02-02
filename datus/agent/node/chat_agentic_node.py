# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ChatAgenticNode implementation for flexible CLI chat interactions.

This module provides a concrete implementation of GenSQLAgenticNode specifically
designed for chat interactions with database and filesystem tool support.
"""
from typing import AsyncGenerator, Optional, override

from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput, ChatNodeResult
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import ContextSearchTools, DBFuncTool
from datus.tools.permission.permission_hooks import CompositeHooks, PermissionHooks
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools.skill_func_tool import SkillFuncTool
from datus.tools.skill_tools.skill_manager import SkillManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ChatAgenticNode(GenSQLAgenticNode):
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
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[ChatNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
    ):
        """
        Initialize the ChatAgenticNode as a specialized GenSQLAgenticNode.

        Args:
            node_id: Unique identifier for the node
            description: Human-readable description of the node
            node_type: Type of the node (should be 'chat')
            input_data: Chat input data
            agent_config: Agent configuration
            tools: List of tools (will be populated in setup_tools)
        """
        # Initialize ChatAgenticNode-specific attributes BEFORE calling parent constructor
        # This is required because parent's __init__ calls setup_tools()
        # Note: permission_manager and skill_manager are initialized by parent AgenticNode
        self.skill_func_tool: Optional[SkillFuncTool] = None
        self.permission_hooks: Optional[PermissionHooks] = None

        # Call parent constructor with node_name="chat"
        # This will initialize max_turns, tool attributes, plan mode attributes, and MCP servers
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools,
            node_name="chat",
        )
        logger.debug(
            f"ChatAgenticNode initialized: {self.agent_config.current_namespace} {self.agent_config.current_database}"
        )

    def setup_input(self, workflow: Workflow) -> dict:
        """
        Setup chat input from workflow context.

        Creates ChatNodeInput with user message from task and context data.

        Args:
            workflow: Workflow instance containing context and task

        Returns:
            Dictionary with success status and message
        """
        # Update database connection if task specifies a different database
        task_database = workflow.task.database_name
        if task_database and self.db_func_tool and task_database != self.db_func_tool.connector.database_name:
            logger.debug(
                f"Updating database connection from '{self.db_func_tool.connector.database_name}' "
                f"to '{task_database}' based on workflow task"
            )
            self._update_database_connection(task_database)

        # Read plan_mode from workflow metadata
        plan_mode = workflow.metadata.get("plan_mode", False)
        auto_execute_plan = workflow.metadata.get("auto_execute_plan", False)

        # Create ChatNodeInput if not already set
        if not self.input:
            self.input = ChatNodeInput(
                user_message=workflow.task.task,
                external_knowledge=workflow.task.external_knowledge,
                catalog=workflow.task.catalog_name,
                database=workflow.task.database_name,
                db_schema=workflow.task.schema_name,
                schemas=workflow.context.table_schemas,
                metrics=workflow.context.metrics,
                reference_sql=None,
                plan_mode=plan_mode,
                auto_execute_plan=auto_execute_plan,
                prompt_version=self.node_config.get("prompt_version"),
            )
        else:
            # Update existing input with workflow data
            self.input.user_message = workflow.task.task
            self.input.external_knowledge = workflow.task.external_knowledge
            self.input.catalog = workflow.task.catalog_name
            self.input.database = workflow.task.database_name
            self.input.db_schema = workflow.task.schema_name
            self.input.schemas = workflow.context.table_schemas
            self.input.metrics = workflow.context.metrics

        return {"success": True, "message": "Chat input prepared from workflow"}

    def update_context(self, workflow: Workflow) -> dict:
        """
        Update workflow context with chat results.

        Stores SQL to workflow context if present in result.

        Args:
            workflow: Workflow instance to update

        Returns:
            Dictionary with success status and message
        """
        if not self.result:
            return {"success": False, "message": "No result to update context"}

        result = self.result

        try:
            if hasattr(result, "sql") and result.sql:
                from datus.schemas.node_models import SQLContext

                # Extract SQL result from the response if available
                sql_result = ""
                if hasattr(result, "response") and result.response:
                    # Try to extract SQL result from the response
                    _, sql_result = self._extract_sql_and_output_from_response({"content": result.response})
                    sql_result = sql_result or ""

                new_record = SQLContext(
                    sql_query=result.sql,
                    explanation=result.response if hasattr(result, "response") else "",
                    sql_return=sql_result,
                )
                workflow.context.sql_contexts.append(new_record)

            return {"success": True, "message": "Updated chat context"}
        except Exception as e:
            logger.error(f"Failed to update chat context: {e}")
            return {"success": False, "message": str(e)}

    @override
    def setup_tools(self):
        """Initialize all tools with default database connection."""
        # Chat node uses all available tools by default
        db_manager = db_manager_instance(self.agent_config.namespaces)
        conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
        self.db_func_tool = DBFuncTool(conn, agent_config=self.agent_config)
        self.context_search_tools = ContextSearchTools(self.agent_config)
        self._setup_date_parsing_tools()
        self._setup_filesystem_tools()
        self._setup_skill_tools()
        self._rebuild_tools()
        self._setup_platform_doc_tools()

        # Setup permission hooks after all tools are initialized
        self._setup_permission_hooks()

    def _setup_skill_tools(self):
        """Setup skill discovery and loading tools with permission control."""
        try:
            # Create PermissionManager from agent config
            self.permission_manager = PermissionManager(
                global_config=self.agent_config.permissions_config,
                node_overrides=self._get_node_permission_overrides(),
            )

            # Set permission callback for ASK permissions (legacy callback, for backward compatibility)
            self.permission_manager.set_permission_callback(self._handle_permission_ask)

            # Create SkillManager with permission support
            self.skill_manager = SkillManager(
                permission_manager=self.permission_manager,
            )
            self.skill_func_tool = SkillFuncTool(
                manager=self.skill_manager,
                node_name="chat",
            )
            logger.debug(f"Setup skill tools: {self.skill_manager.get_skill_count()} skills discovered")
        except Exception as e:
            logger.error(f"Failed to setup skill tools: {e}")

    def _setup_permission_hooks(self):
        """Setup permission hooks and register all tool categories.

        Creates PermissionHooks instance and registers all available tools
        with their respective categories for unified permission checking.
        Uses the InteractionBroker pattern for async user interactions.
        """
        if not self.permission_manager:
            logger.debug("No permission manager available, skipping permission hooks setup")
            return

        try:
            # Get the interaction broker from parent class
            broker = self._get_or_create_broker()

            self.permission_hooks = PermissionHooks(
                broker=broker,
                permission_manager=self.permission_manager,
                node_name=self.get_node_name(),
            )

            # Register tools by category (follows existing FuncTool structure)
            if self.db_func_tool:
                self.permission_hooks.register_tools("db_tools", self.db_func_tool.available_tools())
            if self.context_search_tools:
                self.permission_hooks.register_tools(
                    "context_search_tools", self.context_search_tools.available_tools()
                )
            if self.date_parsing_tools:
                self.permission_hooks.register_tools("date_parsing_tools", self.date_parsing_tools.available_tools())
            if self.filesystem_func_tool:
                self.permission_hooks.register_tools("filesystem_tools", self.filesystem_func_tool.available_tools())
            if self.skill_func_tool:
                self.permission_hooks.register_tools("skills", self.skill_func_tool.available_tools())

            logger.debug(f"Permission hooks setup with {len(self.permission_hooks.tool_registry)} registered tools")
        except Exception as e:
            logger.error(f"Failed to setup permission hooks: {e}")
            self.permission_hooks = None

    async def _handle_permission_ask(
        self,
        tool_category: str,
        tool_name: str,
        context: dict,
    ) -> bool:
        """Handle ASK permission by prompting user for confirmation.

        Args:
            tool_category: Category of the tool (e.g., "skills")
            tool_name: Name of the tool/skill
            context: Additional context about the request

        Returns:
            True if user approved, False otherwise
        """
        try:
            from rich.console import Console
            from rich.prompt import Confirm

            console = Console()
            console.print(f"\n[yellow]Permission required:[/yellow] {tool_category}.{tool_name}")
            if context:
                console.print(f"[dim]Context: {context}[/dim]")

            approved = Confirm.ask(f"Allow {tool_name}?", default=False)

            if approved:
                # Ask if user wants to approve for the entire session
                always = Confirm.ask("Always allow this session?", default=False)
                if always and self.permission_manager:
                    self.permission_manager.approve_for_session(tool_category, tool_name)

            return approved
        except Exception as e:
            logger.error(f"Permission prompt failed: {e}")
            return False

    def _get_node_permission_overrides(self) -> dict:
        """Get node-specific permission overrides from agent config.

        Returns:
            Dictionary of node_name -> PermissionConfig
        """
        # Check if agent config has node-specific permission overrides
        if not self.agent_config:
            return {}

        # Look for permission overrides in agentic_nodes config
        chat_config = self.agent_config.agentic_nodes.get("chat", {})
        if isinstance(chat_config, dict) and "permissions" in chat_config:
            return {"chat": chat_config["permissions"]}

        return {}

    @override
    def _get_execution_config(self, execution_mode: str, original_input) -> dict:
        """Get execution configuration with permission hooks for all tools.

        Overrides parent to add unified permission hooks that check permissions
        before executing any tool (db_tools, skills, MCP, filesystem, etc.).

        Args:
            execution_mode: "normal" or "plan"
            original_input: Original chat input for context

        Returns:
            Configuration dict with tools, instruction, and hooks
        """
        # Get base config from parent
        config = super()._get_execution_config(execution_mode, original_input)

        # Add permission hooks if available
        if self.permission_hooks:
            existing_hooks = config.get("hooks")
            if existing_hooks:
                # Combine with existing hooks (e.g., PlanModeHooks)
                config["hooks"] = CompositeHooks([existing_hooks, self.permission_hooks])
            else:
                config["hooks"] = self.permission_hooks

        return config

    @override
    def _rebuild_tools(self):
        """Rebuild the tools list with current tool instances including skills."""
        self.tools = []
        if self.db_func_tool:
            self.tools.extend(self.db_func_tool.available_tools())
        if self.context_search_tools:
            self.tools.extend(self.context_search_tools.available_tools())
        if self.date_parsing_tools:
            self.tools.extend(self.date_parsing_tools.available_tools())
        if self.filesystem_func_tool:
            self.tools.extend(self.filesystem_func_tool.available_tools())
        # Add skill tools
        if self.skill_func_tool:
            self.tools.extend(self.skill_func_tool.available_tools())

    @override
    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> str:
        """Get system prompt with available skills injected.

        Args:
            conversation_summary: Optional summary from previous conversation compact
            prompt_version: Optional prompt version to use

        Returns:
            System prompt string with available skills XML appended
        """
        # Call parent to get base prompt
        base_prompt = super()._get_system_prompt(conversation_summary, prompt_version)

        # Generate available skills XML and append to prompt
        if self.skill_manager and self.skill_manager.get_skill_count() > 0:
            skills_xml = self.skill_manager.generate_available_skills_xml(node_name="chat")
            if skills_xml:
                base_prompt = base_prompt + "\n\n" + skills_xml

        return base_prompt

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the chat interaction with streaming support.

        Input is accessed from self.input instead of parameters.

        Args:
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        # Get input from self.input (set by setup_input or directly)
        if not self.input:
            raise ValueError("Chat input not set. Call setup_input() first or set self.input directly.")

        user_input = self.input

        is_plan_mode = getattr(user_input, "plan_mode", False)
        if is_plan_mode:
            self.plan_mode_active = True

            # Create plan mode hooks with interaction broker
            from datus.cli.plan_hooks import PlanModeHooks

            broker = self._get_or_create_broker()
            session = self._get_or_create_session()[0]

            # Workflow sets 'auto_execute_plan' in metadata, CLI REPL does not
            auto_mode = getattr(user_input, "auto_execute_plan", False)
            logger.info(f"Plan mode auto_mode: {auto_mode} (from input)")

            self.plan_hooks = PlanModeHooks(broker=broker, session=session, auto_mode=auto_mode)

        # Create initial action
        action_type = "plan_mode_interaction" if is_plan_mode else "chat_interaction"
        action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type=action_type,
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

            # Add database context to user message if provided
            from datus.agent.node.gen_sql_agentic_node import build_enhanced_message

            enhanced_message = build_enhanced_message(
                user_message=user_input.user_message,
                db_type=self.agent_config.db_type,
                catalog=user_input.catalog,
                database=user_input.database,
                db_schema=user_input.db_schema,
                external_knowledge=user_input.external_knowledge,
                schemas=user_input.schemas,
                metrics=user_input.metrics,
                reference_sql=user_input.reference_sql,
            )

            # Execute with streaming
            response_content = ""
            sql_content = None
            tokens_used = 0
            last_successful_output = None

            # Determine execution mode and start unified recursive execution
            execution_mode = "plan" if is_plan_mode and self.plan_hooks else "normal"

            # Start unified recursive execution
            async for stream_action in self._execute_with_recursive_replan(
                prompt=enhanced_message,
                execution_mode=execution_mode,
                original_input=user_input,
                action_history_manager=action_history_manager,
                session=session,
            ):
                yield stream_action

                # Collect response content from successful actions
                if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        last_successful_output = stream_action.output
                        # Look for content in various possible fields
                        # Only collect raw_output if it's from a "message" type action (Thinking messages)
                        raw_output_value = ""
                        if stream_action.action_type == "message" and "raw_output" in stream_action.output:
                            raw_output_value = stream_action.output.get("raw_output", "")

                        response_content = (
                            stream_action.output.get("content", "")
                            or stream_action.output.get("response", "")
                            or raw_output_value
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
                    or last_successful_output.get("raw_output", "")  # Try raw_output from any action type
                    or str(last_successful_output)  # Fallback to string representation
                )

            # Extract SQL directly from summary_report action if available
            sql_content = None
            for stream_action in reversed(action_history_manager.get_actions()):
                if stream_action.action_type == "summary_report" and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        sql_content = stream_action.output.get("sql")
                        # Also get the markdown/content if response_content is still empty
                        if not response_content:
                            response_content = (
                                stream_action.output.get("markdown", "")
                                or stream_action.output.get("content", "")
                                or stream_action.output.get("response", "")
                            )
                        if sql_content:  # Found SQL, stop searching
                            logger.debug(f"Extracted SQL from summary_report action: {sql_content[:100]}...")
                            break

            # Fallback: try to extract SQL and output from response_content if not found
            if not sql_content:
                extracted_sql, extracted_output = self._extract_sql_and_output_from_response(
                    {"content": response_content}
                )
                if extracted_sql:
                    sql_content = extracted_sql
                if extracted_output:
                    response_content = extracted_output

            logger.debug(f"Final response_content: '{response_content}' (length: {len(response_content)})")
            logger.debug(f"Final sql_content: {sql_content[:100] if sql_content else 'None'}...")

            # Extract token usage from final actions using our new approach
            # With our streaming token fix, only the final assistant action will have accurate usage
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

            # Collect action history and calculate execution stats
            all_actions = action_history_manager.get_actions()
            tool_calls = [
                action
                for action in all_actions
                if action.role == ActionRole.TOOL and action.status == ActionStatus.SUCCESS
            ]

            execution_stats = {
                "total_actions": len(all_actions),
                "tool_calls_count": len(tool_calls),
                "tools_used": list(set([a.action_type for a in tool_calls])),
                "total_tokens": int(tokens_used),
            }

            # Create final result with action history
            result = ChatNodeResult(
                success=True,
                response=response_content,
                sql=sql_content,
                tokens_used=int(tokens_used),
                action_history=[action.model_dump() for action in all_actions],
                execution_stats=execution_stats,
            )

            # # Update assistant action with success
            # action_history_manager.update_action_by_id(
            #     assistant_action.action_id,
            #     status=ActionStatus.SUCCESS,
            #     output=result.model_dump(),
            #     messages=(
            #         f"Generated response: {response_content[:100]}..."
            #         if len(response_content) > 100
            #         else response_content
            #     ),
            # )

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
            # Handle user cancellation as success, not error
            if "User cancelled" in str(e) or "UserCancelledException" in str(type(e).__name__):
                logger.info("User cancelled execution, stopping gracefully...")

                # Create cancellation result (success=True)
                result = ChatNodeResult(
                    success=True,
                    response="Execution cancelled by user.",
                    tokens_used=0,
                )

                # Update action with cancellation
                action_history_manager.update_current_action(
                    status=ActionStatus.SUCCESS,
                    output=result.model_dump(),
                    messages="Execution cancelled by user",
                )

                # Create cancellation action
                action = ActionHistory.create_action(
                    role=ActionRole.ASSISTANT,
                    action_type="user_cancellation",
                    messages="Execution cancelled by user",
                    input_data=user_input.model_dump(),
                    output_data=result.model_dump(),
                    status=ActionStatus.SUCCESS,
                )
            else:
                logger.error(f"Chat execution error: {e}")

                # Create error result for all other exceptions
                result = ChatNodeResult(
                    success=False,
                    error=str(e),
                    response="Sorry, I encountered an error while processing your request.",
                    tokens_used=0,
                )

                # Update action with error
                action_history_manager.update_current_action(
                    status=ActionStatus.FAILED,
                    output=result.model_dump(),
                    messages=f"Error: {str(e)}",
                )

                # Create error action
                action = ActionHistory.create_action(
                    role=ActionRole.ASSISTANT,
                    action_type="error",
                    messages=f"Chat interaction failed: {str(e)}",
                    input_data=user_input.model_dump(),
                    output_data=result.model_dump(),
                    status=ActionStatus.FAILED,
                )

            action_history_manager.add_action(action)
            yield action

        finally:
            # Clean up plan mode state
            if is_plan_mode:
                self.plan_mode_active = False
                self.plan_hooks = None
