# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenReportAgenticNode implementation for generic report generation.

This module provides a base implementation of AgenticNode focused on
report generation with semantic and database tools. It can be used directly
or extended by specialized report nodes like AttributionAgenticNode.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.gen_report_agentic_node_models import GenReportNodeInput, GenReportNodeResult
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.semantic_tools import SemanticTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenReportAgenticNode(AgenticNode):
    """
    Generic report generation agentic node.

    This node provides a flexible base for report generation with:
    - Configuration-based tool setup (semantic_tools.*, db_tools.*)
    - Common streaming execution logic
    - Template context building
    - Result extraction framework

    Can be instantiated directly or extended by specialized nodes.
    """

    NODE_NAME = "gen_report"

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: Optional[GenReportNodeInput] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[list] = None,
        node_name: Optional[str] = None,
    ):
        """
        Initialize the GenReportAgenticNode.

        Args:
            node_id: Unique identifier for the node
            description: Human-readable description of the node
            node_type: Type of the node
            input_data: Report generation input data
            agent_config: Agent configuration
            tools: List of tools (will be populated in setup_tools)
            node_name: Name of the node configuration in agent.yml
        """
        # Determine node name from node_type if not provided
        self.configured_node_name = node_name

        self.max_turns = 30
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[node_name]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 30)

        # Initialize tool attributes BEFORE calling parent constructor
        # This is required because parent's __init__ calls _get_system_prompt()
        # which may reference these attributes
        self.db_func_tool: Optional[DBFuncTool] = None
        self.semantic_tools: Optional[SemanticTools] = None

        # Call parent constructor with all required Node parameters
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
            tools=tools or [],
            mcp_servers={},  # No MCP servers for report nodes by default
        )

        # Setup tools based on configuration
        self.setup_tools()
        logger.debug(f"GenReportAgenticNode tools: {len(self.tools)} tools - {[tool.name for tool in self.tools]}")

    def get_node_name(self) -> str:
        """
        Get the configured node name for this report agentic node.

        Returns:
            The configured node name from agent.yml or NODE_NAME default
        """
        return self.configured_node_name or self.NODE_NAME

    def setup_tools(self):
        """
        Setup tools based on configuration.

        Reads 'tools' from node_config and sets up each tool pattern.
        If no tools configured, self.tools remains empty.
        """
        if not self.agent_config:
            return

        self.tools = []

        # Setup tools from configuration
        config_value = self.node_config.get("tools", "")
        if not config_value:
            return  # No tools if not configured

        tool_patterns = [p.strip() for p in config_value.split(",") if p.strip()]
        for pattern in tool_patterns:
            self._setup_tool_pattern(pattern)

        logger.info(f"Setup {len(self.tools)} tools: {[tool.name for tool in self.tools]}")

    def _setup_tool_pattern(self, pattern: str):
        """
        Setup tools based on pattern.

        Supports patterns like:
        - "semantic_tools.*" -> all semantic tools
        - "db_tools.*" -> all db tools
        - "semantic_tools.search_metrics" -> specific method
        - "db_tools.list_tables" -> specific method
        """
        try:
            # Handle wildcard patterns (e.g., "semantic_tools.*")
            if pattern.endswith(".*"):
                base_type = pattern[:-2]
                if base_type == "semantic_tools":
                    self._setup_semantic_tools()
                elif base_type == "db_tools":
                    self._setup_db_tools()
                else:
                    logger.warning(f"Unknown tool type: {base_type}")

            # Handle exact type patterns
            elif pattern == "semantic_tools":
                self._setup_semantic_tools()
            elif pattern == "db_tools":
                self._setup_db_tools()

            # Handle specific method patterns (e.g., "db_tools.describe_table")
            elif "." in pattern:
                tool_type, method_name = pattern.split(".", 1)
                self._setup_specific_tool_method(tool_type, method_name)

            else:
                logger.warning(f"Unknown tool pattern: {pattern}")

        except Exception as e:
            logger.error(f"Failed to setup tool pattern '{pattern}': {e}")

    def _setup_db_tools(self):
        """Setup database tools."""
        try:
            db_manager = db_manager_instance(self.agent_config.namespaces)
            conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
            self.db_func_tool = DBFuncTool(
                conn,
                agent_config=self.agent_config,
                sub_agent_name=self.node_config.get("system_prompt"),
            )
            self.tools.extend(self.db_func_tool.available_tools())
            logger.debug("Added database tools from DBFuncTool")
        except Exception as e:
            logger.error(f"Failed to setup database tools: {e}")

    def _setup_semantic_tools(self):
        """Setup semantic tools for report analysis."""
        try:
            # Get adapter_type from configuration
            adapter_type = self.node_config.get("adapter_type", "metricflow")
            self.semantic_tools = SemanticTools(
                agent_config=self.agent_config,
                sub_agent_name=self.node_config.get("system_prompt"),
                adapter_type=adapter_type,
            )
            self.tools.extend(self.semantic_tools.available_tools())
            logger.debug("Added semantic tools from SemanticTools")
        except Exception as e:
            logger.error(f"Failed to setup semantic tools: {e}")

    def _setup_specific_tool_method(self, tool_type: str, method_name: str):
        """Setup a specific tool method."""
        try:
            if tool_type == "semantic_tools":
                if not self.semantic_tools:
                    adapter_type = self.node_config.get("adapter_type", "metricflow")
                    self.semantic_tools = SemanticTools(
                        agent_config=self.agent_config,
                        sub_agent_name=self.node_config.get("system_prompt"),
                        adapter_type=adapter_type,
                    )
                tool_instance = self.semantic_tools
            elif tool_type == "db_tools":
                if not self.db_func_tool:
                    db_manager = db_manager_instance(self.agent_config.namespaces)
                    conn = db_manager.get_conn(self.agent_config.current_namespace, self.agent_config.current_database)
                    self.db_func_tool = DBFuncTool(
                        conn,
                        agent_config=self.agent_config,
                        sub_agent_name=self.node_config.get("system_prompt"),
                    )
                tool_instance = self.db_func_tool
            else:
                logger.warning(f"Unknown tool type: {tool_type}")
                return

            if hasattr(tool_instance, method_name):
                method = getattr(tool_instance, method_name)
                from datus.tools.func_tool import trans_to_function_tool

                self.tools.append(trans_to_function_tool(method))
                logger.debug(f"Added specific tool method: {tool_type}.{method_name}")
            else:
                logger.warning(f"Method '{method_name}' not found in {tool_type}")
        except Exception as e:
            logger.error(f"Failed to setup {tool_type}.{method_name}: {e}")

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> str:
        """
        Get the system prompt for this report generation node.

        Args:
            conversation_summary: Optional summary from previous conversation compact
            prompt_version: Optional prompt version to use

        Returns:
            System prompt string loaded from the template
        """
        context = {
            "has_semantic_tools": bool(self.semantic_tools),
            "has_db_tools": bool(self.db_func_tool),
            "agent_config": self.agent_config,
            "conversation_summary": conversation_summary,
        }

        # Add rules from configuration
        context["rules"] = self.node_config.get("rules", [])

        # Add agent description from configuration
        context["agent_description"] = self.node_config.get("agent_description", "")

        # Add namespace info
        if self.agent_config:
            context["namespace"] = getattr(self.agent_config, "current_namespace", None)
            context["db_name"] = getattr(self.agent_config, "current_database", None)

        raw_version = prompt_version if prompt_version is not None else self.node_config.get("prompt_version")
        version = None if raw_version in (None, "") else str(raw_version)

        # Construct template name: {system_prompt}_system or fallback to {node_name}_system
        system_prompt_name = self.node_config.get("system_prompt") or self.get_node_name()
        template_name = f"{system_prompt_name}_system"

        # Use prompt manager to render the template
        from datus.prompts.prompt_manager import prompt_manager

        try:
            return prompt_manager.render_template(template_name=template_name, version=version, **context)

        except FileNotFoundError:
            # Template not found - use default gen_report template
            logger.warning(
                f"Failed to render system prompt '{system_prompt_name}', using the default gen_report template"
            )
            return prompt_manager.render_template(template_name="gen_report_system", version=version, **context)

    def _build_enhanced_message(self, user_input: GenReportNodeInput) -> str:
        """
        Build enhanced message with context.

        Base implementation adds database context. Subclasses can override
        to add additional context specific to their report type.

        Args:
            user_input: Report node input

        Returns:
            Enhanced message string with context
        """
        parts = [f"Question: {user_input.user_message}"]

        # Add database context
        if user_input.database:
            parts.append(f"\nDatabase context: {user_input.database}")
        if user_input.db_schema:
            parts.append(f"Schema: {user_input.db_schema}")

        return "\n".join(parts)

    def _extract_report_result(self, actions: List[ActionHistory]) -> Optional[Dict[str, Any]]:
        """
        Extract report result from tool call actions.

        Subclasses can override this to extract specific tool results.

        Args:
            actions: List of action history entries

        Returns:
            Report result dict if found, None otherwise
        """
        # Base implementation returns None - subclasses should override
        return None

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the report generation with streaming support.

        Args:
            action_history_manager: Optional action history manager

        Yields:
            ActionHistory: Progress updates during execution
        """
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        # Get input from self.input
        if not self.input:
            raise ValueError("Report input not set. Call setup_input() first or set self.input directly.")

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
            # Check for auto-compact before session creation
            await self._auto_compact()

            # Get or create session
            session, conversation_summary = self._get_or_create_session()

            system_instruction = self._get_system_prompt(conversation_summary, user_input.prompt_version)

            # Build enhanced message with context
            enhanced_message = self._build_enhanced_message(user_input)

            # Execute with streaming
            response_content = ""
            tokens_used = 0
            last_successful_output = None

            # Create assistant action for processing
            assistant_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="llm_generation",
                messages="Analyzing data and generating report...",
                input_data={"prompt": enhanced_message, "system": system_instruction},
                status=ActionStatus.PROCESSING,
            )
            action_history_manager.add_action(assistant_action)
            yield assistant_action

            logger.debug(f"Tools available: {len(self.tools)} tools - {[tool.name for tool in self.tools]}")

            # Stream response using model's generate_with_tools_stream
            async for stream_action in self.model.generate_with_tools_stream(
                prompt=enhanced_message,
                tools=self.tools,
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
                        response_content = (
                            stream_action.output.get("content", "")
                            or stream_action.output.get("response", "")
                            or stream_action.output.get("raw_output", "")
                            or response_content
                        )

            # If we still don't have response_content, check the last successful output
            if not response_content and last_successful_output:
                response_content = (
                    last_successful_output.get("content", "")
                    or last_successful_output.get("text", "")
                    or last_successful_output.get("response", "")
                    or str(last_successful_output)
                )

            # Extract report result from tool calls (subclass-specific)
            all_actions = action_history_manager.get_actions()
            report_result = self._extract_report_result(all_actions)

            # Extract token usage
            for action in reversed(all_actions):
                if action.role == "assistant":
                    if action.output and isinstance(action.output, dict):
                        usage_info = action.output.get("usage", {})
                        if usage_info and isinstance(usage_info, dict) and usage_info.get("total_tokens"):
                            tokens_used = usage_info.get("total_tokens", 0)
                            self._add_session_tokens(tokens_used)
                            break

            # Collect execution stats
            tool_calls = [action for action in all_actions if action.role == ActionRole.TOOL]
            execution_stats = {
                "total_actions": len(all_actions),
                "tool_calls_count": len(tool_calls),
                "tools_used": list(set([a.action_type for a in tool_calls])),
                "total_tokens": int(tokens_used),
            }

            # Create final result
            result = GenReportNodeResult(
                success=True,
                response=response_content,
                report_result=report_result,
                tokens_used=int(tokens_used),
                action_history=[action.model_dump() for action in all_actions],
                execution_stats=execution_stats,
            )

            # Add to internal actions list
            self.actions.extend(all_actions)

            # Create final action
            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type=f"{self.get_node_name()}_response",
                messages=f"{self.get_node_name()} analysis completed successfully",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except Exception as e:
            logger.error(f"{self.get_node_name()} execution error: {e}")

            # Create error result
            error_result = GenReportNodeResult(
                success=False,
                error=str(e),
                response="Sorry, I encountered an error while generating the report.",
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
                messages=f"{self.get_node_name()} analysis failed: {str(e)}",
                input_data=user_input.model_dump(),
                output_data=error_result.model_dump(),
                status=ActionStatus.FAILED,
            )
            action_history_manager.add_action(error_action)
            yield error_action
