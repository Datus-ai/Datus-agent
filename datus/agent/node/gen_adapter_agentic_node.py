# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenAdapterAgenticNode implementation for adapter code generation.

This module provides a specialized implementation of AgenticNode focused on
generating adapter project scaffolding and assisting with platform-specific
implementation. Tools: AdapterScaffoldTool + FilesystemFuncTool +
PlatformDocSearchTool + AskUserTool.
"""

from typing import AsyncGenerator, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.cli.execution_state import ExecutionInterrupted
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput, SemanticNodeResult
from datus.tools.func_tool.adapter_scaffold_tool import AdapterScaffoldTool
from datus.tools.func_tool.base import trans_to_function_tool
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenAdapterAgenticNode(AgenticNode):
    """
    Adapter generation agentic node.

    This node provides adapter scaffolding and code generation with:
    - AdapterScaffoldTool for project skeleton generation and validation
    - FilesystemFuncTool for reading/editing generated code
    - PlatformDocSearchTool for fetching external API documentation
    - AskUserTool for interactive confirmation
    """

    NODE_NAME = "gen_adapter"

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
    ):
        self.execution_mode = execution_mode

        self.max_turns = 30
        if agent_config and hasattr(agent_config, "agentic_nodes") and self.NODE_NAME in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[self.NODE_NAME]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 30)

        from datus.configuration.node_type import NodeType

        super().__init__(
            node_id=f"{self.NODE_NAME}_node",
            description=f"Adapter generation node: {self.NODE_NAME}",
            node_type=NodeType.TYPE_GEN_ADAPTER,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
        )

        self.adapter_scaffold_tool: Optional[AdapterScaffoldTool] = None
        self.filesystem_func_tool: Optional[FilesystemFuncTool] = None
        self.ask_user_tool = None
        self.setup_tools()

    def get_node_name(self) -> str:
        return self.NODE_NAME

    def setup_tools(self):
        """Setup tools for adapter generation."""
        if not self.agent_config:
            return

        self.tools = []
        self._setup_scaffold_tools()
        self._setup_filesystem_tools()
        self._setup_platform_doc_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.info(f"Setup {len(self.tools)} tools for {self.NODE_NAME}: {[tool.name for tool in self.tools]}")

    def _setup_scaffold_tools(self):
        """Setup adapter scaffolding and validation tools."""
        try:
            self.adapter_scaffold_tool = AdapterScaffoldTool(self.agent_config)
            self.tools.extend(self.adapter_scaffold_tool.available_tools())
            logger.debug("Added adapter scaffold tools: scaffold_adapter, validate_adapter, list_adapter_types")
        except Exception as e:
            logger.error(f"Failed to setup adapter scaffold tools: {e}")

    def _setup_filesystem_tools(self):
        """Setup filesystem tools for reading/editing generated adapter code."""
        try:
            self.filesystem_func_tool = FilesystemFuncTool()
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

    def _setup_platform_doc_tools(self):
        """Setup platform documentation search tools for fetching external API docs."""
        try:
            from datus.tools.func_tool.platform_doc_search import PlatformDocSearchTool

            self.platform_doc_tool = PlatformDocSearchTool(self.agent_config)
            self.tools.extend(self.platform_doc_tool.available_tools())
            logger.debug("Added platform doc search tools")
        except Exception as e:
            logger.error(f"Failed to setup platform doc search tools: {e}")

    def _prepare_template_context(self, user_input: SemanticNodeInput) -> dict:
        context = {}
        context["native_tools"] = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"
        context["mcp_tools"] = ", ".join(list(self.mcp_servers.keys())) if self.mcp_servers else "None"
        context["has_ask_user_tool"] = self.ask_user_tool is not None
        return context

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        version = self.node_config.get("prompt_version")
        template_name = f"{self.NODE_NAME}_system"

        try:
            template_vars = {
                "agent_config": self.agent_config,
                "conversation_summary": conversation_summary,
            }
            if template_context:
                template_vars.update(template_context)

            from datus.prompts.prompt_manager import prompt_manager

            base_prompt = prompt_manager.render_template(template_name=template_name, version=version, **template_vars)
            return self._finalize_system_prompt(base_prompt)

        except FileNotFoundError:
            # Fallback: use skill content as system prompt if no template exists
            logger.debug(f"No template found for '{template_name}', using default prompt")
            return self._finalize_system_prompt(
                "You are an adapter generation assistant. Help users create adapter project scaffolding "
                "for integrating external platforms (BI, DB, Scheduler, Semantic Layer) with Datus. "
                "Use the available tools to scaffold, edit, and validate adapters."
            )
        except Exception as e:
            logger.error(f"Template loading error for '{template_name}': {e}")
            return self._finalize_system_prompt(
                "You are an adapter generation assistant. Help users create adapter projects."
            )

    async def execute_stream(
        self,
        action_history_manager: Optional[ActionHistoryManager] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        if self.input is None:
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(ErrorCode.COMMON_FIELD_REQUIRED, message_args={"field_name": "input"})

        user_input = self.input

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
            session = None
            conversation_summary = None
            if self.execution_mode == "interactive":
                await self._auto_compact()
                session, conversation_summary = self._get_or_create_session()

            template_context = self._prepare_template_context(user_input)
            system_instruction = self._get_system_prompt(conversation_summary, template_context)

            response_content = ""
            last_successful_output = None

            async for stream_action in self.model.generate_with_tools_stream(
                prompt=user_input.user_message,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
                instruction=system_instruction,
                max_turns=user_input.max_turns if user_input.max_turns else self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=None,
                interrupt_controller=self.interrupt_controller,
            ):
                yield stream_action

                if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                    if isinstance(stream_action.output, dict):
                        last_successful_output = stream_action.output
                        raw_output = stream_action.output.get("raw_output", "")
                        if isinstance(raw_output, dict):
                            response_content = raw_output
                        elif raw_output:
                            response_content = raw_output

            if not response_content and last_successful_output:
                raw_output = last_successful_output.get("raw_output", "")
                if isinstance(raw_output, dict):
                    response_content = raw_output
                elif raw_output:
                    response_content = raw_output
                else:
                    response_content = str(last_successful_output)

            tokens_used = 0
            if self.execution_mode == "interactive":
                final_actions = action_history_manager.get_actions()
                for act in reversed(final_actions):
                    if act.role == "assistant":
                        if act.output and isinstance(act.output, dict):
                            usage_info = act.output.get("usage", {})
                            if usage_info and isinstance(usage_info, dict) and usage_info.get("total_tokens"):
                                tokens_used = usage_info.get("total_tokens", 0)
                                if tokens_used > 0:
                                    break

            result = SemanticNodeResult(
                success=True,
                response=response_content,
                semantic_models=[],
                tokens_used=int(tokens_used),
            )

            self.actions.extend(action_history_manager.get_actions())

            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="gen_adapter_response",
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

            error_result = SemanticNodeResult(
                success=False,
                error=str(e),
                response="Sorry, I encountered an error while processing your request.",
                tokens_used=0,
            )

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
