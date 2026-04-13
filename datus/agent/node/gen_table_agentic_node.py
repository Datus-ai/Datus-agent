# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenTableAgenticNode implementation for wide table generation.

This module provides a specialized implementation of AgenticNode focused on
creating database tables via CTAS (from JOIN SQL) or CREATE TABLE (from
natural language descriptions). Only DB tools + DDL + ask_user are included.
"""

from typing import AsyncGenerator, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.cli.execution_state import ExecutionInterrupted
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput, SemanticNodeResult
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.base import trans_to_function_tool
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.message_utils import MessagePart, build_structured_content

logger = get_logger(__name__)


class GenTableAgenticNode(AgenticNode):
    """
    Wide table generation agentic node.

    This node provides table creation capabilities with:
    - Database tools for schema exploration
    - DDL execution tool for table creation
    - Ask-user tool for interactive confirmation
    - Session-based conversation management
    """

    NODE_NAME = "gen_table"

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
    ):
        self.execution_mode = execution_mode

        self.max_turns = 20
        if agent_config and hasattr(agent_config, "agentic_nodes") and self.NODE_NAME in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[self.NODE_NAME]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 20)

        from datus.configuration.node_type import NodeType

        super().__init__(
            node_id=f"{self.NODE_NAME}_node",
            description=f"Table generation node: {self.NODE_NAME}",
            node_type=NodeType.TYPE_GEN_TABLE,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
        )

        self.db_func_tool: Optional[DBFuncTool] = None
        self.ask_user_tool = None
        self.setup_tools()

    def get_node_name(self) -> str:
        return self.NODE_NAME

    def setup_tools(self):
        if not self.agent_config:
            return

        self.tools = []
        self._setup_db_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.debug(f"Setup {len(self.tools)} tools for {self.NODE_NAME}: {[tool.name for tool in self.tools]}")

    def _setup_db_tools(self):
        """Setup database tools including DDL execution."""
        try:
            self.db_func_tool = DBFuncTool.create_dynamic(
                self.agent_config,
                sub_agent_name=self.NODE_NAME,
            )
            # Standard read-only tools (list_tables, describe_table, read_query, etc.)
            self.tools.extend(self.db_func_tool.available_tools())
            # DDL tool (gen-table specific, not in available_tools() default list)
            if hasattr(self.db_func_tool, "execute_ddl"):
                self.tools.append(trans_to_function_tool(self.db_func_tool.execute_ddl))
            logger.debug("Added database tools + execute_ddl from DBFuncTool")
        except Exception as e:
            logger.exception("Failed to setup database tools")
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Failed to setup database tools for {self.NODE_NAME}: {e}"},
            ) from e

    def _prepare_template_context(self, user_input: SemanticNodeInput) -> dict:
        context = {}
        context["native_tools"] = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"
        context["mcp_tools"] = ", ".join(list(self.mcp_servers.keys())) if self.mcp_servers else "None"
        context["has_ask_user_tool"] = self.ask_user_tool is not None
        logger.debug(f"Prepared template context: {context}")
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

            from datus.prompts.prompt_manager import get_prompt_manager

            base_prompt = get_prompt_manager(agent_config=self.agent_config).render_template(
                template_name=template_name, version=version, **template_vars
            )
            return self._finalize_system_prompt(base_prompt)

        except FileNotFoundError as e:
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                code=ErrorCode.COMMON_TEMPLATE_NOT_FOUND,
                message_args={"template_name": template_name, "version": version},
            ) from e
        except Exception as e:
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
        if not action_history_manager:
            action_history_manager = ActionHistoryManager()

        if self.input is None:
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

            response_content = ""
            last_successful_output = None

            async for stream_action in self.model.generate_with_tools_stream(
                prompt=enhanced_message,
                tools=self.tools,
                mcp_servers=self.mcp_servers,
                instruction=system_instruction,
                max_turns=user_input.max_turns if user_input.max_turns else self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
                hooks=None,
                agent_name=self.get_node_name(),
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
                for action in reversed(final_actions):
                    if action.role == "assistant":
                        if action.output and isinstance(action.output, dict):
                            usage_info = action.output.get("usage", {})
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
                action_type="gen_table_response",
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
