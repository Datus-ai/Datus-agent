# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
SchedulerAgenticNode implementation for job scheduling operations.

This module provides a specialized implementation of AgenticNode focused on
job scheduling management via Airflow (submit, monitor, troubleshoot).
Only scheduler tools + ask_user are included — no DB/BI/filesystem tools exposed.
"""

from typing import Any, AsyncGenerator, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.cli.execution_state import ExecutionInterrupted
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.scheduler_agentic_node_models import SchedulerNodeResult
from datus.utils.loggings import get_logger
from datus.utils.message_utils import MessagePart, build_structured_content

logger = get_logger(__name__)


class SchedulerAgenticNode(AgenticNode):
    """
    Job scheduler agentic node.

    This node provides scheduler management capabilities with:
    - Scheduler tools for job CRUD (via SchedulerTools)
    - Ask-user tool for interactive confirmation
    - Session-based conversation management
    """

    NODE_NAME = "scheduler"

    def __init__(
        self,
        agent_config: AgentConfig,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
    ):
        self.execution_mode = execution_mode
        # Support custom node_name for alias subagents (e.g. my_scheduler: {node_class: scheduler})
        self._configured_node_name = node_name or self.NODE_NAME

        self.max_turns = 30
        config_key = self._configured_node_name
        if agent_config and hasattr(agent_config, "agentic_nodes") and config_key in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[config_key]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 30)

        from datus.configuration.node_type import NodeType

        super().__init__(
            node_id=node_id or f"{self.NODE_NAME}_node",
            description=f"Job scheduler node: {self.NODE_NAME}",
            node_type=NodeType.TYPE_SCHEDULER,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
        )

        self.scheduler_tools = None
        self.ask_user_tool = None
        self.setup_tools()

    def get_node_name(self) -> str:
        return self._configured_node_name

    # ── Tool Setup ──────────────────────────────────────────────────────

    def setup_tools(self):
        if not self.agent_config:
            return

        self.tools = []
        self._setup_scheduler_tools()
        if self.execution_mode == "interactive":
            self._setup_ask_user_tool()

        logger.debug(f"Setup {len(self.tools)} tools for {self.NODE_NAME}: {[tool.name for tool in self.tools]}")

    def _setup_scheduler_tools(self):
        """Setup scheduler tools if scheduler_config is present."""
        if not getattr(self.agent_config, "scheduler_config", None):
            return
        try:
            from datus.tools.func_tool.scheduler_tools import SchedulerTools

            self.scheduler_tools = SchedulerTools(self.agent_config)
            self.tools.extend(self.scheduler_tools.available_tools())
            logger.info("Scheduler tools initialized")
        except ImportError as e:
            logger.warning(f"Scheduler adapter package not installed: {e}")
        except Exception as e:
            logger.error(f"Failed to setup scheduler tools: {e}")

    # ── System Prompt ───────────────────────────────────────────────────

    def _prepare_template_context(self) -> dict:
        """Build template context for scheduler prompt."""
        context: dict[str, Any] = {}
        context["native_tools"] = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"
        context["has_ask_user_tool"] = self.ask_user_tool is not None
        return context

    def _get_system_prompt(
        self,
        conversation_summary: Optional[str] = None,
        template_context: Optional[dict] = None,
        prompt_version: Optional[str] = None,
    ) -> str:
        version = prompt_version or self.node_config.get("prompt_version")
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

        except FileNotFoundError:
            logger.warning(f"Template '{template_name}' not found, using inline fallback")
            return self._finalize_system_prompt(self._fallback_system_prompt(template_context or {}))
        except Exception as e:
            logger.error(f"Template loading error for '{template_name}': {e}")
            return self._finalize_system_prompt(self._fallback_system_prompt(template_context or {}))

    def _fallback_system_prompt(self, context: dict) -> str:
        """Inline fallback prompt when template is not found."""
        return (
            "You are a job scheduler specialist working with Airflow.\n\n"
            "Available tools are listed in your tool definitions. "
            "For managing scheduled jobs, follow these guidelines:\n"
            "1. Use submit_sql_job or submit_sparksql_job to create new scheduled jobs\n"
            "2. Use get_scheduler_job and list_scheduler_jobs to check job status\n"
            "3. Use list_job_runs and get_run_log to troubleshoot failures\n"
            "4. Use update_job to modify existing job SQL or configuration\n"
            "5. Use pause_job / resume_job to control job scheduling\n"
            "6. Use trigger_scheduler_job for manual test runs\n\n"
            "Common cron expressions: '0 8 * * *' (daily 8am), '0 * * * *' (hourly), "
            "'0 9 * * 1' (Monday 9am), '0 0 1 * *' (1st of month)."
        )

    # ── Execution ───────────────────────────────────────────────────────

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

            template_context = self._prepare_template_context()
            system_instruction = self._get_system_prompt(
                conversation_summary, template_context, prompt_version=user_input.prompt_version
            )

            enhanced_message = user_input.user_message
            if user_input.database:
                enhanced_message = build_structured_content(
                    [
                        MessagePart(type="enhanced", content=f"Context: database={user_input.database}"),
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
                max_turns=self.max_turns,
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
                            response_content = str(raw_output)
                        elif raw_output:
                            response_content = str(raw_output)

            if not response_content and last_successful_output:
                raw_output = last_successful_output.get("raw_output", "")
                if isinstance(raw_output, dict):
                    response_content = str(raw_output)
                elif raw_output:
                    response_content = str(raw_output)
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

            # Extract structured result from tool outputs
            structured_result = None
            if last_successful_output and isinstance(last_successful_output, dict):
                result_data = last_successful_output.get("result")
                if isinstance(result_data, dict):
                    structured_result = result_data

            result = SchedulerNodeResult(
                success=True,
                response=response_content,
                scheduler_result=structured_result,
                tokens_used=int(tokens_used),
            )

            self.actions.extend(action_history_manager.get_actions())

            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="scheduler_response",
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

            error_result = SchedulerNodeResult(
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
