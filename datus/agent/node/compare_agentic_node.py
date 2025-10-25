"""CompareAgenticNode shim for backwards compatibility."""

from typing import Any, AsyncGenerator, Dict, List, Optional

from agents import Tool
from agents.mcp import MCPServerStdio

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.compare_node_models import CompareInput, CompareResult
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.func_tool import DBFuncTool
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


class CompareAgenticNode(AgenticNode):
    """
    Agentic node implementation for SQL comparison.

    This node leverages the AgenticNode base class to provide session-aware
    streaming interactions while supporting the legacy synchronous compare
    workflow. It prepares comparison prompts, manages tool execution, and
    produces structured comparison results.
    """

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        *,
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerStdio]] = None,
        model: Optional[LLMBaseModel] = None,
        max_turns: Optional[int] = None,
    ):
        # Consider None or empty list as "not provided"
        self._tools_provided = bool(tools)
        self.max_turns = max_turns or 30
        super().__init__(
            tools=tools or [],
            mcp_servers=mcp_servers or {},
            agent_config=agent_config,
        )

        if model is not None:
            self.model = model

        config_max_turns = self.node_config.get("max_turns")
        if config_max_turns:
            self.max_turns = config_max_turns
        elif max_turns:
            self.max_turns = max_turns

        if not self._tools_provided:
            self.setup_tools()

    def setup_tools(self) -> None:
        """
        Prepare default database and context tools when they are not explicitly provided.
        """
        if self.tools:
            return

        if not self.agent_config:
            logger.debug("No agent configuration available; skipping tool setup.")
            return

        try:
            namespace = self.agent_config.current_namespace

            db_manager = db_manager_instance(self.agent_config.namespaces)
            database = getattr(self.agent_config, "current_database", "")
            try:
                connector = db_manager.get_conn(namespace, database)
            except Exception:
                connector = db_manager.first_conn(namespace)

            self.db_func_tool = DBFuncTool(connector, agent_config=self.agent_config)

            self.tools = self.db_func_tool.available_tools()
            logger.debug(
                "CompareAgenticNode configured %d tools: %s",
                len(self.tools),
                [tool.name for tool in self.tools],
            )
        except Exception as exc:
            logger.error(f"Failed to initialize tools for CompareAgenticNode: {exc}")
            self.tools = self.tools or []

    @staticmethod
    def _prepare_prompt_components(input_data: CompareInput) -> tuple[str, str, List[Dict[str, str]]]:
        """
        Render the system instruction, user prompt, and message list for comparison.
        """
        prompt_version = input_data.prompt_version or "1.0"

        system_instruction = prompt_manager.get_raw_template("compare_sql_system_mcp", version=prompt_version)

        sql_context = input_data.sql_context
        sql_query = getattr(sql_context, "sql_query", "")
        sql_explanation = getattr(sql_context, "explanation", "")
        sql_result = getattr(sql_context, "sql_return", "")
        sql_error = getattr(sql_context, "sql_error", "")

        user_prompt = prompt_manager.render_template(
            "compare_sql_user",
            database_type=input_data.sql_task.database_type,
            database_name=input_data.sql_task.database_name,
            sql_task=input_data.sql_task.task,
            external_knowledge=input_data.sql_task.external_knowledge,
            sql_query=sql_query,
            sql_explanation=sql_explanation,
            sql_result=sql_result,
            sql_error=sql_error,
            expectation=input_data.expectation,
            version=prompt_version,
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ]

        return system_instruction, user_prompt, messages

    @staticmethod
    def _parse_comparison_output(raw_output: Any) -> Dict[str, str]:
        """
        Convert model output into a dictionary with explanation and suggestions.
        """
        if isinstance(raw_output, dict):
            return raw_output

        if raw_output is None:
            return {}

        if isinstance(raw_output, str):
            result = llm_result2json(raw_output, expected_type=dict)
            if result is None:
                logger.warning(f"Failed to parse comparison output as JSON: {result}")
                return {
                    "explanation": result or "Failed to parse model response.",
                    "suggest": "Please verify the response format manually.",
                }
            return result

        logger.debug(f"Unexpected comparison output type: {type(raw_output)}")
        return {}

    @optional_traceable()
    async def execute_stream(
        self,
        user_input: CompareInput,
        action_history_manager: Optional[ActionHistoryManager] = None,
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute SQL comparison with streaming support and action history tracking.
        """
        if not isinstance(user_input, CompareInput):
            raise ValueError("Input must be a CompareInput instance")

        if not self.model:
            raise ValueError("Model is not initialized for CompareAgenticNode")

        if action_history_manager is None:
            action_history_manager = ActionHistoryManager()

        user_action = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type="compare_sql_request",
            messages=f"Compare SQL task: {user_input.sql_task.task}",
            input_data=user_input.model_dump(),
            status=ActionStatus.PROCESSING,
        )
        action_history_manager.add_action(user_action)
        yield user_action

        try:
            await self._auto_compact()
            session, conversation_summary = self._get_or_create_session()

            system_instruction, user_prompt, _ = self._prepare_prompt_components(user_input)
            if conversation_summary:
                user_prompt = (
                    f"Previous conversation summary:\n{conversation_summary}\n\n"
                    f"New comparison request:\n{user_prompt}"
                )

            assistant_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="llm_generation",
                messages="Analyzing SQL comparison with tools...",
                input_data={"prompt": user_prompt, "system": system_instruction},
                status=ActionStatus.PROCESSING,
            )
            action_history_manager.add_action(assistant_action)
            yield assistant_action

            response_content: Any = ""
            last_successful_output: Optional[Dict[str, Any]] = None

            async for stream_action in self.model.generate_with_tools_stream(
                prompt=user_prompt,
                tools=self.tools or [],
                mcp_servers=self.mcp_servers or {},
                instruction=system_instruction,
                max_turns=self.max_turns,
                session=session,
                action_history_manager=action_history_manager,
            ):
                yield stream_action

                if stream_action.status == ActionStatus.SUCCESS and stream_action.output:
                    output_value = stream_action.output
                    if isinstance(output_value, dict):
                        last_successful_output = output_value
                        raw_output = output_value.get("raw_output")
                        if raw_output:
                            response_content = raw_output
                    else:
                        response_content = output_value

            if not response_content and last_successful_output:
                logger.debug(f"Trying to extract response from last_successful_output: {last_successful_output}")
                response_content = last_successful_output.get("raw_output", "")

            result_dict = self._parse_comparison_output(response_content)
            tokens_used = 0

            for action in reversed(action_history_manager.get_actions()):
                if action.role == "assistant" and isinstance(action.output, dict):
                    usage = action.output.get("usage", {})
                    if isinstance(usage, dict):
                        conversation_tokens = usage.get("total_tokens") or usage.get("output_tokens")
                        if conversation_tokens:
                            tokens_used = int(conversation_tokens)
                            self._add_session_tokens(tokens_used)
                            break

            result = CompareResult(
                success=True,
                explanation=result_dict.get("explanation", "No explanation provided"),
                suggest=result_dict.get("suggest", "No suggestions provided"),
                tokens_used=tokens_used,
            )

            action_history_manager.update_action_by_id(
                assistant_action.action_id,
                status=ActionStatus.SUCCESS,
                output=result.model_dump(),
                messages="Comparison completed successfully.",
            )

            self.actions.extend(action_history_manager.get_actions())

            final_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="compare_sql_response",
                messages="Comparison completed successfully.",
                input_data=user_input.model_dump(),
                output_data=result.model_dump(),
                status=ActionStatus.SUCCESS,
            )
            action_history_manager.add_action(final_action)
            yield final_action

        except Exception as exc:
            logger.error(f"CompareAgenticNode streaming execution failed: {exc}")

            error_result = CompareResult(
                success=False,
                error=str(exc),
                explanation="Comparison analysis failed",
                suggest="Please check the input parameters and try again",
            )

            action_history_manager.update_current_action(
                status=ActionStatus.FAILED,
                output=error_result.model_dump(),
                messages=f"Comparison failed: {exc}",
            )

            error_action = ActionHistory.create_action(
                role=ActionRole.ASSISTANT,
                action_type="compare_sql_error",
                messages=f"Comparison failed: {exc}",
                input_data=user_input.model_dump(),
                output_data=error_result.model_dump(),
                status=ActionStatus.FAILED,
            )
            action_history_manager.add_action(error_action)
            yield error_action
