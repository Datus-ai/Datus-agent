"""CompareAgenticNode shim for backwards compatibility."""

from typing import Any, Dict, List, Literal, Optional

from datus.agent.node.agentic_node import AgenticNode
from datus.agent.node.stream_run_context import StreamRunContext
from datus.configuration.agent_config import AgentConfig
from datus.prompts.prompt_manager import get_prompt_manager
from datus.schemas.compare_node_models import CompareInput, CompareResult
from datus.tools.func_tool import DBFuncTool
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class CompareAgenticNode(AgenticNode):
    """
    Agentic node implementation for SQL comparison.

    This node leverages the AgenticNode base class to provide session-aware
    streaming interactions while supporting the legacy synchronous compare
    workflow. It prepares comparison prompts, manages tool execution, and
    produces structured comparison results.
    """

    result_class = CompareResult

    def __init__(
        self,
        node_name: str = "compare",
        agent_config: Optional[AgentConfig] = None,
        execution_mode: Literal["interactive", "workflow"] = "interactive",
        is_subagent: bool = False,
        session_id: Optional[str] = None,
    ):
        """
        Initialize CompareAgenticNode.

        Args:
            node_name: Name of the node configuration in agent.yml (default: "compare")
            agent_config: Agent configuration
            execution_mode: ``"interactive"`` (default) enables session
                management, auto-compaction, and token accounting;
                ``"workflow"`` skips those for unattended pipelines.
            is_subagent: When True, skip SubAgentTaskTool setup (2-level depth enforcement)
        """
        self.configured_node_name = node_name
        self.execution_mode = execution_mode

        # Use TYPE_COMPARE as the node type
        from datus.configuration.node_type import NodeType

        node_type = NodeType.TYPE_COMPARE

        # Call parent constructor with all required Node parameters
        super().__init__(
            node_id=f"{node_name}_node",
            description=f"SQL comparison node: {node_name}",
            node_type=node_type,
            input_data=None,
            agent_config=agent_config,
            tools=[],
            mcp_servers={},
            is_subagent=is_subagent,
            session_id=session_id,
        )

        # Get max_turns from agentic_nodes configuration, default to 50
        self.max_turns = 50
        if agent_config and hasattr(agent_config, "agentic_nodes") and node_name in agent_config.agentic_nodes:
            agentic_node_config = agent_config.agentic_nodes[node_name]
            if isinstance(agentic_node_config, dict):
                self.max_turns = agentic_node_config.get("max_turns", 50)

        self.setup_tools()

    def get_node_name(self) -> str:
        """
        Get the configured node name for this SQL summary agentic node.

        Returns:
            The configured node name from agent.yml
        """
        return self.configured_node_name

    def setup_tools(self) -> None:
        """
        Prepare default database and context tools when they are not explicitly provided.
        """

        if not self.agent_config:
            logger.debug("No agent configuration available; skipping tool setup.")
            return

        try:
            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=self.get_node_name(),
            )

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
    def _prepare_prompt_components(
        input_data: CompareInput, agent_config: Optional[Any] = None
    ) -> tuple[str, str, List[Dict[str, str]]]:
        """
        Render the system instruction, user prompt, and message list for comparison.
        """
        prompt_version = input_data.prompt_version

        pm = get_prompt_manager(agent_config=agent_config)
        system_instruction = pm.get_raw_template("compare_sql_system_mcp", version=prompt_version)

        sql_context = input_data.sql_context
        sql_query = getattr(sql_context, "sql_query", "")
        sql_explanation = getattr(sql_context, "explanation", "")
        sql_result = getattr(sql_context, "sql_return", "")
        sql_error = getattr(sql_context, "sql_error", "")

        user_prompt = pm.render_template(
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
                snippet = (
                    (raw_output[:300] + "...") if isinstance(raw_output, str) and len(raw_output) > 300 else raw_output
                )
                logger.warning(f"Failed to parse comparison output as JSON. Raw: {snippet}")
                return {
                    "explanation": f"Failed to parse comparison output as JSON. Raw: {snippet}",
                    "suggest": "Please verify the response format manually.",
                }
            return result

        logger.debug(f"Unexpected comparison output type: {type(raw_output)}")
        return {}

    # ── template hooks ──────────────────────────────────────────────────

    async def _before_stream(self, ctx: StreamRunContext) -> None:
        """Pre-render Compare's hand-written system + user prompts.

        ``_prepare_prompt_components`` returns both prompts together (they share
        Jinja context), so we cache the system instruction on ``self`` for the
        ``_get_system_prompt`` override and stash the rendered user prompt in
        ``ctx.extras`` for ``_build_template_context`` to surface as
        ``user_message_override``.
        """
        system_instruction, raw_user_prompt, _ = self._prepare_prompt_components(
            ctx.user_input, agent_config=self.agent_config
        )
        self._cached_system_instruction = system_instruction
        ctx.extras["compare_user_prompt"] = raw_user_prompt

    def _build_template_context(self, ctx: StreamRunContext) -> Optional[dict]:
        """Forward the rendered user prompt as ``user_message_override``.

        Returns ``None`` because Compare uses a pre-rendered system instruction,
        not template-driven rendering.
        """
        ctx.user_message_override = ctx.extras.pop("compare_user_prompt", "")
        return None

    def _get_system_prompt(
        self,
        prompt_version: Optional[str] = None,
        template_context: Optional[dict] = None,
    ) -> str:
        # Compare bypasses the standard ``{node_name}_system`` resolution and
        # uses ``compare_sql_system_mcp`` rendered in ``_before_stream``.
        return getattr(self, "_cached_system_instruction", "")

    def _build_success_result(self, ctx: StreamRunContext) -> CompareResult:
        response_content = ctx.response_content
        if not response_content and ctx.last_successful_output:
            response_content = ctx.last_successful_output.get("raw_output", "")

        result_dict = self._parse_comparison_output(response_content)
        tokens_used = self._extract_total_tokens(ctx.action_history_manager.get_actions())
        return CompareResult(
            success=True,
            explanation=result_dict.get("explanation", "No explanation provided"),
            suggest=result_dict.get("suggest", "No suggestions provided"),
            tokens_used=tokens_used,
        )
