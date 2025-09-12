from datetime import datetime
from typing import AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.compare_node_models import CompareInput, CompareResult
from datus.schemas.node_models import SQLContext
from datus.tools.llms_tools import LLMTool
from datus.tools.llms_tools.compare_sql import compare_sql_with_mcp_stream
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class CompareNode(Node):
    def execute(self):
        self.result = self._execute_compare()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute SQL comparison with streaming support."""
        async for action in self._compare_sql_stream(action_history_manager):
            yield action

    def setup_input(self, workflow: Workflow) -> Dict:
        # Use the expectation from input_data if provided, otherwise empty string
        expectation = self.input if isinstance(self.input, str) and self.input.strip() else ""

        next_input = CompareInput(
            sql_task=workflow.task,
            sql_context=workflow.get_last_sqlcontext(),
            expectation=expectation,
        )
        self.input = next_input
        return {"success": True, "message": "Compare input setup complete", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict:
        """Update comparison results to workflow context."""
        result = self.result
        try:
            # Add comparison result as a new SQL context for reference
            new_record = SQLContext(
                sql_query=self.input.sql_context.sql_query,
                explanation=f"Comparison Analysis: {result.explanation}. Suggestions: {result.suggest}",
            )
            workflow.context.sql_contexts.append(new_record)
            return {"success": True, "message": "Updated comparison context"}
        except Exception as e:
            logger.error(f"Failed to update comparison context: {str(e)}")
            return {"success": False, "message": f"Comparison context update failed: {str(e)}"}

    def _execute_compare(self) -> CompareResult:
        """Execute comparison analysis between SQL and expectation."""

        if not self.model:
            return CompareResult(
                success=False,
                error="SQL comparison model not provided",
                explanation="",
                suggest="",
            )

        try:
            tool = LLMTool(self.model)
            logger.debug(f"Compare SQL input: {type(self.input)} {self.input}")

            return tool.compare_sql(self.input)
        except Exception as e:
            logger.error(f"SQL comparison execution error: {str(e)}")
            return CompareResult(success=False, error=str(e), explanation="", suggest="")

    async def _compare_sql_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Compare SQL with streaming support and action history tracking."""
        if not self.model:
            logger.error("Model not available for SQL comparison")
            return

        try:
            # Setup comparison context action
            setup_action = ActionHistory(
                action_id="setup_comparison",
                role=ActionRole.WORKFLOW,
                messages="Setting up SQL comparison with database context",
                action_type="comparison_setup",
                input={
                    "database_type": self.input.sql_task.database_type,
                    "database_name": self.input.sql_task.database_name,
                    "task": self.input.sql_task.task,
                    "sql_query": self.input.sql_context.sql_query,
                    "expectation": self.input.expectation,
                },
                status=ActionStatus.SUCCESS,
            )
            yield setup_action

            # Update setup action with success
            setup_action.output = {
                "success": True,
                "comparison_input_prepared": True,
                "database_name": self.input.sql_task.database_name,
                "has_expectation": bool(self.input.expectation),
            }
            setup_action.end_time = datetime.now()

            # Stream the comparison process
            async for action in compare_sql_with_mcp_stream(
                model=self.model,
                input_data=self.input,
                tools=self.tools,
                tool_config={"max_turns": 10},
                action_history_manager=action_history_manager,
            ):
                yield action

        except Exception as e:
            logger.error(f"SQL comparison streaming error: {str(e)}")
            raise
