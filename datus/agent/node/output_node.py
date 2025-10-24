# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import OutputInput
from datus.tools.output_tools import OutputTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class OutputNode(Node):
    def execute(self):
        self.result = self._execute_output()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute output generation with streaming support."""
        async for action in self._output_stream(action_history_manager):
            yield action

    def setup_input(self, workflow: Workflow) -> Dict:
        sql_context = workflow.get_last_sqlcontext()
        # normally last node of workflow
        next_input = OutputInput(
            finished=True,
            task_id=workflow.task.id,
            task=workflow.get_task(),
            database_name=workflow.task.database_name,
            output_dir=workflow.task.output_dir,
            gen_sql=sql_context.sql_query or "",
            sql_result=sql_context.sql_return or "",
            row_count=sql_context.row_count or 0,
            table_schemas=workflow.context.table_schemas,
            metrics=workflow.context.metrics,
            external_knowledge=workflow.task.external_knowledge,
            error=sql_context.sql_error,
        )
        self.input = next_input
        return {"success": True, "message": "Output appears valid", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Any:
        return {"success": True, "message": "Output node, no context update needed"}

    def _execute_output(self) -> Any:
        """Execute output action to present the results."""
        tool = OutputTool()
        return tool.execute(self.input, sql_connector=self._sql_connector(self.input.database_name), model=self.model)
        # return BaseResult(success=True, error="")

    async def _output_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute output generation with streaming support and action history tracking."""
        try:
            # Output generation action
            generation_action = ActionHistory(
                action_id="output_generation",
                role=ActionRole.WORKFLOW,
                messages="Generating final output with results and benchmark data",
                action_type="output_generation",
                input={
                    "task_id": self.input.task_id if hasattr(self.input, "task_id") else "",
                    "has_sql_result": bool(getattr(self.input, "sql_result", None)),
                    "row_count": getattr(self.input, "row_count", 0),
                },
                status=ActionStatus.PROCESSING,
            )
            yield generation_action

            # Execute output generation
            result = self._execute_output()

            generation_action.status = ActionStatus.SUCCESS
            generation_action.output = {
                "output_generated": True,
                "has_benchmark_data": bool(result),
                "success": getattr(result, "success", True) if result else True,
                "sql_query": getattr(self.input, "gen_sql", "") if hasattr(self.input, "gen_sql") else "",
                "sql_result": getattr(self.input, "sql_result", "") if hasattr(self.input, "sql_result") else "",
            }

            # Store result for later use
            self.result = result

            # Yield the updated action with final status
            yield generation_action

        except Exception as e:
            logger.error(f"Output generation streaming error: {str(e)}")
            raise
