from typing import Any, AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import OutputInput
from datus.tools.output_tools import BenchmarkOutputTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class OutputNode(Node):
    def execute(self):
        self.result = self._execute_output()

    def setup_input(self, workflow: Workflow) -> Dict:
        sql_context = workflow.get_last_sqlcontext()
        # normally last node of workflow
        next_input = OutputInput(
            finished=True,
            task_id=workflow.task.id,
            task=workflow.get_task(),
            database_name=workflow.task.database_name,
            output_dir=workflow.task.output_dir,
            gen_sql=sql_context.sql_query,
            sql_result=sql_context.sql_return,
            row_count=sql_context.row_count,
            table_schemas=workflow.context.table_schemas,
            metrics=workflow.context.metrics,
            external_knowledge=workflow.task.external_knowledge,
        )
        self.input = next_input
        return {"success": True, "message": "Output appears valid", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Any:
        return {"success": True, "message": "Output node, no context update needed"}

    def _execute_output(self) -> Any:
        """Execute output action to present the results."""
        tool = BenchmarkOutputTool()
        return tool.execute(self.input, sql_connector=self._sql_connector(), model=self.model)
        # return BaseResult(success=True, error="")

    async def _output_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute output generation with streaming support and action history tracking."""
        try:
            # Output preparation action
            preparation_action = ActionHistory(
                action_id="output_preparation",
                role=ActionRole.WORKFLOW,
                messages="Preparing final output and results formatting",
                action_type="output_preparation",
                input={
                    "task_id": self.input.task_id if hasattr(self.input, "task_id") else "",
                    "database_name": self.input.database_name if hasattr(self.input, "database_name") else "",
                    "has_sql_result": bool(getattr(self.input, "sql_result", None)),
                    "row_count": getattr(self.input, "row_count", 0),
                    "finished": getattr(self.input, "finished", False),
                },
                status=ActionStatus.PROCESSING,
            )
            yield preparation_action

            # Update preparation status
            try:
                preparation_action.status = ActionStatus.SUCCESS
                preparation_action.output = {
                    "preparation_complete": True,
                    "output_ready": True,
                }
            except Exception as e:
                preparation_action.status = ActionStatus.FAILED
                preparation_action.output = {"error": str(e)}
                logger.warning(f"Output preparation failed: {e}")

            # Output generation action
            generation_action = ActionHistory(
                action_id="output_generation",
                role=ActionRole.WORKFLOW,
                messages="Generating final output with results and benchmark data",
                action_type="output_generation",
                input={
                    "gen_sql": self.input.gen_sql[:100] + "..."
                    if hasattr(self.input, "gen_sql") and len(self.input.gen_sql) > 100
                    else getattr(self.input, "gen_sql", ""),
                    "output_dir": getattr(self.input, "output_dir", ""),
                    "task_completed": getattr(self.input, "finished", False),
                },
                status=ActionStatus.PROCESSING,
            )
            yield generation_action

            # Execute output generation - reuse existing logic
            try:
                result = self._execute_output()

                generation_action.status = ActionStatus.SUCCESS
                generation_action.output = {
                    "output_generated": True,
                    "has_benchmark_data": bool(result),
                    "success": getattr(result, "success", True) if result else True,
                }

                # Store result for later use
                self.result = result

            except Exception as e:
                generation_action.status = ActionStatus.FAILED
                generation_action.output = {"error": str(e)}
                logger.error(f"Output generation error: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Output generation streaming error: {str(e)}")
            raise

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute output generation with streaming support."""
        async for action in self._output_stream(action_history_manager):
            yield action
