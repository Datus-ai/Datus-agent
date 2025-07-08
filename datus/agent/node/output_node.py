from typing import Any, Dict

from datus.agent.node import Node
from datus.agent.workflow import Workflow
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
