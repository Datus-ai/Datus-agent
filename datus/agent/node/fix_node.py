from typing import Dict

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.fix_node_models import FixInput, FixResult
from datus.schemas.node_models import SQLContext
from datus.tools.llms_tools import LLMTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class FixNode(Node):
    def execute(self):
        self.result = self._execute_fix()

    def setup_input(self, workflow: Workflow) -> Dict:
        # irrelevant to current node
        next_input = FixInput(
            sql_task=workflow.task,
            sql_context=workflow.get_last_sqlcontext(),
            schemas=workflow.context.table_schemas,
        )
        self.input = next_input
        return {"success": True, "message": "Schema appears valid", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict:
        """Update fix SQL results to workflow context."""
        result = self.result
        try:
            new_record = SQLContext(sql_query=result.sql_query, explanation=result.explanation or "")
            workflow.context.sql_contexts.append(new_record)
            return {"success": True, "message": "Updated fix SQL context"}
        except Exception as e:
            logger.error(f"Failed to update fix SQL context: {str(e)}")
            return {"success": False, "message": f"Fix SQL context update failed: {str(e)}"}

    def _execute_fix(self) -> FixResult:
        """Execute fix action to fix the SQL query."""

        if not self.model:
            return FixResult(
                success=False,
                error="SQL fix model not provided",
                sql_query="",
                explanation="",
            )

        try:
            tool = LLMTool(self.model)
            logger.debug(f"Fix SQL input: {type(self.input)} {self.input}")

            # ToDo: add docs from search tools
            return tool.autofix_sql(self.input, docs=[])
        except Exception as e:
            logger.error(f"SQL fix execution error: {str(e)}")
            return FixResult(success=False, error=str(e), sql_query="", explanation="")
