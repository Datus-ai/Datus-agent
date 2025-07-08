from typing import Dict

from pydantic import ValidationError

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.node_models import ExecuteSQLInput, ExecuteSQLResult
from datus.tools.db_tools import DBTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ExecuteSQLNode(Node):
    def execute(self):
        self.result = self._execute_sql()

    def setup_input(self, workflow: Workflow) -> Dict:
        next_input = ExecuteSQLInput(
            sql_query=self._strip_sql_markdown(workflow.get_last_sqlcontext().sql_query),
            database_name=workflow.task.database_name,
        )
        self.input = next_input
        return {"success": True, "message": "Node input appears valid", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict:
        """Update SQL execution results to workflow context."""
        result = self.result
        try:
            last_record = workflow.context.sql_contexts[-1]
            last_record.sql_return = result.sql_return
            last_record.row_count = result.row_count
            last_record.sql_error = result.error
            # TODO: check if the sql_query is the same as the last one
            # if last_record.sql_query == result.sql_query:
            #    last_record.sql_return = result.sql_return
            #    last_record.row_count = result.row_count
            return {"success": True, "message": "Updated SQL execution context"}
        except Exception as e:
            logger.error(f"Failed to update SQL execution context: {str(e)}")
            return {"success": False, "message": f"SQL execution context update failed: {str(e)}"}

    def _strip_sql_markdown(self, text: str) -> str:
        """Strip markdown SQL code block markers from text.

        Args:
            text (str): Input text containing SQL code block with markdown markers

        Returns:
            str: SQL code with markdown markers removed

        Example:
            >>> text = '''```sql
            ... SELECT * FROM table;
            ... ```'''
            >>> print(strip_sql_markdown(text))
            SELECT * FROM table;
        """
        if not isinstance(text, str):
            logger.warning(f"The input of sql to stripe is not a string: {text}")
            return text
        lines = text.split("\n")

        # Remove ```sql at start and ``` at end if present
        if lines and lines[0].strip() == "```sql":
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        # Join lines back together
        return "\n".join(lines)

    def _execute_sql(self) -> ExecuteSQLResult:
        """Execute SQL query action to run the generated query."""
        try:
            tool = DBTool(self._sql_connector())
            if not tool:
                logger.error("Database connection not initialized in workflow")
                return ExecuteSQLResult(
                    success=False,
                    error="Database connection not initialized in workflow",
                )
            logger.debug(f"SQL execution input: {self.input}")
            result = tool.execute(self.input)
            logger.debug(f"SQL execution result: {result}")
            return result
        except ValidationError as e:
            logger.error(f"SQL execution failed: {str(e)}")
            return ExecuteSQLResult(
                success=False,
                error=str(e),
                sql_query=self.input.sql_query if hasattr(self.input, "sql_query") else "",
            )
        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}")
            return ExecuteSQLResult(
                success=False,
                error=str(e),
                sql_query=self.input.sql_query if hasattr(self.input, "sql_query") else "",
            )
