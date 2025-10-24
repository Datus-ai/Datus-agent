# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import AsyncGenerator, Dict, Optional

from pydantic import ValidationError

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import ExecuteSQLInput, ExecuteSQLResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ExecuteSQLNode(Node):
    def execute(self):
        self.result = self._execute_sql()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute SQL execution with streaming support."""
        async for action in self._execute_sql_stream(action_history_manager):
            yield action

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
            db_connector = self._sql_connector(self.input.database_name)
            if not db_connector:
                logger.error("Database connection not initialized in workflow")
                return ExecuteSQLResult(
                    success=False,
                    error="Database connection not initialized in workflow",
                )
            logger.debug(f"SQL execution input: {self.input}")
            result = db_connector.execute(self.input)
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

    async def _execute_sql_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute SQL with streaming support and action history tracking."""
        try:
            # Database connection action
            connection_action = ActionHistory(
                action_id="database_connection",
                role=ActionRole.WORKFLOW,
                messages="Establishing database connection for SQL execution",
                action_type="database_connection",
                input={
                    "database_name": self.input.database_name if hasattr(self.input, "database_name") else "",
                    "sql_query_preview": self.input.sql_query[:50] + "..."
                    if hasattr(self.input, "sql_query") and len(self.input.sql_query) > 50
                    else getattr(self.input, "sql_query", ""),
                },
                status=ActionStatus.PROCESSING,
            )
            yield connection_action

            # Initialize database connection
            try:
                sql_connector = self._sql_connector(self.input.database_name)

                if not sql_connector:
                    connection_action.status = ActionStatus.FAILED
                    connection_action.output = {"error": "Database connection not initialized"}
                    logger.error("Database connection not initialized in workflow")
                    return

                connection_action.status = ActionStatus.SUCCESS
                connection_action.output = {
                    "connection_established": True,
                    "database_connected": True,
                }
            except Exception as e:
                connection_action.status = ActionStatus.FAILED
                connection_action.output = {"error": str(e)}
                logger.error(f"Database connection failed: {e}")
                raise

            # SQL execution action
            execution_action = ActionHistory(
                action_id="sql_execution",
                role=ActionRole.WORKFLOW,
                messages="Executing SQL query against the database",
                action_type="sql_execution",
                input={
                    "sql_query": self.input.sql_query if hasattr(self.input, "sql_query") else "",
                    "database_name": self.input.database_name if hasattr(self.input, "database_name") else "",
                },
                status=ActionStatus.PROCESSING,
            )
            yield execution_action

            # Execute SQL - reuse existing logic
            try:
                result = self._execute_sql()

                execution_action.status = ActionStatus.SUCCESS if result.success else ActionStatus.FAILED
                execution_action.output = {
                    "success": result.success,
                    "row_count": result.row_count if hasattr(result, "row_count") else 0,
                    "has_results": bool(result.sql_return) if hasattr(result, "sql_return") else False,
                    "sql_result": result.sql_return if hasattr(result, "sql_return") else None,
                    "error": result.error if hasattr(result, "error") and result.error else None,
                }

                # Store result for later use
                self.result = result

            except Exception as e:
                execution_action.status = ActionStatus.FAILED
                execution_action.output = {"error": str(e)}
                logger.error(f"SQL execution error: {str(e)}")
                raise

            # Yield the updated execution action with final status
            yield execution_action

        except Exception as e:
            logger.error(f"SQL execution streaming error: {str(e)}")
            raise
