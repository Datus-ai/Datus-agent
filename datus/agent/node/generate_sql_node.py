from typing import AsyncGenerator, Dict, List, Optional, Tuple

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import GenerateSQLInput, GenerateSQLResult, SQLContext, TableSchema, TableValue
from datus.tools.lineage_graph_tools import SchemaLineageTool
from datus.tools.llms_tools import LLMTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenerateSQLNode(Node):
    def execute(self):
        self.result = self._execute_generate_sql()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute SQL generation with streaming support."""
        async for action in self._generate_sql_stream(action_history_manager):
            yield action

    def setup_input(self, workflow: Workflow) -> Dict:
        if workflow.context.document_result:
            database_docs = "\n Reference documents:\n"
            for _, docs in workflow.context.document_result.docs.items():
                database_docs += "\n".join(docs) + "\n"
        else:
            database_docs = ""
        # irrelevant to current node
        next_input = GenerateSQLInput(
            database_type=workflow.task.database_type,
            sql_task=workflow.task,
            table_schemas=workflow.context.table_schemas,
            data_details=workflow.context.table_values,
            metrics=workflow.context.metrics,
            contexts=workflow.context.sql_contexts,
            external_knowledge=workflow.task.external_knowledge,
            database_docs=database_docs,
        )
        self.input = next_input
        return {"success": True, "message": "Schema appears valid", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict:
        """Update SQL generation results to workflow context."""
        result = self.result
        try:
            # Create new SQL context record and add to context
            new_record = SQLContext(sql_query=result.sql_query, explanation=result.explanation or "")
            workflow.context.sql_contexts.append(new_record)

            # Get and update schema information
            table_schemas, table_values = self._get_schema_and_values(workflow.task.database_name, result.tables)
            if len(table_schemas) == len(result.tables) and len(table_values) == len(result.tables):
                workflow.context.table_schemas = table_schemas
                workflow.context.table_values = table_values
                return {"success": True, "message": "Updated SQL generation context"}
            else:
                error_msg = (
                    f"Failed to get schemas and values for tables {result.tables} " f"{workflow.task.database_name}"
                )
                logger.warning(f"{error_msg}, table_schemas: {table_schemas}, table_values: {table_values}")
                return {"success": True, "message": error_msg}
        except Exception as e:
            logger.error(f"Failed to update SQL generation context: {str(e)}")
            return {"success": False, "message": f"SQL generation context update failed: {str(e)}"}

    def _execute_generate_sql(self) -> GenerateSQLResult:
        """Execute SQL generation action to create SQL query.

        Combines input data from previous nodes into a structured format for SQL generation.
        The input data includes:
        - table_schemas: Database schema information from schema linking
        - data_details: Additional data context
        - metrics: Relevant metrics information
        - database: Database type information

        Returns:
            GenerateSQLResult containing the generated SQL query
        """
        if not self.model:
            return GenerateSQLResult(
                success=False,
                error="SQL generation model not provided",
                sql_query="",
                tables=[],
                explanation=None,
            )

        try:
            tool = LLMTool(self.model)
            logger.debug(f"Generate SQL input: {type(self.input)} {self.input}")
            return tool.generate_sql(self.input)
        except Exception as e:
            logger.error(f"SQL generation execution error: {str(e)}")
            return GenerateSQLResult(success=False, error=str(e), sql_query="", tables=[], explanation=None)

    def _get_schema_and_values(
        self, database_name: str, table_names: List[str]
    ) -> Tuple[List[TableSchema], List[TableValue]]:
        """Get table schemas and values using the schema lineage tool."""
        try:
            # Get the schema lineage tool instance
            schema_tool = SchemaLineageTool(agent_config=self.agent_config)

            # Use the tool to get schemas and values
            logger.debug(f"Getting schemas and values for tables {table_names} from {database_name}")
            return schema_tool.get_table_and_values(database_name, table_names)
        except Exception as e:
            logger.warning(f"Failed to get schemas and values for tables {table_names}: {e}")
            return [], []  # Return empty lists if lookup fails

    async def _generate_sql_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate SQL with streaming support and action history tracking."""
        if not self.model:
            logger.error("Model not available for SQL generation")
            return

        try:
            # SQL generation preparation action
            prep_action = ActionHistory(
                action_id="sql_generation_prep",
                role=ActionRole.WORKFLOW,
                messages="Preparing SQL generation with schema and context information",
                action_type="sql_preparation",
                input={
                    "database_type": self.input.database_type if hasattr(self.input, "database_type") else "",
                    "table_count": len(self.input.table_schemas)
                    if hasattr(self.input, "table_schemas") and self.input.table_schemas
                    else 0,
                    "has_metrics": bool(hasattr(self.input, "metrics") and self.input.metrics),
                    "has_external_knowledge": bool(
                        hasattr(self.input, "external_knowledge") and self.input.external_knowledge
                    ),
                },
                status=ActionStatus.PROCESSING,
            )
            yield prep_action

            # Update preparation status
            try:
                prep_action.status = ActionStatus.SUCCESS
                prep_action.output = {
                    "preparation_complete": True,
                    "input_validated": True,
                }
            except Exception as e:
                prep_action.status = ActionStatus.FAILED
                prep_action.output = {"error": str(e)}
                logger.warning(f"SQL preparation failed: {e}")

            # SQL generation action
            generation_action = ActionHistory(
                action_id="sql_generation",
                role=ActionRole.WORKFLOW,
                messages="Generating SQL query based on schema and requirements",
                action_type="sql_generation",
                input={
                    "task_description": getattr(self.input.sql_task, "task", "")
                    if hasattr(self.input, "sql_task")
                    else "",
                    "database_type": self.input.database_type if hasattr(self.input, "database_type") else "",
                },
                status=ActionStatus.PROCESSING,
            )
            yield generation_action

            # Execute SQL generation - reuse existing logic
            try:
                result = self._execute_generate_sql()

                generation_action.status = ActionStatus.SUCCESS
                generation_action.output = {
                    "success": result.success,
                    "sql_query": result.sql_query,
                    "tables_involved": result.tables if result.tables else [],
                    "has_explanation": bool(result.explanation),
                }

                # Store result for later use
                self.result = result

            except Exception as e:
                generation_action.status = ActionStatus.FAILED
                generation_action.output = {"error": str(e)}
                logger.error(f"SQL generation error: {str(e)}")
                raise

            # Yield the updated generation action with final status
            yield generation_action

        except Exception as e:
            logger.error(f"SQL generation streaming error: {str(e)}")
            raise
