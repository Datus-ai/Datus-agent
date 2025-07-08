from typing import Dict, List, Tuple

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.node_models import GenerateSQLInput, GenerateSQLResult, SQLContext, TableSchema, TableValue
from datus.tools.lineage_graph_tools import SchemaLineageTool
from datus.tools.llms_tools import LLMTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenerateSQLNode(Node):
    def execute(self):
        self.result = self._execute_generate_sql()

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
