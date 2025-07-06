from typing import Dict

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult
from datus.tools.lineage_graph_tools.schema_lineage import SchemaLineageTool
from datus.utils.loggings import get_logger

logger = get_logger("schema_link_node")


class SchemaLinkingNode(Node):
    def execute(self):
        self.result = self._execute_schema_linking()

    def update_context(self, workflow: Workflow) -> Dict:
        """Update schema linking results to workflow context."""
        result = self.result
        try:
            if len(workflow.context.table_schemas) == 0:
                workflow.context.table_schemas = result.table_schemas
                workflow.context.table_values = result.table_values
            else:
                pass  # if it's not the first schema linking, wait it after execute_sql

            return {"success": True, "message": "Updated schema linking context"}
        except Exception as e:
            logger.error(f"Failed to update schema linking context: {str(e)}")
            return {"success": False, "message": f"Schema linking context update failed: {str(e)}"}

    def setup_input(self, workflow: Workflow) -> Dict:
        logger.info("Setup schema linking input")
        # irrelevant to current node: it should be Start or Reflection node now
        matching_rate = self.agent_config.schema_linking_rate
        matching_rates = ["fast", "medium", "slow", "from_llm"]
        start = matching_rates.index(matching_rate)
        final_matching_rate = matching_rates[min(start + workflow.reflection_round, len(matching_rates) - 1)]
        logger.debug(f"Final matching rate: {final_matching_rate}")
        next_input = SchemaLinkingInput(
            input_text=workflow.task.task,
            matching_rate=final_matching_rate,
            database_type=workflow.task.database_type,
            database_name=workflow.task.database_name,
            sql_context=workflow.get_last_sqlcontext(),
            table_type=workflow.task.schema_linking_type,
        )
        self.input = next_input
        return {"success": True, "message": "Schema appears valid"}

    def _execute_schema_linking(self) -> SchemaLinkingResult:
        """Execute schema linking action to analyze database schema.
        Input:
            query - The input query to analyze.
        Returns:
            A validated SchemaLinkingResult containing table schemas and values.
        """
        import os

        path = self.agent_config.rag_storage_path()
        logger.debug(f"Checking if rag storage path exists: {path}")
        if not os.path.exists(path):
            logger.info(f"RAG storage path `{path}` does not exist.")
            return self._execute_schema_linking_fallback(SchemaLineageTool())
        else:
            tool = SchemaLineageTool(db_path=path)
            try:
                # Import SchemaLineageTool only when needed
                if tool:
                    result = tool.execute(self.input, self.model)
                    if not result.success:
                        logger.warning(f"Schema linking failed: {result.error}")
                        return self._execute_schema_linking_fallback(tool)
                    logger.info(f"Schema linking result: found {len(result.table_schemas)} tables")
                    if len(result.table_schemas) > 0:
                        return result
                    logger.info("No tables found, using fallback method")
                    return self._execute_schema_linking_fallback(tool)
                else:
                    logger.warning("Schema linking tool not found")
                    return self._execute_schema_linking_fallback(tool)
            except Exception as e:
                logger.warning(f"Schema linking tool initialization failed: {e}")
                return self._execute_schema_linking_fallback(tool)

    def _execute_schema_linking_fallback(self, tool: SchemaLineageTool) -> SchemaLinkingResult:
        # Fallback: directly get tables from current database
        logger.info("Get tables directly from database")
        try:
            # Get database connector through db_manager
            from datus.tools.db_tools.db_manager import db_manager_instance

            db_manager = db_manager_instance(self.agent_config.namespaces)

            # Get current namespace and database connection
            current_namespace = self.agent_config.current_namespace
            database_name = self.input.database_name if hasattr(self.input, "database_name") else ""

            # Get database connector
            connector = db_manager.get_conn(current_namespace, self.agent_config.db_type, database_name)

            return tool.get_schems_by_db(connector=connector, input_param=self.input)

        except Exception as e:
            logger.warning(f"Schema linking failed: {e}")
            return SchemaLinkingResult(
                success=False,
                error=f"Schema linking failed: {e}",
                schema_count=0,
                value_count=0,
                table_schemas=[],
                table_values=[],
            )
