from contextlib import contextmanager
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput, GenerateSemanticModelResult
from datus.storage.metric.init_utils import gen_semantic_model_id
from datus.storage.metric.store import rag_by_configuration
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.llms_tools import LLMTool
from datus.tools.llms_tools.generate_semantic_model import generate_semantic_model_with_mcp_stream
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_table_name_parts

logger = get_logger(__name__)


@contextmanager
def get_db_connector(db_manager, current_namespace, database_name):
    connector = None
    try:
        connector = db_manager.get_conn(current_namespace, database_name)
        yield connector
    finally:
        if connector:
            try:
                connector.close()
                logger.debug(f"Database connector closed for {database_name}")
            except Exception as e:
                logger.warning(f"Error closing database connector: {e}")


class GenerateSemanticModelNode(Node):
    """
    Node for generating semantic model.
    """

    def execute(self):
        self.result = self._generate_semantic_model()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute semantic model generation with streaming support."""
        async for action in self._generate_semantic_model_stream(action_history_manager):
            yield action

    def validate_input(self):
        if not isinstance(self.input, GenerateSemanticModelInput):
            raise ValueError("Input must be a GenerateSemanticModelInput instance")
        return True

    def _generate_semantic_model(self) -> GenerateSemanticModelResult:
        """Generate semantic model for the given SQL query."""
        if not self.model:
            return GenerateSemanticModelResult(
                success=False,
                error="Semantic model generation model not provided",
                table_name=self.input.table_name,
                semantic_model_file="",
            )
        try:
            # Use the provided table name directly
            if not self.input.table_name:
                return GenerateSemanticModelResult(
                    success=False,
                    error="Table name is required",
                    table_name=self.input.table_name,
                    semantic_model_file="",
                )

            # Parse table name parts using the new utility function
            table_parts = parse_table_name_parts(self.input.table_name, self.agent_config.db_type)

            # Get database manager and connector using context manager
            db_manager = db_manager_instance(self.agent_config.namespaces)
            current_namespace = self.agent_config.current_namespace

            # Use parsed parts or fallback to sql_task values
            catalog_name = table_parts["catalog_name"] or self.input.sql_task.catalog_name
            database_name = table_parts["database_name"] or self.input.sql_task.database_name
            schema_name = table_parts["schema_name"] or self.input.sql_task.schema_name
            table_name = table_parts["table_name"]
            db_type = self.agent_config.db_type

            logger.debug(
                f"table_name: {self.input.table_name}\n"
                f"Parsed table parts: {table_parts}\n"
                f"catalog_name: {catalog_name}\n"
                f"database_name: {database_name}\n"
                f"schema_name: {schema_name}\n"
                f"table_name: {table_name}\n"
                f"db_type: {db_type}\n"
                f"current_namespace: {current_namespace}\n"
            )

            # Generate semantic model ID
            semantic_model_id = gen_semantic_model_id(
                catalog_name,
                database_name,
                schema_name,
                table_name,
            )

            # Check if semantic model already exists in lancedb
            try:
                semantic_metrics_rag = rag_by_configuration(self.agent_config)
                existing_models = semantic_metrics_rag.semantic_model_storage.filter_by_id(semantic_model_id)

                if existing_models:
                    logger.info(f"Semantic model with ID {semantic_model_id} already exists, skipping generation")

                    # Return success without generating new model
                    return GenerateSemanticModelResult(
                        success=True,
                        error="",
                        table_name=table_name,
                        semantic_model_file=existing_models[0].get("semantic_file_path", ""),
                    )
            except Exception as e:
                logger.warning(f"Failed to check existing semantic model: {str(e)}, continuing with generation")

            logger.info(f"Generating semantic model for {table_name} in {catalog_name}.{database_name}.{schema_name}")
            with get_db_connector(db_manager, current_namespace, database_name) as connector:
                # Get tables with DDL
                tables_with_ddl = connector.get_tables_with_ddl(
                    tables=[table_name],
                    catalog_name=catalog_name,
                    database_name=database_name,
                    schema_name=schema_name,
                )

                if len(tables_with_ddl) == 0:
                    return GenerateSemanticModelResult(
                        success=False,
                        error="No tables with DDL found",
                        table_name=table_name,
                        semantic_model_file="",
                    )

                logger.debug(f"Tables with DDL: {tables_with_ddl}")

                # Generate semantic model
                tool = LLMTool(self.model)
                return tool.generate_semantic_model(
                    tables_with_ddl[0]["definition"],
                    self.input,
                    self.agent_config.current_db_config(database_name),
                )

        except Exception as e:
            logger.error(f"Semantic model generation error: {str(e)}")
            return GenerateSemanticModelResult(
                success=False,
                error=str(e),
                table_name=self.input.table_name,
                semantic_model_file="",
            )

    async def _generate_semantic_model_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate semantic model with streaming support and action history tracking."""
        if not self.model:
            logger.error("Model not available for semantic model generation")
            return

        try:
            # Use the provided table name directly
            if not self.input.table_name:
                logger.error("Table name is required")
                return

            # Parse table name parts
            table_parts = parse_table_name_parts(self.input.table_name, self.agent_config.db_type)

            # Database connection action
            db_action = ActionHistory(
                action_id="db_connect",
                role=ActionRole.WORKFLOW,
                messages="Connecting to database and retrieving table schema",
                action_type="schema_linking",
                input={
                    "table_name": self.input.table_name,
                    "database_type": self.agent_config.db_type,
                    "namespace": self.agent_config.current_namespace,
                },
                status=ActionStatus.SUCCESS,
            )
            yield db_action

            # Get database manager and connector
            db_manager = db_manager_instance(self.agent_config.namespaces)
            current_namespace = self.agent_config.current_namespace

            # Use parsed parts or fallback to sql_task values
            catalog_name = table_parts["catalog_name"] or self.input.sql_task.catalog_name
            database_name = table_parts["database_name"] or self.input.sql_task.database_name
            schema_name = table_parts["schema_name"] or self.input.sql_task.schema_name
            table_name = table_parts["table_name"]

            with get_db_connector(db_manager, current_namespace, database_name) as connector:
                tables_with_ddl = connector.get_tables_with_ddl(
                    catalog_name=catalog_name,
                    database_name=database_name,
                    tables=[table_name],
                    schema_name=schema_name,
                )

                if len(tables_with_ddl) == 0:
                    logger.error(f"No tables with DDL found for table: {table_name}, schema: {schema_name}")
                    return

                # Update the action with success
                db_action.output = {
                    "success": True,
                    "tables_found": len(tables_with_ddl),
                    "table_definition": tables_with_ddl[0]["definition"][:100] + "...",
                }
                db_action.status = ActionStatus.SUCCESS
                db_action.end_time = datetime.now()

                # Stream the semantic model generation
                async for action in generate_semantic_model_with_mcp_stream(
                    model=self.model,
                    table_definition=tables_with_ddl[0]["definition"],
                    input_data=self.input,
                    db_config=self.agent_config.current_db_config(database_name),
                    tool_config={},
                    action_history_manager=action_history_manager,
                ):
                    yield action

        except Exception as e:
            logger.error(f"Semantic model generation error: {str(e)}")
            raise

    def update_context(self, workflow: Workflow) -> Dict:
        return {}

    def setup_input(self, workflow: Workflow) -> Dict:
        try:
            # table_name will be populated from user input
            next_input = GenerateSemanticModelInput(sql_task=workflow.task, table_name="")
            self.input = next_input
            return {"success": True, "message": "Semantic model input setup", "suggestions": [next_input]}
        except Exception as e:
            logger.error(f"Error in setup_input: {e}")
            return {"success": False, "message": f"Setup input failed: {str(e)}", "suggestions": []}
