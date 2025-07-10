from contextlib import contextmanager
from typing import Dict

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput, GenerateSemanticModelResult
from datus.storage.metric.init_utils import gen_semantic_model_id
from datus.storage.metric.store import rag_by_configuration
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.llms_tools import LLMTool
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_table_names, parse_table_name_parts

logger = get_logger(__name__)


@contextmanager
def get_db_connector(db_manager, current_namespace, db_type, database_name):
    connector = None
    try:
        connector = db_manager.get_conn(current_namespace, db_type, database_name)
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
                semantic_model_meta=self.input.semantic_model_meta,
                semantic_model_file="",
            )
        try:
            # Extract table names from SQL
            table_names = extract_table_names(self.input.sql_query, self.agent_config.db_type)
            if len(table_names) == 0:
                return GenerateSemanticModelResult(
                    success=False,
                    error="No tables found in SQL query",
                    semantic_model_meta=self.input.semantic_model_meta,
                    semantic_model_file="",
                )

            # Parse table name parts using the new utility function
            table_parts = parse_table_name_parts(table_names[0], self.agent_config.db_type)

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
                f"sql_query: {self.input.sql_query}\n"
                f"Table names: {table_names}\n"
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

                    # Update input metadata with existing model info
                    self.input.semantic_model_meta.table_name = table_name
                    self.input.semantic_model_meta.schema_name = schema_name
                    self.input.semantic_model_meta.catalog_name = catalog_name
                    self.input.semantic_model_meta.database_name = database_name

                    # Return success without generating new model
                    return GenerateSemanticModelResult(
                        success=True,
                        error="",
                        semantic_model_meta=self.input.semantic_model_meta,
                        semantic_model_file=existing_models[0].get("semantic_file_path", ""),
                    )
            except Exception as e:
                logger.warning(f"Failed to check existing semantic model: {str(e)}, continuing with generation")

            logger.info(f"Generating semantic model for {table_name} in {catalog_name}.{database_name}.{schema_name}")
            with get_db_connector(db_manager, current_namespace, db_type, database_name) as connector:
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
                        semantic_model_meta=self.input.semantic_model_meta,
                        semantic_model_file="",
                    )

                logger.debug(f"Tables with DDL: {tables_with_ddl}")
                self.input.semantic_model_meta.table_name = table_name
                self.input.semantic_model_meta.schema_name = schema_name
                self.input.semantic_model_meta.catalog_name = catalog_name
                self.input.semantic_model_meta.database_name = database_name

                # Generate semantic model
                tool = LLMTool(self.model)
                return tool.generate_semantic_model(
                    tables_with_ddl[0]["definition"],
                    self.input,
                    self.agent_config.current_db_config(),
                )

        except Exception as e:
            logger.error(f"Semantic model generation error: {str(e)}")
            return GenerateSemanticModelResult(
                success=False,
                error=str(e),
                semantic_model_meta=self.input.semantic_model_meta,
                semantic_model_file="",
            )

    def update_context(self, workflow: Workflow) -> Dict:
        return {}

    def setup_input(self, workflow: Workflow) -> Dict:
        return {}
