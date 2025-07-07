from contextlib import contextmanager
from typing import Dict

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput, GenerateSemanticModelResult
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.llms_tools import LLMTool
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_table_names

logger = get_logger(__file__)


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

            # Get database manager and connector using context manager
            db_manager = db_manager_instance(self.agent_config.namespaces)
            current_namespace = self.agent_config.current_namespace
            catalog_name = self.input.sql_task.catalog_name
            database_name = self.input.sql_task.database_name
            schema_name = self.input.sql_task.schema_name
            db_type = self.agent_config.db_type

            with get_db_connector(db_manager, current_namespace, db_type, database_name) as connector:
                # Get tables with DDL
                table_name = table_names[0].split(".")[-1]

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
        pass

    def setup_input(self, workflow: Workflow) -> Dict:
        pass
