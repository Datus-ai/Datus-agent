# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, Optional

from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.schemas.node_models import TableSchema, TableValue
from datus.schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.base import BaseTool
from datus.tools.db_tools.base import BaseSqlConnector
from datus.tools.llms_tools.match_schema import MatchSchemaTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SchemaLineageTool(BaseTool):
    """Tool for managing and querying schema lineage information."""

    def __init__(
        self,
        storage: Optional[SchemaWithValueRAG] = None,
        agent_config: Optional[AgentConfig] = None,
        **kwargs,
    ):
        """Initialize the schema lineage tool.

        Args:
            db_path: Path to the LanceDB database directory
        """
        super().__init__(**kwargs)
        if storage:
            self.store = storage
        else:
            self.store = SchemaWithValueRAG(agent_config)

    def validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate the input data for schema lineage operations.
        Args:
            input_data: containing:
                - input_text: The text to search for similar schemas
                - top_n: (Optional) Number of similar schemas to return (default: 5)
                - database_name: (Optional) Database name for storing new schema
        """

    def execute(self, input_param: SchemaLinkingInput, model: Optional[LLMBaseModel] = None) -> SchemaLinkingResult:
        """Execute schema lineage operations.

        Args:
            input_params: containing:
                - input_text: The text to search for similar schemas
                - top_n: (Optional) Number of similar schemas to return (default: 5)
                - database_name: (Optional) Database name for storing new schema

        Returns:
            SchemaLinkingResult: Validated operation results with table schemas and values
        """
        if self.store is None:
            return SchemaLinkingResult(
                success=False,
                error="Schema linking tool not found",
                schema_count=0,
                value_count=0,
                table_schemas=[],
                table_values=[],
            )

        # Leave exceptions to the higher-ups to handle.
        if input_param.matching_rate == "from_llm":
            tool = MatchSchemaTool(model, storage=self.store.schema_store)
            if not tool:
                return SchemaLinkingResult(
                    success=False,
                    error="Schema metadata has not yet been built",
                    schema_count=0,
                    value_count=0,
                    table_schemas=[],
                    table_values=[],
                )
            return tool.execute(input_param)
        return self._search_similar_schemas(input_param, input_param.top_n_by_rate())

    def _search_similar_schemas(self, input_param: SchemaLinkingInput, top_n: int = 5) -> SchemaLinkingResult:
        (similar_schemas, similar_values) = self.store.search_similar(
            input_param.input_text,
            top_n=top_n,
            catalog_name=input_param.catalog_name,
            database_name=input_param.database_name,
            schema_name=input_param.schema_name,
            table_type=input_param.table_type,
        )

        # Convert dictionaries to proper model instances
        table_schemas = TableSchema.from_arrow(similar_schemas)
        table_values = TableValue.from_arrow(similar_values)

        return SchemaLinkingResult(
            success=True,
            error=None,
            table_schemas=table_schemas,
            schema_count=len(table_schemas),
            table_values=table_values,
            value_count=len(table_values),
        )
        # Return validated result using Pydantic model

    def search_similar_schemas_by_schema(self, input_param: SchemaLinkingInput, top_n=20) -> SchemaLinkingResult:
        """Get most similar under all schemas

        Args:
            input_param (SchemaLinkingInput): The input parameters
            top_n (int): The number of most similar schemas to return

        Returns:
            SchemaLinkingResult: The result of the search
        """
        return self.store.schema_store.search_top_tables_by_every_schema(
            input_param.input_text,
            database_name=input_param.database_name,
            catalog_name=input_param.catalog_name,
            top_n=top_n,
        )

    def get_schems_by_db(self, connector: BaseSqlConnector, input_param: SchemaLinkingInput) -> SchemaLinkingResult:
        from datus.schemas.node_models import TableSchema

        tables_with_ddl = connector.get_tables_with_ddl(
            catalog_name=input_param.catalog_name,
            database_name=input_param.database_name,
            schema_name=input_param.schema_name,
        )
        top_n = input_param.top_n_by_rate()
        # Limit to top_n tables
        tables_with_ddl = tables_with_ddl[:top_n] if tables_with_ddl else []

        # Convert to TableSchema objects
        table_schemas = []
        for table_info in tables_with_ddl:
            table_schema = TableSchema(
                identifier=table_info.get("identifier", ""),
                catalog_name=table_info.get("catalog_name", ""),
                table_name=table_info.get("table_name", ""),
                database_name=table_info.get("database_name", input_param.database_name),
                schema_name=table_info.get("schema_name", ""),
                definition=table_info.get("definition", ""),
                table_type=table_info.get("table_type", "table"),
            )
            table_schemas.append(table_schema)

        # Return successful result
        return SchemaLinkingResult(
            success=True,
            error="",
            schema_count=len(table_schemas),
            value_count=0,  # len(table_values),
            table_schemas=table_schemas,
            table_values=[],  # table_values,
        )
