"""
Service for handling Database Management operations.
"""

import os
from datetime import datetime
from typing import List, Optional

from datus_db_core import BaseSqlConnector

from datus.api.models.base_models import Result
from datus.api.models.config_models import ErrorCode
from datus.api.models.database_models import (
    DatabaseInfo,
    ListDatabasesData,
    ListDatabasesInput,
)
from datus.api.models.table_models import (
    ColumnInfo,
    GetSemanticModelData,
    GetTableDetailData,
    SemanticModelInput,
    TableDetailData,
    ValidateSemanticModelData,
)
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config_loader import AgentConfig
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_table_name_parts

logger = get_logger(__name__)
# Database types that do NOT support schema switching
_NO_SCHEMA_TYPES = {"sqlite", "duckdb", "mysql"}


class DatabaseService:
    """Service for handling database management operations."""

    def __init__(self, agent_config: Optional[AgentConfig] = None):
        """
        Initialize the database service.

        Args:
            agent_config: Datus agent configuration
        """
        self.agent_config = agent_config

        self.db_manager = DBManager(agent_config.namespaces)
        self.current_namespace = agent_config.current_namespace
        self.semantic_rag = SemanticModelRAG(self.agent_config)

        self.current_db_connector = None
        self.current_database = None
        self._initialize_connection()

    def _get_database_type(self, database_name: Optional[str] = None) -> tuple[str, str]:
        """
        Get database type from agent configuration.

        Args:
            database_name: Optional database name. If not provided, uses current database.

        Returns:
            Database type string (e.g., 'starrocks', 'mysql', etc.)
            db_name: Database name
        """
        db_type = "unknown"
        target_db = database_name or self.current_database

        try:
            if self.agent_config and self.current_namespace in self.agent_config.namespaces:
                namespace_config = self.agent_config.namespaces[self.current_namespace]
                if target_db and target_db in namespace_config:
                    db_config = namespace_config[target_db]
                    db_type = db_config.type.value if hasattr(db_config.type, "value") else str(db_config.type)
                elif len(namespace_config) == 1:
                    # Single database in namespace
                    db_config = list(namespace_config.values())[0]
                    db_type = db_config.type.value if hasattr(db_config.type, "value") else str(db_config.type)
        except Exception as e:
            logger.warning(f"Failed to get db type from config: {e}")

        return db_type, target_db

    def _initialize_connection(self):
        """Initialize the current database connection."""
        if self.db_manager and self.current_namespace:
            try:
                db_name, connector = self.db_manager.first_conn_with_name(self.current_namespace)
                self.current_db_connector = connector
                self.current_database = connector.database_name or db_name
            except Exception as e:
                logger.warning(f"Failed to initialize database connection: {e}")
                self.current_db_connector = None
                self.current_database = None

    def _get_connection_info(
        self,
        connector,
        ds_id: str,
        request: ListDatabasesInput,
    ) -> List[DatabaseInfo]:
        """Get connection information for a database connector.

        Enumerates all databases (and schemas if supported), marks the
        connector's configured database as ``current``.  Request-level
        filters (database_name, schema_name, catalog_name) narrow the
        result set when provided.
        """
        from datus_db_core import connector_registry

        dialect = getattr(connector, "dialect", "unknown")
        has_schema = connector_registry.support_schema(dialect)
        catalog_name = request.catalog_name or getattr(connector, "catalog_name", None)
        now = datetime.now().isoformat() + "Z"

        def _disconnected(db_name: str) -> DatabaseInfo:
            return DatabaseInfo(
                name=db_name,
                uri=_get_uri(connector),
                type=dialect,
                current=(db_name == connector.database_name),
                catalog_name=catalog_name,
                schema_name=None,
                connection_status="disconnected",
                tables_count=None,
                last_accessed=now,
            )

        try:
            if not connector.test_connection():
                return [_disconnected(connector.database_name)]
        except Exception:
            logger.exception("Connection test failed for %s", connector.database_name)
            return [_disconnected(connector.database_name)]

        # 1) Enumerate databases — fatal if this fails since we have nothing to iterate.
        try:
            if request.database_name:
                db_names = [request.database_name]
            elif hasattr(connector, "get_databases"):
                db_names = connector.get_databases(
                    catalog_name=catalog_name,
                    include_sys=request.include_sys_schemas,
                )
            elif connector.database_name:
                db_names = [connector.database_name]
            else:
                db_names = []
        except Exception as e:
            logger.warning("Failed to enumerate databases for %s: %s", connector.database_name, e)
            return [_disconnected(connector.database_name)]

        db_infos: List[DatabaseInfo] = []
        for db_name in db_names:
            if has_schema:
                # 2) Resolve schemas for this db — a single failing db must not
                # abort the whole listing. Report the db as disconnected and
                # keep going.
                if request.schema_name:
                    schemas = [request.schema_name]
                elif hasattr(connector, "get_schemas"):
                    try:
                        schemas = connector.get_schemas(
                            catalog_name=request.catalog_name,
                            database_name=db_name,
                            include_sys=request.include_sys_schemas,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to get schemas for db=%s dialect=%s: %s",
                            db_name,
                            dialect,
                            e,
                        )
                        db_infos.append(_disconnected(db_name))
                        continue
                else:
                    schemas = ["public"]

                for schema in schemas:
                    # 3) Fetch tables for this (db, schema). A failure here only
                    # invalidates this entry, not sibling schemas.
                    try:
                        tables = connector.get_tables(
                            catalog_name=catalog_name, database_name=db_name, schema_name=schema
                        )
                        tables.sort()
                    except Exception as e:
                        logger.warning(
                            "Failed to get tables for db=%s schema=%s: %s",
                            db_name,
                            schema,
                            e,
                        )
                        db_infos.append(
                            DatabaseInfo(
                                name=db_name,
                                uri=_get_uri(connector),
                                type=dialect,
                                current=(db_name == connector.database_name),
                                catalog_name=catalog_name,
                                schema_name=schema,
                                connection_status="disconnected",
                                tables_count=None,
                                last_accessed=now,
                            )
                        )
                        continue

                    db_infos.append(
                        DatabaseInfo(
                            name=db_name,
                            uri=_get_uri(connector),
                            type=dialect,
                            current=(db_name == connector.database_name),
                            catalog_name=catalog_name,
                            schema_name=schema,
                            connection_status="connected",
                            tables_count=len(tables),
                            last_accessed=now,
                            tables=tables,
                        )
                    )
            else:
                # No schema support — get tables directly. Isolate per-db failures.
                try:
                    tables = connector.get_tables(
                        catalog_name=catalog_name, database_name=db_name, schema_name=request.schema_name
                    )
                    tables.sort()
                except Exception as e:
                    logger.warning("Failed to get tables for db=%s: %s", db_name, e)
                    db_infos.append(_disconnected(db_name))
                    continue

                db_infos.append(
                    DatabaseInfo(
                        name=db_name,
                        uri=_get_uri(connector),
                        type=dialect,
                        current=(db_name == connector.database_name),
                        catalog_name=catalog_name,
                        schema_name=None,
                        connection_status="connected",
                        tables_count=len(tables),
                        last_accessed=now,
                        tables=tables,
                    )
                )
        return db_infos

    def list_databases(self, request: ListDatabasesInput) -> Result[ListDatabasesData]:
        """
        List available databases.

        Args:
            request: List databases request

        Returns:
            ListDatabasesResult with databases list
        """
        # FIXME try use project_id
        try:
            if not self.db_manager:
                return Result(
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Database manager not initialized",
                )

            # Get connections from the specified namespace
            namespace = request.datasource_id or self.current_namespace
            connections = self.db_manager.get_connections(namespace)

            databases = []
            # Handle both single connector and dictionary of connectors
            if isinstance(connections, dict):
                for _ds_id, connector in connections.items():
                    db_info = self._get_connection_info(connector, _ds_id, request)
                    databases.extend(db_info)
            else:
                # Single connector case
                db_info = self._get_connection_info(connections, namespace, request)
                databases.extend(db_info)

            data = ListDatabasesData(
                databases=databases,
                total_count=len(databases),
                current_database=self.current_database,
            )

            return Result(success=True, data=data)

        except Exception as e:
            logger.error(f"Failed to list databases: {e}")
            return Result(
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    def get_table_schema(self, full_path: str) -> Result[GetTableDetailData]:
        """
        Get table schema details.

        Args:
            full_path: table name, [catalog.][database.][schema.]table

        Returns:
            GetTableSchemaResult with table schema
        """
        try:
            if not self.current_db_connector:
                return Result(
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="No database connection",
                )

            # Get table schema
            name_parts = parse_table_name_parts(full_path, self.current_db_connector.get_type())

            try:
                # For StarRocks: catalog.database.table (no schema level)
                # Use current database if not specified
                catalog_name = name_parts["catalog_name"] or getattr(self.current_db_connector, "catalog_name", "")
                database_name = (
                    name_parts["database_name"]
                    or self.current_database
                    or getattr(self.current_db_connector, "database", "")
                )
                schema_name = name_parts["schema_name"] or getattr(self.current_db_connector, "schema_name", "")
                table_name = name_parts["table_name"]

                schema_info = self.current_db_connector.get_schema(
                    catalog_name=catalog_name,
                    database_name=database_name,
                    schema_name=schema_name,
                    table_name=table_name,
                )
                if not schema_info:
                    return Result(
                        success=False,
                        errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                        errorMessage=f"Table '{table_name}' not found or schema not available",
                    )

                # Convert schema info to ColumnInfo objects
                columns = []
                if isinstance(schema_info, list):
                    for _i, col in enumerate(schema_info):
                        if isinstance(col, dict):
                            column_info = ColumnInfo(
                                name=col.get("name", ""),
                                type=col.get("type", ""),
                                nullable=col.get("notnull", 1) == 0,  # SQLite style: notnull=0 means nullable
                                default_value=col.get("dflt_value"),
                                pk=col.get("pk", 0) == 1,
                            )
                        else:
                            # Handle string or other formats
                            column_info = ColumnInfo(
                                name=str(col),
                                type="TEXT",
                                nullable=True,
                                default_value=None,
                                pk=False,
                            )
                        columns.append(column_info)
                else:
                    # Handle other schema formats
                    columns = []

                data = GetTableDetailData(table=TableDetailData(name=table_name, columns=columns, indexes=[]))

                return Result(success=True, data=data)

            except Exception as e:
                return Result(
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Failed to get table schema: {str(e)}",
                )

        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            return Result(
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    def _get_semantic_model(self, full_name: str):
        # Parse table name parts
        name_parts = parse_table_name_parts(full_name, self.current_db_connector.get_type())
        current_db_config = self.agent_config.current_db_config()
        catalog_name = name_parts["catalog_name"] or current_db_config.catalog
        database_name = name_parts["database_name"] or self.current_database or current_db_config.database
        schema_name = name_parts["schema_name"] or current_db_config.schema
        table_name = name_parts["table_name"]

        # Get semantic model using SemanticMetricsRAG
        semantic_model = self.semantic_rag.get_semantic_model(
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
        )
        return semantic_model

    def get_semantic_model(self, full_name: str) -> Result[GetSemanticModelData]:
        """Get SemanticModel YAML.

        Business logic:
        1. Parse table name to get catalog, database, schema, table components
        2. Use SemanticMetricsRAG.get_semantic_model() to retrieve semantic model by table_name
        3. Get semantic_file_path from the result
        4. Return the raw YAML file content

        Args:
            full_name: Full table name: [catalog.][database.][schema.]table

        Returns:
            Result[GetSemanticModelData] with YAML content
        """
        try:
            semantic_model = self._get_semantic_model(full_name)
            if not semantic_model:
                return Result[GetSemanticModelData](
                    success=True,
                )

            # Get semantic file path from result
            semantic_file_path = semantic_model.get("yaml_path", "")

            if not semantic_file_path or not os.path.exists(semantic_file_path):
                return Result[GetSemanticModelData](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Semantic file not found: {semantic_file_path}",
                )

            # Read and return the raw YAML file content
            with open(semantic_file_path, "r", encoding="utf-8") as f:
                yaml_content = f.read()

            return Result[GetSemanticModelData](success=True, data=GetSemanticModelData(yaml=yaml_content))

        except Exception as e:
            logger.error(f"Failed to get semantic model: {e}")
            return Result[GetSemanticModelData](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def save_semantic_model(self, request: SemanticModelInput) -> Result[dict]:
        """Save SemanticModel YAML.

        Args:
            request: Save semantic model input with table name and YAML

        Returns:
            Result[dict]
        """
        # Step 1: Validate the YAML first
        validation_result = await self.validate_semantic_model(request)

        if not validation_result.success:
            return Result[dict](
                success=False,
                errorCode=validation_result.errorCode,
                errorMessage=validation_result.errorMessage,
            )

        # Check if validation passed
        if validation_result.data and not validation_result.data.valid:
            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage="; ".join(validation_result.data.invalid_message or []),
            )

        # Step 2: Get semantic file path
        semantic_model = self._get_semantic_model(request.table)
        if not semantic_model:
            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=f"Semantic model not found for table: {request.table}",
            )

        semantic_file_path = semantic_model.get("yaml_path", "")

        # Step 3: Write YAML to file
        try:
            with open(semantic_file_path, "w", encoding="utf-8") as f:
                f.write(request.yaml)
        except Exception as e:
            return Result[dict](
                success=False,
                errorCode=ErrorCode.INTERNAL_COMMAND_ERROR,
                errorMessage=f"Failed to write semantic model file: {e}",
            )

        # Step 4: Sync semantic model to database
        try:
            sync_result = GenerationHooks._sync_semantic_to_db(
                semantic_file_path,
                self.agent_config,
                include_semantic_objects=True,
                include_metrics=False,
            )
            if not sync_result.get("success", False):
                error_msg = sync_result.get("error", "Unknown error")
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INTERNAL_COMMAND_ERROR,
                    errorMessage=f"Failed to sync semantic model to database: {error_msg}",
                )
        except Exception as e:
            return Result[dict](
                success=False,
                errorCode=ErrorCode.INTERNAL_COMMAND_ERROR,
                errorMessage=f"Failed to sync semantic model to database: {e}",
            )

        return Result[dict](success=True, data={})

    async def validate_semantic_model(self, request: SemanticModelInput) -> Result[ValidateSemanticModelData]:
        """Validate SemanticModel YAML with full semantic model validation.

        This method performs complete validation by:
        1. Creating a temporary file with the input YAML
        2. Using ConfigLinter to check YAML format/structure
        3. Combining with existing semantic models in the namespace directory
        4. Using parse_yaml_file_paths_to_model for full semantic validation
           (including cross-file reference checks)
        5. Using ModelValidator for semantic validation
        6. Cleaning up the temporary file after validation

        Args:
            request: Validate semantic model input with YAML content

        Returns:
            Result[ValidateSemanticModelData] with validation status
        """
        logger.info("Validating semantic model YAML")
        try:
            full_name = request.table
            semantic_model = self._get_semantic_model(full_name)
            if not semantic_model:
                return Result[ValidateSemanticModelData](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Semantic model not found for table: {full_name}",
                )

            # Get semantic file path from result
            semantic_file_path = semantic_model.get("yaml_path", "")

            # Validate using shared utility (deep validation when metricflow is available)
            from datus.api.utils.semantic_validation import validate_semantic_yaml

            is_valid, error_messages = validate_semantic_yaml(
                yaml_content=request.yaml,
                file_path=semantic_file_path,
                datus_home=self.agent_config.home,
                namespace=self.agent_config.current_namespace,
            )

            if not is_valid:
                return Result[ValidateSemanticModelData](
                    success=True,
                    data=ValidateSemanticModelData(valid=False, invalid_message=error_messages),
                )

            return Result[ValidateSemanticModelData](
                success=True,
                data=ValidateSemanticModelData(valid=True, invalid_message=None),
            )
        except Exception as e:
            logger.error(f"Failed to validate semantic model: {e}")
            return Result[ValidateSemanticModelData](
                success=False,
                errorCode=ErrorCode.INTERNAL_COMMAND_ERROR,
                errorMessage=str(e),
            )


def _get_uri(connector: BaseSqlConnector) -> str:
    if not connector:
        return ""
    connection_string = getattr(connector, "connection_string", "")
    if connection_string:
        return connection_string
    return f"{connector.dialect}://"
