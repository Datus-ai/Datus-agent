# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Literal, Optional, Set, Tuple

from datus.schemas.base import TABLE_TYPE
from datus.schemas.node_models import ExecuteSQLInput, ExecuteSQLResult
from datus.tools.db_tools.config import ConnectionConfig
from datus.utils.constants import SQLType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import metadata_identifier, parse_sql_type

logger = get_logger(__name__)


class BaseSqlConnector(ABC):
    def __init__(self, config: ConnectionConfig, dialect: str):
        self.config = config
        self.timeout_seconds = config.timeout_seconds
        self.connection: Any = None
        self.dialect = dialect
        self.catalog_name = ""
        self.database_name = ""
        self.schema_name = ""

    def close(self):
        """
        Close the database connection.
        """
        if self.connection:
            self.connection.close()
            self.connection = None

    def connect(self):
        return

    # ==================== Context Manager Support ====================

    def __enter__(self):
        """Enter context: establish database connection.

        Returns:
            Self for use in with statement
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: cleanup resources.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred

        Returns:
            False to not suppress exceptions
        """
        if exc_type:
            # Try to rollback on exception
            try:
                self._safe_rollback()
            except Exception as e:
                logger.warning(f"Failed to rollback during cleanup: {e}")

        # Always close connection
        try:
            self.close()
        except Exception as e:
            logger.warning(f"Failed to close connection during cleanup: {e}")

        return False  # Don't suppress exceptions

    def _safe_rollback(self):
        """Safe rollback transaction (can be overridden by subclasses).

        Default implementation attempts to rollback if connection has rollback method.
        Subclasses can override this for database-specific rollback behavior.
        """
        if hasattr(self, "connection") and self.connection:
            try:
                if hasattr(self.connection, "rollback"):
                    self.connection.rollback()
            except Exception:
                pass

    def execute(
        self, input_params: Any, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"
    ) -> ExecuteSQLResult:
        """Execute a SQL query against the database.

        Args:
            input_params: Dictionary containing the input parameters including:
                - sql_query: The SQL query to execute
            result_format: The format of the result to return

        Returns:
            A dictionary containing the query results
        """
        self.validate_input(input_params)
        if isinstance(input_params, dict):
            input_params = ExecuteSQLInput(**input_params)
        sql_query = input_params.sql_query.strip()
        try:
            sql_type = parse_sql_type(sql_query, self.dialect)
            if sql_type == SQLType.INSERT:
                result = self.execute_insert(sql_query)
            elif sql_type in (SQLType.UPDATE, SQLType.DELETE, SQLType.MERGE):
                result = self.execute_update(sql_query)
            elif sql_type == SQLType.CONTENT_SET:
                result = self.execute_content_set(sql_query)
            elif sql_type == SQLType.DDL:
                result = self.execute_ddl(sql_query)
            elif sql_type == SQLType.SELECT:
                result = self.execute_query(sql_query, result_format)
            elif sql_type == SQLType.METADATA_SHOW:
                result = self.execute_query(sql_query, result_format)
            elif sql_type == SQLType.EXPLAIN:
                result = self.execute_explain(sql_query, result_format)
            else:
                return ExecuteSQLResult(
                    success=False,
                    error="Unknown type of SQL",
                    sql_query=sql_query,
                    sql_return="",
                    row_count=0,
                    result_format=result_format,
                )

            result.success = True
            return result
        except Exception as e:
            logger.error(f"Executing SQL query failed: {e}")
            return ExecuteSQLResult(
                success=True,
                error=str(e),
                sql_query=sql_query,
                sql_return="",
                row_count=0,
                result_format=result_format,
            )

    @abstractmethod
    def execute_insert(self, sql: str) -> ExecuteSQLResult:
        """Execute an INSERT SQL statement.

        Args:
            sql: The INSERT SQL statement to execute

        Returns:
            A dictionary containing the insert operation results
        """
        raise NotImplementedError()

    @abstractmethod
    def execute_update(self, sql: str) -> ExecuteSQLResult:
        """Execute an UPDATE SQL statement.

        Args:
            sql: The UPDATE SQL statement to execute

        Returns:
            A dictionary containing the update operation results
        """
        raise NotImplementedError()

    @abstractmethod
    def execute_delete(self, sql: str) -> ExecuteSQLResult:
        """Execute a DELETE SQL statement.

        Args:
            sql: The DELETE SQL statement to execute

        Returns:
            A dictionary containing the delete operation results
        """
        raise NotImplementedError()

    def validate_input(self, input_params: Any):
        """Validate the input parameters before execution.

        Args:
            input_params: Dictionary containing the input parameters to validate

        Raises:
            ValueError: If the input parameters are invalid
        """
        if not hasattr(input_params, "sql_query") and "sql_query" not in input_params:
            raise ValueError("'sql_query' parameter is required")
        if not isinstance(input_params["sql_query"], str):
            raise ValueError("'sql_query' must be a string")

    def execute_arrow(self, sql: str) -> ExecuteSQLResult:
        raise NotImplementedError()

    @abstractmethod
    def execute_query(
        self, sql: str, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"
    ) -> ExecuteSQLResult:
        """
        The best performing query in the current connector
        """
        raise NotImplementedError()

    def execute_explain(
        self, sql: str, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"
    ) -> ExecuteSQLResult:
        """Run EXPLAIN/EXPLAIN ANALYZE statements. Default implementation reuses execute_query."""
        return self.execute_query(sql, result_format)

    @abstractmethod
    def execute_pandas(self, sql: str) -> ExecuteSQLResult:
        raise NotImplementedError()

    @abstractmethod
    def execute_ddl(self, sql: str) -> ExecuteSQLResult:
        raise NotImplementedError()

    @abstractmethod
    def execute_csv(self, sql: str) -> ExecuteSQLResult:
        raise NotImplementedError()

    # ==================== Core Methods (Required) ====================
    # Methods that all database connectors must implement

    @abstractmethod
    def get_databases(self, catalog_name: str = "", include_sys: bool = False) -> List[str]:
        """Get list of database names.

        Args:
            catalog_name: Optional catalog name (ignored if catalog not supported)
            include_sys: Whether to include system databases

        Returns:
            List of database names
        """
        raise NotImplementedError()

    @abstractmethod
    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get all table names from the database.

        Args:
            catalog_name: Optional catalog name
            database_name: Optional database name
            schema_name: Optional schema name

        Returns:
            List of table names
        """
        raise NotImplementedError()

    def get_views(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get all view names from the database (optional, returns empty list by default).

        Args:
            catalog_name: Optional catalog name
            database_name: Optional database name
            schema_name: Optional schema name

        Returns:
            List of view names
        """
        return []

    # Note: The following methods have been moved to Mixin classes:
    # - get_catalogs() -> CatalogSupportMixin
    # - get_materialized_views() -> MaterializedViewSupportMixin
    # - get_schemas() -> SchemaNamespaceMixin
    # Use isinstance() to check if a connector supports these features

    def _sys_databases(self) -> Set[str]:
        return set()

    def _sys_schemas(self) -> Set[str]:
        return set()

    def execute_csv_iterator(
        self, query: str, max_rows: int = 100, with_header: bool = True
    ) -> Iterator[Tuple[str, ...]]:
        # if with_header is True, the first batch is the column names
        raise NotImplementedError()

    # for internal streaming inter IPC or networking
    # def execute_to_arrow_stream(self, query: str, output_stream: Any,
    #                           compression: Optional[str] = 'lz4') -> None:
    #    pass
    @abstractmethod
    def test_connection(self):
        raise NotImplementedError()

    def get_type(self) -> str:
        return self.dialect

    @abstractmethod
    def execute_queries(self, queries: List[str]) -> List[Any]:
        raise NotImplementedError()

    def get_tables_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Get all tables with DDL from the database.
        Namespace parameters (such as catalog_name, database_name, schema_name) should be passed via kwargs and
        handled by subclasses as needed.
        parameters:
            tables: The table names to filter the tables.
            catalog_name: The catalog name to filter the tables.
            database_name: The database name to filter the tables.
            schema_name: The schema name to filter the tables.
        """
        raise NotImplementedError()

    def _reset_filter_tables(
        self, tables: Optional[List[str]] = None, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[str]:
        filter_tables = []
        if tables:
            catalog_name = catalog_name or self.catalog_name
            database_name = database_name or self.database_name
            schema_name = schema_name or self.schema_name
            for table_name in tables:
                filter_tables.append(
                    self.full_name(
                        table_name=table_name,
                        catalog_name=catalog_name,
                        database_name=database_name,
                        schema_name=schema_name,
                    )
                )
        return filter_tables

    def get_views_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[Dict[str, str]]:
        """
        Get all views with DDL from the database.
        Namespace parameters (such as catalog_name, database_name, schema_name)
        should be passed via kwargs and handled by subclasses as needed.
        parameters:
            catalog_name: The catalog name to filter the views.
            database_name: The database name to filter the views.
            schema_name: The schema name to filter the views.
        """
        raise NotImplementedError()

    def switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        """
        Switch context, including catalogs, databases and schemas.
        Parameters contains catalog_name, database_name and schema_name
        """
        self.connect()
        self.do_switch_context(catalog_name=catalog_name, database_name=database_name, schema_name=schema_name)
        if catalog_name:
            self.catalog_name = catalog_name
        if database_name:
            self.database_name = database_name
        if schema_name:
            self.schema_name = schema_name

    def do_switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        return None

    def get_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> List[Dict[str, str]]:
        """
        Get schema information for the specified table.
        Namespace parameters (such as catalog_name, database_name, schema_name, table_name)
        should be passed via kwargs and handled by subclasses as needed.
        """
        raise NotImplementedError()

    def get_sample_rows(
        self,
        tables: Optional[List[str]] = None,
        top_n: int = 5,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "table",
    ) -> List[Dict[str, Any]]:
        """
        Get sample values for each table from the database.
        Namespace parameters (such as catalog_name, database_name, schema_name) should be passed via kwargs and
        handled by subclasses as needed.
        Args:
            tables: List of table names to sample from. If None, sample from all tables.
            top_n: Number of sample rows to retrieve per table.
            catalog_name: The Catalog of database
            database_name: The Database of table
            schema_name:
            table_type: table/view/mv(Abbreviated Materialized View)
        Returns:
            A list of dictionaries containing sample values for each table.
        """
        raise NotImplementedError()

    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        """
        Get the full name of the table. Special characters need to be taken into consideration during implementation.
        """
        raise NotImplementedError()

    def identifier(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        return metadata_identifier(
            dialect=self.dialect,
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
        )

    @abstractmethod
    def execute_content_set(self, sql_query: str) -> ExecuteSQLResult:
        """
        Execute commands such as use/set to switch contexts
        """
        raise NotImplementedError()


def list_to_in_str(prefix: str, values: Optional[List[str]] = None) -> str:
    if not values:
        return ""
    value_str = ",".join(_to_sql_literal(v, around_with_quotes=True) for v in values)
    return f"{prefix} ({value_str})"


def _escape_sql_string_standard(value: str) -> str:
    # Standard SQL single quote escaping rules: single quote -> two single quotes
    return value.replace("'", "''")


def _to_sql_literal(value: Optional[str], around_with_quotes: bool = False) -> str:
    """Return a Snowflake-safe single-quoted SQL literal for strings."""
    if value is None:
        return "NULL"
    if not value:
        return "" if not around_with_quotes else "''"
    replace_value = _escape_sql_string_standard(value)
    if not around_with_quotes:
        return replace_value
    else:
        return f"'{replace_value}'"
