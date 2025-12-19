# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, Dict, List, Literal, Optional, Set, override

import duckdb
from pydantic import BaseModel, Field

from datus.schemas.base import TABLE_TYPE
from datus.schemas.node_models import ExecuteSQLResult
from datus.tools.db_tools.base import BaseSqlConnector, list_to_in_str
from datus.tools.db_tools.config import DuckDBConfig
from datus.tools.db_tools.mixins import SchemaNamespaceMixin
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class _DBMetadataNames(BaseModel):
    """
    The corresponding database commands are SHOW/SHOW CREAT/INFORMATION_SCHEMA.<TABLES>
    """

    info_table: str = Field(..., init=True, description="The name of metadata table")
    name_field: str = Field(..., init=True, description="Fields corresponding to names in metadata table")
    has_sql_field: bool = Field(True, init=True, description="Is there a SQL field.")


METADATA_DICT: Dict[str, _DBMetadataNames] = {
    "database": _DBMetadataNames(info_table="duckdb_databases", name_field="database_name", has_sql_field=False),
    "schema": _DBMetadataNames(info_table="duckdb_schemas", name_field="schema_name", has_sql_field=True),
    "table": _DBMetadataNames(info_table="duckdb_tables", name_field="table_name", has_sql_field=True),
    "view": _DBMetadataNames(info_table="duckdb_views", name_field="view_name", has_sql_field=True),
}


def _metadata_names(_type: str) -> _DBMetadataNames:
    if _type not in METADATA_DICT:
        raise DatusException(ErrorCode.COMMON_FIELD_INVALID, f"Invalid type `{_type}` for Database table type")
    return METADATA_DICT[_type]


class DuckdbConnector(BaseSqlConnector, SchemaNamespaceMixin):
    """
    Connector for DuckDB databases with schema namespace support using native DuckDB SDK.
    """

    def __init__(self, config: DuckDBConfig):
        super().__init__(config, dialect=DBType.DUCKDB)
        self.db_path = config.db_path.replace("duckdb:///", "")
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
        self.enable_external_access = config.enable_external_access
        self.memory_limit = config.memory_limit

        if config.database_name:
            self.database_name = config.database_name
        else:
            from datus.configuration.agent_config import file_stem_from_uri

            self.database_name = file_stem_from_uri(self.db_path)

    @override
    def connect(self):
        """Establish connection to DuckDB database."""
        if self.connection:
            return

        try:
            # Connect to DuckDB
            self.connection = duckdb.connect(self.db_path)

            # Configure settings
            if self.memory_limit:
                self.connection.execute(f"SET memory_limit='{self.memory_limit}'")

            if not self.enable_external_access:
                self.connection.execute("SET enable_external_access=false")

        except Exception as e:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message_args={"error_message": str(e)},
            ) from e

    @override
    def close(self):
        """Close the database connection."""
        if self.connection:
            try:
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing DuckDB connection: {e}")
            finally:
                self.connection = None

    @override
    def test_connection(self) -> bool:
        """Test the database connection."""
        opened_here = self.connection is None
        try:
            self.connect()
            self.connection.execute("SELECT 1").fetchone()
            return True
        except Exception as e:
            raise DatusException(
                ErrorCode.DB_CONNECTION_FAILED,
                message_args={"error_message": str(e)},
            ) from e
        finally:
            if opened_here:
                self.close()

    def _handle_exception(self, e: Exception, sql: str = "") -> DatusException:
        """Handle DuckDB exceptions and map to appropriate Datus ErrorCode."""
        if isinstance(e, DatusException):
            return e

        error_msg = str(e).lower()

        # Check for common error patterns
        if "syntax error" in error_msg or "parser error" in error_msg:
            return DatusException(
                ErrorCode.DB_EXECUTION_SYNTAX_ERROR,
                message_args={"sql": sql, "error_message": str(e)},
            )
        elif "table" in error_msg and "does not exist" in error_msg:
            return DatusException(
                ErrorCode.DB_TABLE_NOT_EXISTS,
                message_args={"table_name": sql, "error_message": str(e)},
            )
        elif "constraint" in error_msg or "unique" in error_msg:
            return DatusException(
                ErrorCode.DB_CONSTRAINT_VIOLATION,
                message_args={"sql": sql, "error_message": str(e)},
            )
        elif "timeout" in error_msg:
            return DatusException(
                ErrorCode.DB_EXECUTION_TIMEOUT,
                message_args={"sql": sql, "error_message": str(e)},
            )
        else:
            return DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={"sql": sql, "error_message": str(e)},
            )

    @override
    def execute_insert(self, sql: str) -> ExecuteSQLResult:
        """Execute an INSERT SQL statement."""
        try:
            self.connect()
            result = self.connection.execute(sql)
            # Check if result has a description (i.e., returns rows)
            if getattr(result, "description", None):
                fetched = result.fetchone()
                row_count = fetched[0] if fetched else 0
            else:
                # For DML without result set, use rowcount
                row_count = getattr(self.connection, "rowcount", 0) or 0
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(row_count),
                row_count=row_count,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_update(self, sql: str) -> ExecuteSQLResult:
        """Execute an UPDATE SQL statement."""
        try:
            self.connect()
            result = self.connection.execute(sql)
            # Check if result has a description (i.e., returns rows)
            if getattr(result, "description", None):
                fetched = result.fetchone()
                row_count = fetched[0] if fetched else 0
            else:
                # For DML without result set, use rowcount
                row_count = getattr(self.connection, "rowcount", 0) or 0
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(row_count),
                row_count=row_count,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_delete(self, sql: str) -> ExecuteSQLResult:
        """Execute a DELETE SQL statement."""
        try:
            self.connect()
            result = self.connection.execute(sql)
            # Check if result has a description (i.e., returns rows)
            if getattr(result, "description", None):
                fetched = result.fetchone()
                row_count = fetched[0] if fetched else 0
            else:
                # For DML without result set, use rowcount
                row_count = getattr(self.connection, "rowcount", 0) or 0
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(row_count),
                row_count=row_count,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_ddl(self, sql: str) -> ExecuteSQLResult:
        """Execute a DDL SQL statement."""
        try:
            self.connect()
            self.connection.execute(sql)
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return="Success",
                row_count=0,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_query(
        self, sql: str, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"
    ) -> ExecuteSQLResult:
        """Execute a SELECT query."""
        try:
            self.connect()
            result = self.connection.execute(sql)

            if result_format == "csv":
                df = result.df()
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql,
                    sql_return=df.to_csv(index=False),
                    row_count=len(df),
                    result_format=result_format,
                )
            elif result_format == "arrow":
                arrow_table = result.arrow()
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql,
                    sql_return=arrow_table,
                    row_count=len(arrow_table),
                    result_format=result_format,
                )
            elif result_format == "pandas":
                df = result.df()
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql,
                    sql_return=df,
                    row_count=len(df),
                    result_format=result_format,
                )
            else:  # list
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description]
                result_list = [dict(zip(columns, row)) for row in rows]
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql,
                    sql_return=result_list,
                    row_count=len(rows),
                    result_format=result_format,
                )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    @override
    def execute_pandas(self, sql: str) -> ExecuteSQLResult:
        """Execute query and return pandas DataFrame."""
        return self.execute_query(sql, result_format="pandas")

    @override
    def execute_csv(self, sql: str) -> ExecuteSQLResult:
        """Execute query and return CSV format."""
        return self.execute_query(sql, result_format="csv")

    @override
    def execute_queries(self, queries: List[str]) -> List[Any]:
        """Execute multiple queries."""
        results = []
        self.connect()
        try:
            for query in queries:
                result = self.connection.execute(query)
                if result.description:
                    rows = result.fetchall()
                    columns = [desc[0] for desc in result.description]
                    results.append([dict(zip(columns, row)) for row in rows])
                else:
                    results.append(0)
        except Exception as e:
            raise self._handle_exception(e, "\n".join(queries))
        return results

    @override
    def execute_content_set(self, sql_query: str) -> ExecuteSQLResult:
        """Execute SET/USE commands."""
        try:
            self.connect()
            self.connection.execute(sql_query)

            # Parse context switch
            from datus.utils.sql_utils import parse_context_switch

            switch_context = parse_context_switch(sql=sql_query, dialect=self.dialect)
            if switch_context:
                if database_name := switch_context.get("database_name"):
                    self.database_name = database_name
                if schema_name := switch_context.get("schema_name"):
                    self.schema_name = schema_name

            return ExecuteSQLResult(
                success=True,
                sql_query=sql_query,
                sql_return="Success",
                row_count=0,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql_query)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql_query,
            )

    @override
    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get all table names."""
        self.connect()
        sql = "SELECT table_name FROM duckdb_tables() WHERE database_name != 'system'"
        if database_name:
            sql += f" AND database_name = '{database_name}'"
        if schema_name:
            sql += f" AND schema_name = '{schema_name}'"

        result = self.connection.execute(sql)
        return [row[0] for row in result.fetchall()]

    @override
    def get_views(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get all view names."""
        self.connect()
        database_name = database_name or self.database_name
        schema_name = schema_name or self.schema_name
        sql = "SELECT view_name FROM duckdb_views() WHERE database_name != 'system'"
        if database_name:
            sql += f" AND database_name = '{database_name}'"
        if schema_name:
            sql += f" AND schema_name = '{schema_name}'"

        result = self.connection.execute(sql)
        return [row[0] for row in result.fetchall()]

    @override
    def get_databases(self, catalog_name: str = "", include_sys: bool = False) -> List[str]:
        """Get list of database names."""
        self.connect()
        sql = "SELECT database_name FROM duckdb_databases()"
        if not include_sys:
            sql += " WHERE database_name not in ('system', 'temp')"

        result = self.connection.execute(sql)
        return [row[0] for row in result.fetchall()]

    @override
    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "main", table_name: str = ""
    ) -> str:
        if database_name:
            if schema_name:
                return f'"{database_name}"."{schema_name}"."{table_name}"'
            return f'"{database_name}"."{table_name}"'
        return f'"{schema_name}"."{table_name}"' if schema_name else table_name

    @override
    def get_schemas(self, catalog_name: str = "", database_name: str = "", include_sys: bool = False) -> List[str]:
        self.connect()
        sql = "SELECT schema_name FROM duckdb_schemas()"
        has_where = False
        database_name = database_name or self.database_name
        if database_name:
            sql += f" WHERE database_name='{database_name}'"
            has_where = True

        if not include_sys:
            sys_schemas = list(self._sys_schemas())
            if not has_where:
                sql += list_to_in_str(" WHERE schema_name NOT IN", sys_schemas)
            else:
                sql += list_to_in_str(" AND schema_name NOT IN", sys_schemas)

        result = self.connection.execute(sql)
        return [row[0] for row in result.fetchall()]

    @override
    def _sys_schemas(self) -> Set[str]:
        return {"system", "temp", "information_schema"}

    @override
    def do_switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        self.connect()
        if schema_name:
            self.connection.execute(f'USE "{schema_name}"')

    @override
    def get_tables_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Get tables with DDL definitions."""
        filter_tables = self._reset_filter_tables(
            tables, catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )
        return self._get_meta_with_ddl(
            database_name=database_name,
            schema_name=schema_name,
            _type="table",
            filter_tables=filter_tables,
        )

    def _get_meta_with_ddl(
        self,
        database_name: str = "",
        schema_name: str = "",
        _type: str = "",
        filter_tables: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Get metadata with DDL for tables or views."""
        self.connect()
        metadata_names = _metadata_names(_type)
        sql_field = "" if not metadata_names.has_sql_field else ', "sql"'
        query_sql = (
            f"SELECT database_name, schema_name, {metadata_names.name_field}{sql_field}"
            f" FROM {metadata_names.info_table}() WHERE database_name != 'system'"
        )
        database_name = database_name or self.database_name
        schema_name = schema_name or self.schema_name
        if database_name:
            query_sql += f" AND database_name = '{database_name}'"
        if schema_name:
            query_sql += f" AND schema_name = '{schema_name}'"

        result_set = self.connection.execute(query_sql)
        rows = result_set.fetchall()
        columns = [desc[0] for desc in result_set.description]

        result = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            table_name = str(row_dict[metadata_names.name_field])
            full_name = self.full_name(
                database_name=str(row_dict["database_name"]),
                schema_name=str(row_dict["schema_name"]),
                table_name=table_name,
            )
            if not database_name:
                full_name = ".".join(full_name.split(".")[1:])
            if filter_tables and full_name not in filter_tables:
                continue

            result.append(
                {
                    "identifier": self.identifier(
                        database_name=str(row_dict["database_name"]),
                        schema_name=str(row_dict["schema_name"]),
                        table_name=table_name,
                    ),
                    "catalog_name": "",
                    "database_name": row_dict["database_name"],
                    "schema_name": row_dict["schema_name"],
                    "table_name": table_name,
                    "definition": row_dict.get("sql", ""),
                    "table_type": _type,
                }
            )
        return result

    @override
    def get_views_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[Dict[str, str]]:
        """Get views with DDL definitions."""
        return self._get_meta_with_ddl(
            database_name=database_name,
            schema_name=schema_name,
            _type="view",
        )

    @override
    def get_sample_rows(
        self,
        tables: Optional[List[str]] = None,
        top_n: int = 5,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_type: TABLE_TYPE = "table",
    ) -> List[Dict[str, str]]:
        """Get sample values from tables."""
        self.connect()
        try:
            samples = []
            if tables:
                logger.debug(f"Getting sample data from tables {tables} LIMIT {top_n}")
                for table_name in tables:
                    if schema_name:
                        if database_name:
                            prefix = f'"{database_name}"."{schema_name}"'
                        else:
                            prefix = f'"{schema_name}"'
                    else:
                        prefix = "" if not schema_name else f'"{schema_name}"'
                    if prefix:
                        query = f"""SELECT * FROM {prefix}."{table_name}" LIMIT {top_n}"""
                    else:
                        query = f"""SELECT * FROM "{table_name}" LIMIT {top_n}"""

                    result = self.connection.execute(query)
                    df = result.df()
                    if len(df) > 0:
                        samples.append(
                            {
                                "catalog_name": "",
                                "database_name": database_name,
                                "table_name": table_name,
                                "schema_name": schema_name,
                                "sample_rows": df.to_csv(index=False),
                            }
                        )
            else:
                tables_with_ddl = []
                if table_type == "mv":
                    return []
                if table_type in ("full", "table"):
                    tables_with_ddl.extend(
                        self.get_tables_with_ddl(database_name=database_name, schema_name=schema_name)
                    )
                if table_type in ("full", "view"):
                    tables_with_ddl.extend(
                        self.get_views_with_ddl(database_name=database_name, schema_name=schema_name)
                    )
                for table in tables_with_ddl:
                    query = (
                        f'SELECT * FROM "{table["database_name"]}"."{table["schema_name"]}"."{table["table_name"]}" '
                        f"LIMIT {top_n}"
                    )
                    result = self.connection.execute(query)
                    df = result.df()
                    if len(df) > 0:
                        samples.append(
                            {
                                "catalog_name": "",
                                "database_name": table["database_name"],
                                "table_name": table["table_name"],
                                "schema_name": table["schema_name"],
                                "sample_rows": df.to_csv(index=False),
                            }
                        )
            return samples
        except DatusException:
            raise
        except Exception as e:
            raise DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={
                    "error_message": str(e),
                },
            ) from e

    @override
    def get_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> List[Dict[str, str]]:
        """Get schema information for a table."""
        if not table_name:
            return []

        self.connect()
        database_name = database_name or self.database_name
        schema_name = schema_name or self.schema_name or "main"
        full_name = self.full_name(database_name=database_name, schema_name=schema_name, table_name=table_name)

        sql = f"PRAGMA table_info('{full_name}')"
        try:
            result = self.connection.execute(sql)
            rows = result.fetchall()
            columns = [desc[0] for desc in result.description]
            # Normalize field names to match standard schema
            schema_list = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                # Convert notnull to nullable and dflt_value to default_value
                normalized = {
                    "cid": row_dict.get("cid"),
                    "name": row_dict.get("name"),
                    "type": row_dict.get("type"),
                    "nullable": not bool(row_dict.get("notnull", 0)),  # Invert notnull to nullable
                    "default_value": row_dict.get("dflt_value"),  # Rename dflt_value to default_value
                    "pk": row_dict.get("pk"),
                }
                schema_list.append(normalized)
            return schema_list
        except DatusException as e:
            if "error_message" in e.message_args:
                message = e.message_args["error_message"]
            else:
                message = e.message
            raise DatusException(ErrorCode.DB_QUERY_METADATA_FAILED, message=message)
        except Exception as e:
            raise DatusException(
                ErrorCode.DB_QUERY_METADATA_FAILED,
                message_args={"error_message": str(e), "sql": sql},
            ) from e

    def to_dict(self) -> Dict[str, Any]:
        """Convert connector to serializable dictionary."""
        return {"db_type": DBType.DUCKDB, "db_path": self.db_path}

    def get_type(self) -> str:
        return DBType.DUCKDB
