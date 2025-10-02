import json
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, override

import pyarrow as pa
from pandas import DataFrame
from snowflake.connector import Connect, SnowflakeConnection
from snowflake.connector.errors import (
    DatabaseError,
    DataError,
    ForbiddenError,
    IntegrityError,
    InterfaceError,
    InternalError,
    NotSupportedError,
    OperationalError,
    ProgrammingError,
    RequestTimeoutError,
    ServiceUnavailableError,
)

from datus.schemas.node_models import ExecuteSQLResult
from datus.tools.db_tools.base import BaseSqlConnector
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _handle_snowflake_exception(e: Exception, sql: str = "") -> DatusException:
    """Handle Snowflake exceptions and map to appropriate Datus ErrorCode."""

    if isinstance(e, ProgrammingError):
        # SQL syntax errors, invalid queries, etc.
        # detailed_msg = f"errno:{e.errno}, sqlstate: {e.sqlstate}, message: {e.msg}, query_id: {e.sfqid}"
        return DatusException(
            ErrorCode.DB_EXECUTION_SYNTAX_ERROR, message_args={"sql": sql, "error_message": e.raw_msg}
        )

    elif isinstance(e, (OperationalError, DatabaseError)):
        # Database operational issues
        return DatusException(ErrorCode.DB_EXECUTION_ERROR, message_args={"sql": sql, "error_message": e.raw_msg})

    elif isinstance(e, IntegrityError):
        # Constraint violations, unique key violations, etc.
        return DatusException(ErrorCode.DB_CONSTRAINT_VIOLATION, message_args={"sql": sql, "error_message": e.raw_msg})

    elif isinstance(e, (RequestTimeoutError, ServiceUnavailableError)):
        # Timeout and service availability issues
        return DatusException(ErrorCode.DB_EXECUTION_TIMEOUT, message_args={"sql": sql, "error_message": e.raw_msg})

    elif isinstance(e, (InterfaceError, InternalError)):
        # Connection and internal errors
        return DatusException(ErrorCode.DB_CONNECTION_FAILED, message_args={"error_message": e.raw_msg})

    elif isinstance(e, ForbiddenError):
        # Permission denied errors
        return DatusException(
            ErrorCode.DB_PERMISSION_DENIED, message_args={"operation": "query execution", "error_message": e.raw_msg}
        )

    elif isinstance(e, (DataError, NotSupportedError)):
        # Data type errors and unsupported operations
        return DatusException(ErrorCode.DB_EXECUTION_ERROR, message_args={"sql": sql, "error_message": e.raw_msg})

    else:
        # Generic database failure for unknown exceptions
        return DatusException(ErrorCode.DB_FAILED, message_args={"error_message": str(e)})


class SnowflakeConnector(BaseSqlConnector):
    """
    Connector for Snowflake databases.
    """

    def __init__(
        self,
        account: str,
        user: str,
        password: str,
        warehouse: str,
        database: str = "",
        schema: str = "",
        timeout_seconds: int = 30,
    ):
        super().__init__(dialect=DBType.SNOWFLAKE, timeout_seconds=timeout_seconds)
        # FIXME lazy init
        self.connection: SnowflakeConnection = Connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database=database if database else None,
            schema=schema if schema else None,
            login_timeout=timeout_seconds,
            network_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )
        self.database_name = database
        self.schema_name = schema

    def test_connection(self) -> Dict[str, Any]:
        """"""
        with self.connection.cursor() as cursor:
            cursor.execute("select 1")
            cursor.fetchall()
            return {
                "success": True,
                "message": "Connection successful",
                "databases": "",
            }

    def close(self):
        self.connection.close()

    def do_switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        try:
            with self.connection.cursor() as cursor:
                if not schema_name:
                    if not database_name:
                        return
                    sql = f'USE DATABASE "{database_name}"'
                else:
                    sql = (
                        f'USE SCHEMA "{schema_name}"' if not database_name else f'USE "{database_name}"."{schema_name}"'
                    )
                cursor.execute(sql)
        except Exception as e:
            raise _handle_snowflake_exception(e, sql) from e

    @override
    def execute_ddl(self, sql: str) -> ExecuteSQLResult:
        return self._execute_update_or_delete(sql)

    @override
    def execute_insert(self, sql: str) -> ExecuteSQLResult:
        """Execute an INSERT SQL statement on Snowflake."""
        # FIXME check sql type
        try:
            with self.connection.cursor() as cursor:
                # For INSERT operations, return affected rows and last insert ID
                rowcount = cursor.rowcount
                last_rowid = cursor.sfqid  # Snowflake query ID as identifier

                return ExecuteSQLResult(
                    sql_query=sql,
                    row_count=rowcount,
                    sql_return=str(last_rowid),
                    success=True,
                    error=None,
                )
        except Exception as e:
            ex = _handle_snowflake_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                sql_query=sql,
                error=str(ex),
            )

    @override
    def execute_update(
        self,
        sql: str,
    ) -> ExecuteSQLResult:
        """Execute an UPDATE SQL statement on Snowflake."""
        return self._execute_update_or_delete(sql)

    @override
    def execute_delete(self, sql: str) -> ExecuteSQLResult:
        """Execute a DELETE SQL statement on Snowflake."""
        return self._execute_update_or_delete(sql)

    def _execute_update_or_delete(self, sql: str):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)

                # For DELETE operations, return affected rows count
                rowcount = cursor.rowcount

                return ExecuteSQLResult(
                    sql_query=sql,
                    row_count=rowcount,
                    sql_return=str(rowcount),
                    success=True,
                    error=None,
                )
        except Exception as e:
            ex = _handle_snowflake_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                sql_query=sql,
                error=str(ex),
            )

    def _do_execute_arrow(self, input_params) -> ExecuteSQLResult:
        """Execute SQL query on Snowflake and return results in Apache Arrow format.

        Args:
            input_params: Dictionary containing sql_query and optional params

        Returns:
            ExecuteSQLResult with sql_return containing Arrow table bytes
        """
        with self.connection.cursor() as cursor:
            # Enable arrow result format
            cursor.execute("ALTER SESSION SET PYTHON_CONNECTOR_QUERY_RESULT_FORMAT='ARROW'")

            # Execute the query
            cursor.execute(
                input_params["sql_query"],
                input_params["params"] if "params" in input_params else None,
            )

            # Fetch the Arrow result
            arrow_table = cursor.fetch_arrow_all(force_return_table=True)

            # Handle case where arrow_table is None
            if arrow_table is None:
                logger.debug(f"[DEBUG] Arrow table is None for query. Row count from cursor: {cursor.rowcount}")
                row_count = 0
                # Create an empty arrow table or use None
                arrow_table = None
            else:
                row_count = arrow_table.num_rows

            # Keep the Arrow table as is for CLI compatibility
            return ExecuteSQLResult(
                sql_query=input_params["sql_query"],
                row_count=row_count,
                sql_return=arrow_table,
                success=True,
                error=None,
                result_format="arrow",
            )

    def validate_input(self, input_params: Dict[str, Any]):
        super().validate_input(input_params)
        if "params" in input_params:
            if not isinstance(input_params["params"], Sequence) and not isinstance(input_params["params"], dict):
                raise ValueError("params must be dict or Sequence")

    def get_schema(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        table_name: str = "",
        table_type: str = "table",
    ) -> List[Dict[str, Any]]:
        """
        Get schema information for tables, views, and materialized views including column name, type, nullability,
        and primary key information.

        Args:
            catalog_name: Catalog name (not used in Snowflake)
            database_name: Database name
            schema_name: Schema name
            table_name: Table name to get schema for
            table_type: Type of table_name to get schema

        Returns:
            List of dictionaries containing column information and table information
        """
        # If no table name specified, return empty list
        if not table_name:
            return []

        catalog_name = catalog_name or self.catalog_name
        database_name = database_name or self.database_name
        schema_name = schema_name or self.schema_name

        full_name = self.full_name(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_name=table_name
        )
        table_type = table_type.upper()

        describe_target = {
            "TABLE": "TABLE",
            "VIEW": "VIEW",
            "MATERIALIZED VIEW": "MATERIALIZED VIEW",
            "MATERIALIZED_VIEW": "MATERIALIZED VIEW",
            "MV": "MATERIALIZED VIEW",
        }.get(table_type, "TABLE")

        describe_sql = f"DESCRIBE {describe_target} {full_name}"

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(describe_sql)
                describe_results = cursor.fetchall()
                column_names = [col[0].lower() for col in cursor.description]
        except Exception as e:
            raise _handle_snowflake_exception(e, describe_sql) from e

        def _row_map(row: Sequence[Any]) -> Dict[str, Any]:
            return {column_names[idx]: row[idx] for idx in range(min(len(column_names), len(row)))}

        schemas: List[Dict[str, Any]] = []
        columns_list: List[Dict[str, Any]] = []
        column_index = 0

        for row in describe_results:
            row_info = _row_map(row)
            kind = (row_info.get("kind") or "COLUMN").upper()
            if "COLUMN" not in kind:
                # Skip metadata rows such as constraints output
                continue

            column_name = row_info.get("name")
            if not column_name:
                continue

            data_type = row_info.get("type", "")
            nullable_flag = str(row_info.get("null?") or row_info.get("null? ") or "").upper()
            default_value = row_info.get("default")
            comment = row_info.get("comment")
            pk_flag = str(row_info.get("primary key") or "").upper()

            column_info = {
                "cid": column_index,
                "name": column_name,
                "type": data_type,
                "nullable": nullable_flag == "Y",
                "pk": pk_flag == "Y" and table_type == "TABLE",
                "default_value": default_value,
                "comment": comment,
            }

            schemas.append(column_info)
            columns_list.append({"name": column_name, "type": data_type})
            column_index += 1

        schemas.append(
            {
                "table": table_name,
                "columns": columns_list,
                "table_type": table_type.lower(),
            }
        )

        return schemas

    def execute_query_to_df(
        self,
        sql: str,
        params: Sequence[Any] | dict[Any, Any] | None = None,
    ) -> DataFrame:
        with self.connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetch_pandas_all()

    def _sys_databases(self) -> Set[str]:
        return {"SNOWFLAKE", "SNOWFLAKE_SAMPLE_DATA"}

    def _sys_schemas(self) -> Set[str]:
        return {"INFORMATION_SCHEMA"}

    @override
    def get_databases(self, catalog_name: str = "", include_sys: bool = False) -> List[str]:
        res = self._execute_show(sql="SHOW DATABASES", result_format="arrow").sql_return
        databases = res["name"]
        if not include_sys:
            # Filter out system databases
            system_dbs = pa.array(self._sys_databases(), type=pa.string())
            import pyarrow.compute as pc

            databases = databases.filter(pc.invert(pc.is_in(databases, system_dbs)))
            databases = [db.as_py() for db in databases if db.as_py().upper() not in system_dbs]
        else:
            databases = databases.to_pylist()
        return databases

    @override
    def get_schemas(self, catalog_name: str = "", database_name: str = "", include_sys: bool = False) -> List[str]:
        """
        Get schema names using SHOW SCHEMAS command which is more efficient than INFORMATION_SCHEMA queries.

        Args:
            catalog_name: Catalog name (not used in Snowflake)
            database_name: Database name to get schemas from
            include_sys: Whether to include system schemas in the results

        Returns:
            List of schema names
        """
        database_name = database_name or self.database_name

        # Use SHOW SCHEMAS which is more efficient than querying INFORMATION_SCHEMA
        if database_name:
            sql = f'SHOW SCHEMAS IN DATABASE "{database_name}"'
        else:
            sql = "SHOW SCHEMAS"

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()

                schemas = []
                for row in results:
                    schema_name = row[1]  # Schema name is in the second column
                    sys_schemas = self._sys_schemas()
                    # Skip system schemas
                    if include_sys or schema_name.upper() not in sys_schemas:
                        schemas.append(schema_name)

                return schemas
        except Exception as e:
            logger.warning(f"Failed to get schemas using SHOW SCHEMAS, falling back to INFORMATION_SCHEMA: {e}")
            # Fallback to original method if SHOW SCHEMAS fails
            select_table_name = (
                "INFORMATION_SCHEMA.SCHEMATA" if not database_name else f'"{database_name}".INFORMATION_SCHEMA.SCHEMATA'
            )

            sql = f"SELECT SCHEMA_NAME FROM {select_table_name}"
            if not include_sys:
                sql += " WHERE SCHEMA_NAME NOT IN ('INFORMATION_SCHEMA')"

            if database_name:
                if not include_sys:
                    sql += f" AND CATALOG_NAME='{database_name}'"
                else:
                    sql += f" WHERE CATALOG_NAME='{database_name}'"
            try:
                df = self.execute_query_to_df(sql=sql)
                return [item for item in df["SCHEMA_NAME"]]
            except Exception as e:
                raise _handle_snowflake_exception(e, sql) from e

    @override
    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """
        Get table names using SHOW TABLES command which is more efficient than INFORMATION_SCHEMA queries.

        Args:
            catalog_name: Catalog name (not used in Snowflake)
            database_name: Database name to get tables from
            schema_name: Schema name to get tables from

        Returns:
            List of table names
        """
        database_name = database_name or self.database_name
        schema_name = schema_name or self.schema_name

        # Use SHOW TABLES which is more efficient than querying INFORMATION_SCHEMA
        if schema_name:
            if database_name:
                sql = f'SHOW TABLES IN SCHEMA "{database_name}"."{schema_name}"'
            else:
                sql = f'SHOW TABLES IN SCHEMA "{schema_name}"'
        elif database_name:
            sql = f'SHOW TABLES IN DATABASE "{database_name}"'
        else:
            sql = "SHOW TABLES"

        logger.info(f"Getting tables with command: {sql}")

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()

                # Extract table names from the result
                # Table name is in the second column (index 1) in SHOW TABLES result
                return [row[1] for row in results]
        except Exception as e:
            logger.warning(f"Failed to get tables using SHOW TABLES, falling back to INFORMATION_SCHEMA: {e}")
            # Fallback to original method if SHOW TABLES fails
            select_table_name = (
                "INFORMATION_SCHEMA.TABLES" if not database_name else f'"{database_name}".INFORMATION_SCHEMA.TABLES'
            )
            sql = f"SELECT TABLE_SCHEMA, TABLE_NAME FROM {select_table_name} WHERE TABLE_TYPE = 'BASE TABLE'"
            if schema_name:
                sql += f" AND TABLE_SCHEMA= '{schema_name}'"
            try:
                df = self.execute_query_to_df(sql)
                return [item for item in df["TABLE_NAME"]]
            except Exception as e:
                raise _handle_snowflake_exception(e, sql) from e

    @override
    def get_tables_with_ddl(
        self,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        tables: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Return table metadata together with their DDL definitions."""

        database_name = database_name or self.database_name
        schema_name = schema_name or self.schema_name

        filter_tables = self._reset_filter_tables(
            tables=tables,
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
        )

        show_sql: str
        if schema_name:
            if database_name:
                show_sql = f'SHOW TERSE TABLES IN SCHEMA "{database_name}"."{schema_name}"'
            else:
                show_sql = f'SHOW TERSE TABLES IN SCHEMA "{schema_name}"'
        elif database_name:
            show_sql = f'SHOW TERSE TABLES IN DATABASE "{database_name}"'
        else:
            show_sql = "SHOW TERSE TABLES"

        column_names: List[str] = []
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(show_sql)
                show_results = cursor.fetchall()
                if cursor.description:
                    column_names = [col[0].lower() for col in cursor.description]
        except Exception as e:
            raise _handle_snowflake_exception(e, show_sql) from e

        table_entries: List[Dict[str, str]] = []
        seen: Set[tuple[str, str, str]] = set()
        for row in show_results:
            row_dict = {
                column_names[idx]: row[idx] for idx in range(min(len(column_names), len(row))) if column_names[idx]
            }

            table_name = row_dict.get("name") or row_dict.get("table_name")
            if not table_name and len(row) > 1:
                table_name = row[1]
            if not table_name:
                continue

            row_database = row_dict.get("database_name") or database_name or ""
            row_schema = row_dict.get("schema_name") or schema_name or ""

            full_name = self.full_name(database_name=row_database, schema_name=row_schema, table_name=table_name)

            if filter_tables and full_name not in filter_tables:
                continue

            key = (row_database or "", row_schema or "", table_name)
            if key in seen:
                continue
            seen.add(key)

            table_entries.append(
                {
                    "catalog_name": catalog_name or "",
                    "database_name": row_database or "",
                    "schema_name": row_schema or "",
                    "table_name": table_name,
                    "full_name": full_name,
                }
            )

        if not table_entries:
            return []

        ddl_batch_sql = """
            SELECT
                value:database_name::string AS database_name,
                value:schema_name::string AS schema_name,
                value:table_name::string AS table_name,
                GET_DDL('TABLE', value:full_name::string) AS ddl
            FROM TABLE(FLATTEN(INPUT => PARSE_JSON(%s)))
        """

        results: List[Dict[str, str]] = []
        batch_size = 10

        for start in range(0, len(table_entries), batch_size):
            batch = table_entries[start : start + batch_size]
            payload = json.dumps(batch)
            batch_lookup = {
                (entry["database_name"], entry["schema_name"], entry["table_name"]): entry for entry in batch
            }

            try:
                with self.connection.cursor() as cursor:
                    cursor.execute(ddl_batch_sql, (payload,))
                    ddl_rows = cursor.fetchall()
            except Exception as e:
                logger.warning("Bulk GET_DDL failed for Snowflake tables, falling back to per-table requests: %s", e)
                ddl_rows = self._fetch_table_ddls_individually(batch)
                # _fetch_table_ddls_individually already returns tuples, skip further processing
                for db_name, sch_name, tbl_name, ddl in ddl_rows:
                    entry = batch_lookup.get((db_name, sch_name, tbl_name))
                    if not entry:
                        continue
                    results.append(
                        {
                            "identifier": self.identifier(
                                database_name=db_name,
                                schema_name=sch_name,
                                table_name=tbl_name,
                            ),
                            "catalog_name": entry.get("catalog_name", ""),
                            "database_name": db_name,
                            "schema_name": sch_name,
                            "table_name": tbl_name,
                            "definition": ddl,
                            "table_type": "table",
                        }
                    )
                continue

            for row in ddl_rows:
                db_name = (row[0] or "").strip()
                sch_name = (row[1] or "").strip()
                tbl_name = (row[2] or "").strip()
                ddl = row[3] if len(row) > 3 and row[3] else ""

                entry = batch_lookup.get((db_name, sch_name, tbl_name))
                if not entry:
                    continue

                results.append(
                    {
                        "identifier": self.identifier(
                            database_name=db_name,
                            schema_name=sch_name,
                            table_name=tbl_name,
                        ),
                        "catalog_name": entry.get("catalog_name", ""),
                        "database_name": db_name,
                        "schema_name": sch_name,
                        "table_name": tbl_name,
                        "definition": ddl,
                        "table_type": "table",
                    }
                )

        return results

    def _fetch_table_ddls_individually(self, entries: List[Dict[str, str]]):
        """Fallback to retrieve table DDLs one by one when bulk retrieval fails."""

        results = []
        with self.connection.cursor() as cursor:
            for entry in entries:
                full_name = entry["full_name"]
                try:
                    cursor.execute("SELECT GET_DDL('TABLE', %s)", (full_name,))
                    row = cursor.fetchone()
                    ddl = row[0] if row else ""
                except Exception as e:  # pragma: no cover - depends on permissions/state
                    logger.warning("Failed to get DDL for %s: %s", full_name, e)
                    ddl = f"-- DDL not available for {full_name}: {e}"

                results.append(
                    (
                        entry.get("database_name", ""),
                        entry.get("schema_name", ""),
                        entry.get("table_name", ""),
                        ddl,
                    )
                )

        return results

    def get_type(self) -> str:
        return DBType.SNOWFLAKE

    @override
    def execute_query(
        self, sql: str, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"
    ) -> ExecuteSQLResult:
        if sql.lower().startswith("show"):
            return self._execute_show(sql, result_format)
        if result_format == "csv":
            result = self.execute_csv(sql)
            return result
        elif result_format == "pandas":
            return self.execute_pandas(sql)
        else:
            result = self.execute_arrow(sql)
            if result_format == "arrow":
                return result
            if result and result.success:
                result.sql_return = result.sql_return.to_pylist()
                result.result_format = result_format
            return result

    def _execute_show(
        self, sql: str, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"
    ) -> ExecuteSQLResult:
        sql = sql.strip()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()
                col_names = [col[0] for col in cursor.description][:7]
                row_count = len(result)
                if result:
                    columns = list(zip(*[row[:7] for row in result]))
                    arrow_result = pa.Table.from_arrays([pa.array(col) for col in columns], names=col_names)
                else:
                    arrow_result = pa.Table.from_arrays([])

                if result_format == "arrow":
                    final_result = arrow_result
                elif result_format == "list":
                    final_result = arrow_result.to_pylist()
                else:
                    df = arrow_result.to_pandas()
                    final_result = df if result_format == "pandas" else df.to_csv(index=False)
                return ExecuteSQLResult(
                    success=True,
                    result_format=result_format,
                    sql_return=final_result,
                    row_count=row_count,
                )
        except Exception as e:
            ex = _handle_snowflake_exception(e, sql)
            return ExecuteSQLResult(success=False, sql_query=sql, result_format=result_format, error=str(ex))

    def execute_pandas(self, sql: str) -> ExecuteSQLResult:
        try:
            df = self.execute_query_to_df(sql)
            return ExecuteSQLResult(
                sql_query=sql,
                row_count=len(df),
                sql_return=df,
                success=True,
                error=None,
                result_format="pandas",
            )
        except Exception as e:
            ex = _handle_snowflake_exception(e, sql)
            return ExecuteSQLResult(success=False, sql_query=sql, result_format="pandas", error=str(ex))

    def execute_arrow(self, sql: str) -> ExecuteSQLResult:
        """Execute a SQL query and return results in Arrow format.

        Args:
            sql: SQL query string to execute

        Returns:
            ExecuteSQLResult with Arrow data
        """
        input_params = {"sql_query": sql}
        try:
            return self._do_execute_arrow(input_params)
        except Exception as e:
            ex = _handle_snowflake_exception(e, sql)
            return ExecuteSQLResult(success=False, sql_query=sql, error=str(ex))

    def execute_csv(self, query: str) -> ExecuteSQLResult:
        """Execute a SQL query and return results in CSV format.

        Args:
            query: SQL query string to execute

        Returns:
            ExecuteSQLResult with CSV data
        """
        result = self.execute_pandas(query)
        result.result_format = "csv"
        if result.success and result.row_count > 0:
            result.sql_return = result.sql_return.to_csv(index=False)
        return result

    @override
    def execute_content_set(self, sql_query: str) -> ExecuteSQLResult:
        """Execute a SQL query and return results in Context format."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql_query)
            return ExecuteSQLResult(
                success=True,
                sql_query=sql_query,
                sql_return="Successful",
                row_count=0,
            )
        except Exception as e:
            ex = _handle_snowflake_exception(e, sql_query)
            return ExecuteSQLResult(success=False, sql_query=sql_query, error=str(ex))

    def execute_queries(self, queries: List[str]) -> List[ExecuteSQLResult]:
        """Execute multiple SQL queries on Snowflake.

        Args:
            queries: List of SQL queries to execute

        Returns:
            List of ExecuteSQLResult for each query
        """
        results = []
        for sql in queries:
            result = self.execute_query(sql)
            results.append(result)
        return results

    def execute_queries_arrow(self, queries: List[str]) -> List[ExecuteSQLResult]:
        """Execute multiple SQL queries on Snowflake and return results in Arrow format.

        Args:
            queries: List of SQL queries to execute

        Returns:
            List of ExecuteSQLResult for each query with Arrow data
        """
        results = []
        for sql in queries:
            result = self.execute_arrow(sql)
            results.append(result)
        return results

    @override
    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        if schema_name:
            full_name = f'"{schema_name}"."{table_name}"'
        else:
            full_name = f'"{table_name}"'
        return full_name if not database_name else f'"{database_name}".{full_name}'
