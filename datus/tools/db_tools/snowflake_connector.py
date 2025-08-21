from typing import Any, Dict, List, Literal, Sequence, Set, override

from pandas import DataFrame
from snowflake.connector import Connect, SnowflakeConnection
from snowflake.connector.errors import ProgrammingError

from datus.schemas.node_models import ExecuteSQLResult
from datus.tools.db_tools.base import BaseSqlConnector
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


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

    def do_execute(self, input_params, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"):
        """Execute SQL query on Snowflake."""
        try:
            with self.connection.cursor() as cursor:
                # Set cursor to return dictionary format results
                cursor.execute(
                    input_params["sql_query"],
                    input_params["params"] if "params" in input_params else None,
                )

                # got columns
                columns = [desc[0] for desc in cursor.description]

                # got result from cursor
                if result_format == "arrow":
                    final_result = cursor.fetch_arrow_all()
                    # Handle case where arrow result is None
                    if final_result is None:
                        row_count = 0
                    else:
                        row_count = final_result.num_rows
                elif result_format == "list":
                    rows = cursor.fetchall()
                    final_result = [{columns[i]: value for i, value in enumerate(row)} for row in rows]
                    row_count = len(rows)
                elif result_format == "pandas":
                    rows = cursor.fetch_pandas_all()
                    final_result = rows
                    row_count = len(rows)
                else:
                    rows = cursor.fetch_pandas_all()
                    final_result = rows.to_csv(index=False)
                    row_count = len(rows)

                return ExecuteSQLResult(
                    sql_query=input_params["sql_query"],
                    row_count=row_count,
                    sql_return=final_result,
                    success=True,
                    error=None,
                    result_format=result_format,
                )
        except ProgrammingError as e:
            return ExecuteSQLResult(
                sql_query=input_params["sql_query"] if isinstance(input_params, dict) else str(input_params),
                row_count=0,
                sql_return="",
                success=True,  # Continue to execute next step if failed, reflection node will handle it
                error=f"errno:{e.errno}, sqlstate: {e.sqlstate}, message: {e.msg}, query_id: {e.sfqid}",
                result_format="csv",
            )
        except Exception as e:
            return ExecuteSQLResult(
                sql_query=input_params["sql_query"] if isinstance(input_params, dict) else str(input_params),
                row_count=0,
                sql_return="",
                success=False,
                error=f"Unknown error: {str(e)}",
                result_format="csv",
            )

    def do_switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        with self.connection.cursor() as cursor:
            if not schema_name:
                if not database_name:
                    return
                cursor.execute(f'USE DATABASE "{database_name}"')
            else:
                cursor.execute(f'USE SCHEMA "{database_name}"."{schema_name}"')

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
        except ProgrammingError as e:
            return ExecuteSQLResult(
                sql_query=sql,
                row_count=0,
                sql_return="",
                success=False,
                error=f"errno:{e.errno}, sqlstate: {e.sqlstate}, message: {e.msg}, query_id: {e.sfqid}",
            )
        except Exception as e:
            return ExecuteSQLResult(
                sql_query=sql,
                row_count=0,
                sql_return="",
                success=False,
                error=f"Unknown error: {str(e)}",
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
        except ProgrammingError as e:
            return ExecuteSQLResult(
                sql_query=sql,
                row_count=0,
                sql_return="",
                success=False,
                error=f"errno:{e.errno}, sqlstate: {e.sqlstate}, message: {e.msg}, query_id: {e.sfqid}",
            )
        except Exception as e:
            return ExecuteSQLResult(
                sql_query=sql,
                row_count=0,
                sql_return="",
                success=False,
                error=f"Unknown error: {str(e)}",
            )

    def do_execute_arrow(self, input_params) -> ExecuteSQLResult:
        """Execute SQL query on Snowflake and return results in Apache Arrow format.

        Args:
            input_params: Dictionary containing sql_query and optional params

        Returns:
            ExecuteSQLResult with sql_return containing Arrow table bytes
        """
        try:
            with self.connection.cursor() as cursor:
                # Enable arrow result format
                cursor.execute("ALTER SESSION SET PYTHON_CONNECTOR_QUERY_RESULT_FORMAT='ARROW'")

                # Execute the query
                cursor.execute(
                    input_params["sql_query"],
                    input_params["params"] if "params" in input_params else None,
                )

                # Fetch the Arrow result
                arrow_table = cursor.fetch_arrow_all()

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
        except ProgrammingError as e:
            logger.debug(f"[DEBUG] Snowflake ProgrammingError: errno={e.errno}, sqlstate={e.sqlstate}, msg={e.msg}")
            return ExecuteSQLResult(
                sql_query=input_params["sql_query"] if isinstance(input_params, dict) else str(input_params),
                row_count=0,
                sql_return="",
                success=True,  # Continue to execute next step if failed, reflection node will handle it
                error=f"errno:{e.errno}, sqlstate: {e.sqlstate}, message: {e.msg}, query_id: {e.sfqid}",
                result_format="arrow",
            )
        except Exception as e:
            logger.debug(f"[DEBUG] Snowflake General Exception: {type(e).__name__}: {str(e)}")
            import traceback

            logger.debug(f"[DEBUG] Traceback: {traceback.format_exc()}")
            return ExecuteSQLResult(
                sql_query=input_params["sql_query"] if isinstance(input_params, dict) else str(input_params),
                row_count=0,
                sql_return="",
                success=False,
                error=f"Unknown error: {str(e)}",
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

        with self.connection.cursor() as cursor:
            # Build the query to get column information from INFORMATION_SCHEMA
            catalog_name = catalog_name or self.catalog_name
            database_name = database_name or self.database_name
            schema_name = schema_name or self.schema_name
            full_name = self.full_name(
                catalog_name=catalog_name, database_name=database_name, schema_name=schema_name, table_name=table_name
            )

            table_type = table_type.upper()

            # Initialize primary key columns - only for base tables
            pk_columns = set()
            if table_type == "TABLE":
                try:
                    # Get primary key information using SHOW PRIMARY KEYS command
                    cursor.execute(f"""SHOW PRIMARY KEYS IN TABLE {full_name}""")
                    pk_results = cursor.fetchall()
                    # Extract column names that are primary keys
                    # The column name is in position 4 (0-indexed) according to Snowflake documentation
                    pk_columns = set(row[4] for row in pk_results)
                except Exception as e:
                    # If SHOW PRIMARY KEYS fails (e.g., for views), skip primary key detection
                    logger.debug(f"Failed to get primary keys for {full_name}: {e}")

            columns_table_name = (
                "INFORMATION_SCHEMA.COLUMNS" if not database_name else f'"{database_name}".INFORMATION_SCHEMA.COLUMNS'
            )
            # Get column information
            columns_query = f"""
                SELECT
                    ordinal_position,
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    comment
                FROM {columns_table_name}
                WHERE table_name = '{table_name}'
            """

            if schema_name:
                columns_query += f" AND table_schema = '{schema_name}'"

            columns_query += " ORDER BY ordinal_position"

            cursor.execute(columns_query)
            columns_results = cursor.fetchall()

            # Process column information
            schemas: List[Dict[str, Any]] = []
            columns_list = []

            for row in columns_results:
                ordinal_position = row[0]
                column_name = row[1]
                data_type = row[2]
                is_nullable = row[3]
                column_default = row[4]
                comment = row[5]

                column_info = {
                    "cid": ordinal_position - 1,  # Convert to 0-based index
                    "name": column_name,
                    "type": data_type,
                    "nullable": is_nullable.upper() == "YES",
                    "pk": column_name in pk_columns,
                    "default_value": column_default,
                    "comment": comment,
                }

                schemas.append(column_info)
                columns_list.append({"name": column_name, "type": data_type})

            # Add table information with type
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
        return {"PUBLIC", "INFORMATION_SCHEMA"}

    def _execute_list(
        self,
        sql: str,
        params: Sequence[Any] | dict[Any, Any] | None = None,
    ) -> list[tuple] | list[dict]:
        with self.connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()

    @override
    def get_databases(self, catalog_name: str = "", include_sys: bool = False) -> List[str]:
        res = self._execute_list(sql="SHOW DATABASES")
        databases = [it[1] for it in res]
        if not include_sys:
            # Filter out system databases
            system_dbs = self._sys_databases()
            databases = [db for db in databases if db.upper() not in system_dbs]
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
                sql += " WHERE SCHEMA_NAME NOT IN ('PUBLIC', 'INFORMATION_SCHEMA')"

            if database_name:
                if not include_sys:
                    sql += f" AND CATALOG_NAME='{database_name}'"
                else:
                    sql += f" WHERE CATALOG_NAME='{database_name}'"

            df = self.execute_query_to_df(sql=sql)
            return [item for item in df["SCHEMA_NAME"]]

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

            df = self.execute_query_to_df(sql)
            return [item for item in df["TABLE_NAME"]]

    def get_type(self) -> str:
        return DBType.SNOWFLAKE

    @override
    def execute_query(self, query_sql: str) -> ExecuteSQLResult:
        return self.do_execute_arrow({"sql_query": query_sql})

    def execute_pandas(self, query_sql: str) -> ExecuteSQLResult:
        try:
            df = self.execute_query_to_df(query_sql)
            return ExecuteSQLResult(
                sql_query=query_sql,
                row_count=len(df),
                sql_return=df,
                success=True,
                error=None,
                result_format="pandas",
            )
        except Exception as e:
            return ExecuteSQLResult(
                sql_query=query_sql,
                row_count=0,
                sql_return=None,
                success=False,
                error=str(e),
                result_format="pandas",
            )

    def execute_arrow(self, query_sql: str) -> ExecuteSQLResult:
        """Execute a SQL query and return results in Arrow format.

        Args:
            query_sql: SQL query string to execute

        Returns:
            ExecuteSQLResult with Arrow data
        """
        input_params = {"sql_query": query_sql}
        return self.do_execute_arrow(input_params)

    def execute_csv(self, query: str) -> ExecuteSQLResult:
        """Execute a SQL query and return results in CSV format.

        Args:
            query: SQL query string to execute

        Returns:
            ExecuteSQLResult with CSV data
        """
        input_params = {"sql_query": query}
        return self.do_execute(input_params, result_format="csv")

    def execute_queries(self, queries: List[str]) -> List[ExecuteSQLResult]:
        """Execute multiple SQL queries on Snowflake.

        Args:
            queries: List of SQL queries to execute

        Returns:
            List of ExecuteSQLResult for each query
        """
        results = []
        for sql in queries:
            input_params = {"sql_query": sql}
            result = self.do_execute(input_params)
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
            input_params = {"sql_query": sql}
            result = self.do_execute_arrow(input_params)
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
