from typing import Any, Dict, List, Literal, Sequence

import snowflake.connector

from datus.schemas.node_models import ExecuteSQLResult
from datus.tools.db_tools.base import BaseSqlConnector
from datus.utils.loggings import get_logger

logger = get_logger("snowflake_connector")


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
    ):
        super().__init__(dialect="snowflake")
        self.connection = snowflake.connector.Connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database=database if database else None,
            schema=schema if schema else None,
        )

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

    def do_execute(self, input_params, result_format: Literal["csv", "arrow", "list"] = "csv"):
        """Execute SQL query on Snowflake."""
        try:
            with self.connection.cursor() as cursor:
                # 设置游标返回字典格式的结果
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
        except snowflake.connector.errors.ProgrammingError as e:
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
        except snowflake.connector.errors.ProgrammingError as e:
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

    def get_schema(self, schema_name: str = "", **kwargs) -> List[Dict[str, str]]:
        """
        Get the schema of the database.
        1. Get All Databases
        SELECT DATABASE_NAME
        FROM INFORMATION_SCHEMA.DATABASES
        WHERE DATABASE_OWNER IS NOT NULL; -- filter system database
        2. Get All Schemas
        SELECT
        'SELECT GET_DDL(''TABLE'', ''' || TABLE_SCHEMA || '.' || TABLE_NAME || ''') AS DDL;' AS GENERATED_QUERY
        FROM
        AUSTIN.INFORMATION_SCHEMA.TABLES
        WHERE
        TABLE_TYPE = 'BASE TABLE'; -- Only base table, exclude view
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT DATABASE_NAME  FROM INFORMATION_SCHEMA.DATABASES  WHERE DATABASE_OWNER IS NOT NULL; "
            )
            databases = cursor.fetchall()
            schema_list = []
            for database in databases:
                cursor.execute(
                    """ SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME,
                'SELECT GET_DDL(''TABLE'', ''' || TABLE_SCHEMA || '.' || TABLE_NAME || ''') AS DDL;' AS GENERATED_QUERY
                FROM
                {database_name}.INFORMATION_SCHEMA.TABLES
                WHERE
                TABLE_TYPE = 'BASE TABLE';""".format(
                        database_name=database[0]
                    )
                )
                schemas = cursor.fetchall()
                for schema in schemas:
                    cursor.execute("SELECT GET_DDL('TABLE', '{}') AS DDL;".format(schema[0]))
                    schema_list.append(
                        {
                            "database_name": schema[0],
                            "schema_name": schema[1],
                            "table_name": schema[2],
                            "definition": cursor.fetchone()[0],
                            "table_type": "table",
                        }
                    )
            return schema_list

    def get_type(self) -> str:
        return "snowflake"

    def execute_arrow(self, query: str) -> ExecuteSQLResult:
        """Execute a SQL query and return results in Arrow format.

        Args:
            query: SQL query string to execute

        Returns:
            ExecuteSQLResult with Arrow data
        """
        input_params = {"sql_query": query}
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
