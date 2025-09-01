# -*- coding: utf-8 -*-
import json
from typing import Any, Callable, List, Optional

from agents import FunctionTool, Tool, function_tool
from pydantic import BaseModel, Field

from datus.tools.db_tools import BaseSqlConnector
from datus.utils.compress_utils import DataCompressor
from datus.utils.constants import SUPPORT_CATALOG_DIALECTS, SUPPORT_DATABASE_DIALECTS, SUPPORT_SCHEMA_DIALECTS, DBType


class FuncToolResult(BaseModel):
    success: int = Field(
        default=1, description="Whether the execution is successful or not, 1 is success, 0 is failure", init=True
    )
    error: Optional[str] = Field(
        default=None, description="Error message: field is not empty when success=0", init=True
    )
    result: Optional[Any] = Field(default=None, description="Result of the execution", init=True)


def trans_to_function_tool(bound_method: Callable) -> FunctionTool:
    """
    Transfer a bound method to a function tool.
    This method is to solve the problem that `@function_tool` can only be applied to static methods
    """
    tool_template = function_tool(bound_method)

    corrected_schema = json.loads(json.dumps(tool_template.params_json_schema))
    if "self" in corrected_schema.get("properties", {}):
        del corrected_schema["properties"]["self"]
    if "self" in corrected_schema.get("required", []):
        corrected_schema["required"].remove("self")

    # The invoker MUST be an `async` function.
    # We define a closure to correctly capture the `bound_method` for each iteration.
    def create_async_invoker(method_to_call: Callable) -> Callable:
        async def final_invoker(tool_ctx, args_str: str) -> dict:
            """
            This is an async wrapper around our synchronous tool method.
            The agent framework will `await` this coroutine.
            """
            # The actual work (JSON parsing, method call) is synchronous.
            args_dict = json.loads(args_str)
            result_dict = method_to_call(**args_dict)
            if isinstance(result_dict, FuncToolResult):
                result_dict = result_dict.model_dump()
            return result_dict

        return final_invoker

    async_invoker = create_async_invoker(bound_method)

    final_tool = FunctionTool(
        name=tool_template.name,
        description=tool_template.description,
        params_json_schema=corrected_schema,
        on_invoke_tool=async_invoker,  # <--- Assign the async function
    )
    return final_tool


class DBFuncTool:
    def __init__(self, connector: BaseSqlConnector):
        self.connector = connector
        self.compressor = DataCompressor()

    def available_tools(self) -> List[Tool]:
        bound_tools = []
        methods_to_convert: List[Callable] = [
            self.list_tables,
            self.describe_table,
            self.read_query,
        ]

        if self.connector.dialect in SUPPORT_CATALOG_DIALECTS:
            bound_tools.append(trans_to_function_tool(self.list_catalogs))

        if self.connector.dialect in SUPPORT_DATABASE_DIALECTS:
            bound_tools.append(trans_to_function_tool(self.list_databases))

        if self.connector.dialect in SUPPORT_SCHEMA_DIALECTS:
            bound_tools.append(trans_to_function_tool(self.list_schemas))

        for bound_method in methods_to_convert:
            bound_tools.append(trans_to_function_tool(bound_method))
        return bound_tools

    def list_catalogs(self) -> FuncToolResult:
        """
        List all catalogs in the database.

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[List[str]]): A list of catalog names on success.
        """
        try:
            catalogs = self.connector.get_catalogs()
            return FuncToolResult(result=catalogs)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def list_databases(self, catalog: Optional[str] = "", include_sys: Optional[bool] = False) -> FuncToolResult:
        """
        List all databases in the database system.

        Args:
            catalog: Optional catalog name to filter databases (depends on database type)
            include_sys: Whether to include system databases in the results

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[List[str]]): A list of database names on success.
        """
        try:
            databases = self.connector.get_databases(catalog, include_sys=include_sys)
            return FuncToolResult(result=databases)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def list_schemas(
        self, catalog: Optional[str] = "", database: Optional[str] = "", include_sys: bool = False
    ) -> FuncToolResult:
        """
        List all schemas in the database.

        Args:
            catalog: Optional catalog name to filter schemas
            database: Optional database name to filter schemas
            include_sys: Whether to include system schemas in the results

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[List[str]]): A list of schema names on success.
        """
        try:
            schemas = self.connector.get_schemas(catalog, database, include_sys=include_sys)
            return FuncToolResult(result=schemas)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def list_tables(
        self,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        include_views: Optional[bool] = True,
    ) -> FuncToolResult:
        """
        List all tables, views, and materialized views in the database.

        Args:
            catalog: Optional catalog name to filter tables
            database: Optional database name to filter tables
            schema_name: Optional schema name to filter tables
            include_views: Whether to include views and materialized views in results

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[List[Dict[str, str]]]): A list of table names and type on success.
        """
        try:
            result = []
            for tb in self.connector.get_tables(catalog, database, schema_name):
                result.append({"type": "table", "name": tb})

            if not include_views:
                return FuncToolResult(result=result)

            # Add views
            try:
                views = self.connector.get_views(catalog, database, schema_name)
                for view in views:
                    result.append({"type": "view", "name": view})
            except (NotImplementedError, AttributeError):
                # Some connectors may not support get_views
                pass

            # Add materialized views
            try:
                materialized_views = self.connector.get_materialized_views(catalog, database, schema_name)
                for mv in materialized_views:
                    result.append({"type": "materialized_view", "name": mv})
            except (NotImplementedError, AttributeError):
                # Some connectors may not support get_materialized_views
                pass

            return FuncToolResult(result=result)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def describe_table(
        self,
        table_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Get the schema information for a specific table.

        Args:
            table_name: The name of the table to describe
            catalog: Optional catalog name
            database: Optional database name
            schema_name: Optional schema name

        Returns:
        dict: A dictionary representing the execution result with the following keys:
            - 'success' (int): 1 if the execution was successful, 0 otherwise.
            - 'error' (Optional[str]): An error message if success is 0, otherwise None.
            - 'result' (Optional[Any]): The result of the execution. For this function,
                                        it's a list of dictionaries, each representing a table.
        """
        try:
            result = self.connector.get_schema(
                catalog_name=catalog, database_name=database, schema_name=schema_name, table_name=table_name
            )
            return FuncToolResult(result=result)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def read_query(self, sql: str) -> FuncToolResult:
        """
        Execute a SQL query and return the results.

        Args:
            sql: The SQL query to execute

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[dict]): Query results on success, including original_rows, original_columns,
                   is_compressed, and compressed_data.
        """
        try:
            result = self.connector.execute_query(
                sql, result_format="arrow" if self.connector.dialect == DBType.SNOWFLAKE else "list"
            )
            if result.success:
                data = result.sql_return
                return FuncToolResult(result=self.compressor.compress(data))
            else:
                return FuncToolResult(success=0, error=result.error)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))
