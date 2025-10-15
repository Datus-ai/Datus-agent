# -*- coding: utf-8 -*-
import json
import warnings
from typing import Any, Callable, List, Optional

from agents import FunctionTool, Tool, function_tool
from pydantic import BaseModel, Field

from datus.configuration.agent_config import AgentConfig
from datus.tools.db_tools import BaseSqlConnector
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.utils.compress_utils import DataCompressor
from datus.utils.constants import SUPPORT_CATALOG_DIALECTS, SUPPORT_DATABASE_DIALECTS, SUPPORT_SCHEMA_DIALECTS, DBType

# Suppress Pydantic field name shadowing warnings
warnings.filterwarnings("ignore", message=".*shadows an attribute in parent.*")


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
    This method is to solve the problem that '@function_tool' can only be applied to static methods
    """
    tool_template = function_tool(bound_method)

    corrected_schema = json.loads(json.dumps(tool_template.params_json_schema))
    if "self" in corrected_schema.get("properties", {}):
        del corrected_schema["properties"]["self"]
    if "self" in corrected_schema.get("required", []):
        corrected_schema["required"].remove("self")

    # The invoker MUST be an 'async' function.
    # We define a closure to correctly capture the 'bound_method' for each iteration.
    def create_async_invoker(method_to_call: Callable) -> Callable:
        async def final_invoker(tool_ctx, args_str: str) -> dict:
            """
            This is an async wrapper for tool methods.
            The agent framework will 'await' this coroutine.
            """
            # The actual work (JSON parsing, method call)
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
    def __init__(self, connector: BaseSqlConnector, agent_config: Optional[AgentConfig] = None):
        self.connector = connector
        self.compressor = DataCompressor()
        self.agent_config = agent_config
        self.schema_rag = None
        if agent_config:
            from datus.storage.schema_metadata.store import rag_by_configuration as schema_metadata_by_configuration

            self.schema_rag = schema_metadata_by_configuration(agent_config)

    def available_tools(self) -> List[Tool]:
        bound_tools = []
        methods_to_convert: List[Callable] = [
            self.list_tables,
            self.describe_table,
            self.read_query,
            self.get_table_ddl,
        ]

        if self.connector.dialect in SUPPORT_CATALOG_DIALECTS:
            bound_tools.append(trans_to_function_tool(self.list_catalogs))

        if self.connector.dialect in SUPPORT_DATABASE_DIALECTS:
            bound_tools.append(trans_to_function_tool(self.list_databases))

        if self.connector.dialect in SUPPORT_SCHEMA_DIALECTS:
            bound_tools.append(trans_to_function_tool(self.list_schemas))

        # Add search_table if schema_rag is available
        if self.schema_rag:
            bound_tools.append(trans_to_function_tool(self.search_table))

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
        List all schemas within a database. Schemas are logical containers that organize tables and other database
         objects.

        Use this tool when you need to:
        - Discover what schemas exist in a database
        - Navigate to specific schemas for table exploration
        - Find schemas related to specific applications or business areas

        Args:
            catalog: Optional catalog name to filter schemas. Leave empty if not specified.
            database: Optional database name to filter schemas. Leave empty if not specified.
            include_sys: Whether to include system schemas (default False). Set to True for maintenance tasks.

        Returns:
            dict: Schema list containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if operation failed
                - 'result' (list): List of schema names
        """
        try:
            schemas = self.connector.get_schemas(catalog, database, include_sys=include_sys)
            return FuncToolResult(result=schemas)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def search_table(
        self,
        query_text: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        db_schema: Optional[str] = "",
        top_n: int = 5,
        simple_sample_data: bool = True,
    ) -> FuncToolResult:
        """
        Search for database tables using natural language queries with vector similarity.

        This tool helps find relevant tables by searching through table names, schemas (DDL),
        and sample data using semantic search. Use this FIRST before describe_table to
        efficiently discover relevant tables.

        Use this tool when you need to:
        - Find tables related to a specific business concept or domain
        - Discover tables containing certain types of data
        - Locate tables for SQL query development
        - Understand what tables are available in a database

        **Application Guidance**:
        1. If table matches (via definition/sample_data), use it directly
        2. If partitioned (e.g., date-based in definition), explore correct partition via describe_table
        3. If no match, use list_tables for broader exploration

        Args:
            query_text: Natural language description of what you're looking for
                       (e.g., "customer data", "sales transactions", "user profiles")
            catalog: Optional catalog name to filter search results. Leave empty if not specified.
            database: Optional database name to filter search results. Leave empty if not specified.
            db_schema: Optional schema name to filter search results. Leave empty if not specified.
            top_n: Maximum number of results to return (default 5)
            simple_sample_data: If True, return simplified sample data without catalog/database/schema fields

        Returns:
            dict: Search results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if search failed
                - 'result' (dict): Search results with:
                    - 'metadata' (list): Table information including catalog_name, database_name, schema_name,
                         table_name, table_type ('table'/'view'/'mv'), definition (DDL), identifier, and _distance
                    - 'sample_data' (list): Sample rows from matching tables with identifier, table_type,
                         sample_rows, and _distance

        Example:
            search_table("customer information", database="prod", top_n=3)
            Returns tables with DDL and sample data related to customers
        """
        if not self.schema_rag:
            return FuncToolResult(success=0, error="search_table is not available. schema_rag not initialized.")

        try:
            metadata, sample_values = self.schema_rag.search_similar(
                query_text,
                catalog_name=catalog,
                database_name=database,
                schema_name=db_schema,
                table_type="full",
                top_n=top_n,
            )
            result_dict = {"metadata": [], "sample_data": []}
            if metadata:
                result_dict["metadata"] = metadata.select(
                    [
                        "catalog_name",
                        "database_name",
                        "schema_name",
                        "table_name",
                        "table_type",
                        "definition",
                        "identifier",
                        "_distance",
                    ]
                ).to_pylist()

            if sample_values:
                if simple_sample_data:
                    selected_fields = ["identifier", "table_type", "sample_rows", "_distance"]
                else:
                    selected_fields = [
                        "identifier",
                        "catalog_name",
                        "database_name",
                        "schema_name",
                        "table_type",
                        "table_name",
                        "sample_rows",
                        "_distance",
                    ]
                result_dict["sample_data"] = sample_values.select(selected_fields).to_pylist()
            return FuncToolResult(success=1, error=None, result=result_dict)
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
        Get the complete schema information for a specific table, including column definitions, data types, and
         constraints.

        Use this tool when you need to:
        - Understand the structure of a specific table
        - Get column names, data types, and constraints for SQL query writing
        - Analyze table schema for data modeling or analysis
        - Verify table structure before running queries

        **IMPORTANT**: Only use AFTER search_table if no match or for partitioned tables.
        Always prefer search_table first for semantic table discovery.

        Args:
            table_name: Name of the table to describe
            catalog: Optional catalog name for precise table identification. Leave empty if not specified.
            database: Optional database name for precise table identification. Leave empty if not specified.
            schema_name: Optional schema name for precise table identification. Leave empty if not specified.

        Returns:
            dict: Table schema information containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if operation failed
                - 'result' (list): Detailed table schema including columns, data types, and constraints
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

    def get_table_ddl(
        self,
        table_name: str,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Get complete DDL definition for a database table.

        Use this tool when you need to:
        - Generate semantic models (LLM needs complete DDL for accurate generation)
        - Understand table structure including constraints, indexes, and relationships
        - Analyze foreign key relationships for semantic model generation

        Args:
            table_name: Name of the database table
            catalog: Optional catalog name to filter tables
            database: Optional database name to filter tables
            schema_name: Optional schema name to filter tables

        Returns:
            dict: DDL results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (dict): Contains:
                    - 'identifier' (str): Full table identifier
                    - 'catalog_name' (str): Catalog name
                    - 'database_name' (str): Database name
                    - 'schema_name' (str): Schema name
                    - 'table_name' (str): Table name
                    - 'definition' (str): Complete CREATE TABLE DDL statement
                    - 'table_type' (str): Table type (table, view, etc.)
        """
        try:
            # Get tables with DDL
            tables_with_ddl = self.connector.get_tables_with_ddl(
                catalog_name=catalog, database_name=database, schema_name=schema_name, tables=[table_name]
            )

            if not tables_with_ddl:
                return FuncToolResult(success=0, error=f"Table '{table_name}' not found or no DDL available")

            # Return the first (and only) table's DDL
            table_info = tables_with_ddl[0]
            return FuncToolResult(result=table_info)

        except Exception as e:
            return FuncToolResult(success=0, error=str(e))


def db_function_tool_instance(agent_config: AgentConfig, database_name: str = "") -> DBFuncTool:
    db_manager = db_manager_instance(agent_config.namespaces)
    return DBFuncTool(
        db_manager.get_conn(agent_config.current_namespace, database_name or agent_config.current_database),
        agent_config=agent_config,
    )


def db_function_tools(agent_config: AgentConfig, database_name: str = "") -> List[Tool]:
    return db_function_tool_instance(agent_config, database_name).available_tools()
