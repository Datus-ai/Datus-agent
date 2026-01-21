# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""
Datus MCP Server

This module implements a Model Context Protocol (MCP) server that exposes
Datus's database and context search tools as MCP-compatible tools.

Supported Transport Modes:
    - http: Streamable HTTP (bidirectional, default)
    - sse: Server-Sent Events over HTTP (for web clients)
    - stdio: Standard input/output (for Claude Desktop and CLI tools)

Usage:
    # Run with uv (recommended for development)
    uv run datus-mcp --namespace demo
    uv run datus-mcp --namespace demo --transport stdio

    # Run with uvx (after installing from PyPI)
    uvx --from datus-agent datus-mcp --namespace demo
    uvx --from datus-agent datus-mcp --namespace demo --transport stdio

    # Run with HTTP streamable mode (default)
    python -m datus.mcp_server --namespace demo
    python -m datus.mcp_server --namespace demo --host 0.0.0.0 --port 8000

    # Run with HTTP SSE mode
    python -m datus.mcp_server --namespace demo --transport sse --port 8000

    # Run with stdio (for Claude Desktop)
    python -m datus.mcp_server --namespace demo --transport stdio

    # For Claude Desktop config (claude_desktop_config.json):
    {
        "mcpServers": {
            "datus": {
                "command": "uvx",
                "args": ["--from", "datus-agent", "datus-mcp", "--namespace", "demo", "--transport", "stdio"]
            }
        }
    }

    # Alternative config using python directly:
    {
        "mcpServers": {
            "datus": {
                "command": "python",
                "args": ["-m", "datus.mcp_server", "--namespace", "demo", "--transport", "stdio"]
            }
        }
    }

    # For HTTP clients, connect to:
    # Streamable HTTP (default): http://localhost:8000/mcp
    # SSE: http://localhost:8000/sse
"""

import argparse
import logging
from typing import Any, Dict, List, Literal, Optional, Union

from mcp.server.fastmcp import FastMCP

from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.func_tool.database import db_function_tool_instance
from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)

# Suppress verbose logging for MCP server
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class DatusMCPServer:
    """
    MCP Server that wraps Datus's database and context search tools.

    This server exposes the following tool categories:
    1. Database Tools (DBFuncTool):
       - list_databases, list_schemas, list_tables
       - search_table, describe_table, get_table_ddl
       - read_query

    2. Context Search Tools (ContextSearchTools):
       - list_subject_tree
       - search_metrics, get_metrics
       - search_reference_sql, get_reference_sql
       - search_semantic_objects
       - search_knowledge, get_knowledge
    """

    def __init__(
        self,
        namespace: str,
        sub_agent: Optional[str] = None,
        database_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the Datus MCP Server.

        Args:
            namespace: The database namespace to use (required)
            sub_agent: Optional sub-agent name for scoped context
            database_name: Optional database name override
            config_path: Optional path to agent configuration file
        """
        self.namespace = namespace
        self.sub_agent = sub_agent
        self.database_name = database_name or ""

        # Initialize FastMCP server
        self.mcp = FastMCP(
            name="datus",
            instructions=(
                "Datus is a data engineering agent that provides tools for querying databases, "
                "searching metrics, reference SQL, semantic models, and business knowledge. "
                "Use search_table or list_tables to discover tables, describe_table for schema details, "
                "and read_query to execute SQL queries."
            ),
        )

        # Load agent configuration
        config_kwargs = {"namespace": namespace}
        if config_path:
            config_kwargs["config"] = config_path
        if database_name:
            config_kwargs["database"] = database_name

        self.agent_config = load_agent_config(**config_kwargs)
        # Initialize tool instances
        self._init_db_tools()
        self._init_context_tools()

        # Register all MCP tools
        self._register_tools()

    def _init_db_tools(self):
        """Initialize database function tools."""
        try:
            self.db_tool = db_function_tool_instance(
                self.agent_config,
                database_name=self.database_name,
                sub_agent_name=self.sub_agent,
            )
            self._has_db_tools = True
            logger.info(f"Database tools initialized for namespace: {self.namespace}")
        except Exception as e:
            logger.warning(f"Failed to initialize database tools: {e}")
            self.db_tool = None
            self._has_db_tools = False

    def _init_context_tools(self):
        """Initialize context search tools."""
        try:
            self.context_tool = ContextSearchTools(
                self.agent_config,
                sub_agent_name=self.sub_agent,
            )
            self._has_context_tools = True
            logger.info("Context search tools initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize context search tools: {e}")
            self.context_tool = None
            self._has_context_tools = False

    def close(self):
        """
        Release all resources held by the MCP server.

        This method should be called when the server is no longer needed,
        especially for HTTP transport modes where the server lifecycle
        is managed manually.
        """
        # Close database connection
        if self._has_db_tools and self.db_tool:
            try:
                if hasattr(self.db_tool, "connector") and self.db_tool.connector:
                    self.db_tool.connector.close()
                    logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")

        # Clear tool references
        self.db_tool = None
        self.context_tool = None
        self._has_db_tools = False
        self._has_context_tools = False
        logger.info("MCP server resources released")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resources are released."""
        self.close()
        return False

    def _register_tools(self):
        """Register all available tools with the MCP server."""
        if self._has_db_tools:
            self._register_db_tools()
        if self._has_context_tools:
            self._register_context_tools()

    def _register_db_tools(self):
        """Register database tools with MCP server."""

        # list_databases
        @self.mcp.tool()
        def list_databases(
            catalog: Optional[str] = None,
            include_sys: bool = False,
        ) -> Dict[str, Any]:
            """
            Enumerate databases accessible through the current connection.

            Args:
                catalog: Optional catalog to scope the lookup (dialect dependent).
                include_sys: Set True to include system databases; defaults to False.

            Returns:
                Dictionary with 'success', 'error', and 'result' (list of database names).
            """
            result = self.db_tool.list_databases(
                catalog=catalog,
                include_sys=include_sys,
            )
            return self._format_result(result)

        # list_schemas
        @self.mcp.tool()
        def list_schemas(
            catalog: Optional[str] = None,
            database: Optional[str] = None,
            include_sys: bool = False,
        ) -> Dict[str, Any]:
            """
            List schema names under the supplied catalog/database coordinate.

            Args:
                catalog: Optional catalog filter.
                database: Optional database filter.
                include_sys: Set True to include system schemas; defaults to False.

            Returns:
                Dictionary with 'success', 'error', and 'result' (list of schema names).
            """
            result = self.db_tool.list_schemas(
                catalog=catalog,
                database=database or self.database_name,
                include_sys=include_sys,
            )
            return self._format_result(result)

        # list_tables
        @self.mcp.tool()
        def list_tables(
            catalog: Optional[str] = None,
            database: Optional[str] = None,
            schema_name: Optional[str] = None,
            include_views: bool = True,
        ) -> Dict[str, Any]:
            """
            Return table-like objects (tables, views, materialized views) visible to the connector.

            Args:
                catalog: Optional catalog filter.
                database: Optional database filter.
                schema_name: Optional schema filter.
                include_views: When True (default) also include views and materialized views.

            Returns:
                Dictionary with 'success', 'error', and 'result' (list of table objects).
            """
            result = self.db_tool.list_tables(
                catalog=catalog,
                database=database or self.database_name,
                schema_name=schema_name,
                include_views=include_views,
            )
            return self._format_result(result)

        # search_table
        if self.db_tool.has_schema:

            @self.mcp.tool()
            def search_table(
                query_text: str,
                catalog: Optional[str] = None,
                database_name: Optional[str] = None,
                schema_name: Optional[str] = None,
                top_n: int = 5,
            ) -> Dict[str, Any]:
                """
                Search for tables using semantic similarity over stored schema metadata.

                Use this tool when you need to find tables related to a specific business
                concept or domain, or discover tables containing certain types of data.

                Args:
                    query_text: Description of the table you want (e.g., "daily active users").
                    catalog: Optional catalog filter.
                    database_name: Optional database filter.
                    schema_name: Optional schema filter.
                    top_n: Maximum number of results to return (default 5).

                Returns:
                    Dictionary with metadata and sample_data for matching tables.
                """
                result = self.db_tool.search_table(
                    query_text=query_text,
                    catalog_name=catalog,
                    database_name=database_name or self.database_name,
                    schema_name=schema_name or "",
                    top_n=top_n,
                )
                return self._format_result(result)

        # describe_table
        @self.mcp.tool()
        def describe_table(
            table_name: str,
            catalog: Optional[str] = None,
            database: Optional[str] = None,
            schema_name: Optional[str] = None,
        ) -> Dict[str, Any]:
            """
            Fetch detailed column metadata for a table, enriched with Semantic Model info.

            Use this tool to understand the table schema and business meanings of columns.

            Args:
                table_name: Table identifier to describe.
                catalog: Optional catalog override.
                database: Optional database override.
                schema_name: Optional schema override.

            Returns:
                Dictionary with 'columns' list and optional 'table' metadata from semantic model.
            """
            result = self.db_tool.describe_table(
                table_name=table_name,
                catalog=catalog,
                database=database or self.database_name,
                schema_name=schema_name,
            )
            return self._format_result(result)

        # get_table_ddl
        @self.mcp.tool()
        def get_table_ddl(
            table_name: str,
            catalog: Optional[str] = None,
            database: Optional[str] = None,
            schema_name: Optional[str] = None,
        ) -> Dict[str, Any]:
            """
            Return the DDL definition (CREATE statement) for the requested table.

            Use this when you need a full CREATE statement for semantic modelling
            or schema verification.

            Args:
                table_name: Target table identifier.
                catalog: Optional catalog override.
                database: Optional database override.
                schema_name: Optional schema override.

            Returns:
                Dictionary with DDL definition including identifier, table_type, and definition.
            """
            result = self.db_tool.get_table_ddl(
                table_name=table_name,
                catalog=catalog,
                database=database or self.database_name,
                schema_name=schema_name,
            )
            return self._format_result(result)

        # read_query
        @self.mcp.tool()
        def read_query(sql: str) -> Dict[str, Any]:
            """
            Execute SQL query and return the result rows.

            Args:
                sql: SQL text to run against the database.

            Returns:
                Dictionary with query results or error message.
            """
            result = self.db_tool.read_query(sql=sql)
            return self._format_result(result)

    def _register_context_tools(self):
        """Register context search tools with MCP server."""

        # list_subject_tree
        @self.mcp.tool()
        def list_subject_tree() -> Dict[str, Any]:
            """
            Get the domain-layer taxonomy from subject_tree store.

            Returns a hierarchical structure showing available metrics, reference SQL,
            and knowledge organized by domain and layer.

            Returns:
                Dictionary with hierarchical subject tree structure.
            """
            result = self.context_tool.list_subject_tree()
            return self._format_result(result)

        # search_metrics
        if self.context_tool.has_metrics:

            @self.mcp.tool()
            def search_metrics(
                query_text: str,
                subject_path: Optional[List[str]] = None,
                top_n: int = 5,
            ) -> Dict[str, Any]:
                """
                Search for business metrics and KPIs using natural language queries.

                Args:
                    query_text: Natural language description (e.g., "revenue metrics").
                    subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue']).
                    top_n: Maximum number of results to return (default 5).

                Returns:
                    List of matching metrics with name, description, constraint, and sql_query.
                """
                result = self.context_tool.search_metrics(
                    query_text=query_text,
                    subject_path=subject_path,
                    top_n=top_n,
                )
                return self._format_result(result)

            @self.mcp.tool()
            def get_metrics(
                subject_path: List[str],
                name: str = "",
            ) -> Dict[str, Any]:
                """
                Get detailed information about a specific metric.

                Args:
                    subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1']).
                    name: The name of the metric.

                Returns:
                    Metric details including name, description, constraint, and sql_query.
                """
                result = self.context_tool.get_metrics(
                    subject_path=subject_path,
                    name=name,
                )
                return self._format_result(result)

        # search_reference_sql
        if self.context_tool.has_reference_sql:

            @self.mcp.tool()
            def search_reference_sql(
                query_text: str,
                subject_path: Optional[List[str]] = None,
                top_n: int = 5,
            ) -> Dict[str, Any]:
                """
                Search for reference SQL queries using natural language.

                MUST call `list_subject_tree` first to get available subject paths.

                Args:
                    query_text: Natural language query representing the desired SQL intent.
                    subject_path: Optional subject hierarchy path.
                    top_n: Maximum number of results to return (default 5).

                Returns:
                    List of matching SQL entries with sql, tags, summary, and file_path.
                """
                result = self.context_tool.search_reference_sql(
                    query_text=query_text,
                    subject_path=subject_path,
                    top_n=top_n,
                )
                return self._format_result(result)

            @self.mcp.tool()
            def get_reference_sql(
                subject_path: List[str],
                name: str = "",
            ) -> Dict[str, Any]:
                """
                Get a specific reference SQL query by subject path and name.

                Args:
                    subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue']).
                    name: The name of the reference SQL.

                Returns:
                    SQL entry with sql, tags, summary, and file_path.
                """
                result = self.context_tool.get_reference_sql(
                    subject_path=subject_path,
                    name=name,
                )
                return self._format_result(result)

        # search_semantic_objects
        if self.context_tool.has_semantic_objects:

            @self.mcp.tool()
            def search_semantic_objects(
                query_text: str,
                kinds: Optional[List[str]] = None,
                top_n: int = 5,
            ) -> Dict[str, Any]:
                """
                Search for semantic objects (metrics, columns, tables, entities).

                Args:
                    query_text: Natural language query describing what you're looking for.
                    kinds: List of object kinds to filter by: ["metric", "column", "table", "entity"].
                           If None, searches all kinds.
                    top_n: Maximum number of results to return (default 5).

                Returns:
                    List of matching objects with kind, name, description, and similarity score.
                """
                result = self.context_tool.search_semantic_objects(
                    query_text=query_text,
                    kinds=kinds,
                    top_n=top_n,
                )
                return self._format_result(result)

        # search_knowledge
        if self.context_tool.has_knowledge:

            @self.mcp.tool()
            def search_knowledge(
                query_text: str,
                subject_path: Optional[List[str]] = None,
                top_n: int = 5,
            ) -> Dict[str, Any]:
                """
                Search for external business knowledge using natural language.

                Args:
                    query_text: Natural language query for searching knowledge entries.
                    subject_path: Optional subject hierarchy path.
                    top_n: Maximum number of results to return (default 5).

                Returns:
                    List of matching entries with search_text and explanation.
                """
                result = self.context_tool.search_knowledge(
                    query_text=query_text,
                    subject_path=subject_path,
                    top_n=top_n,
                )
                return self._format_result(result)

            @self.mcp.tool()
            def get_knowledge(
                subject_path: List[str],
                name: str = "",
            ) -> Dict[str, Any]:
                """
                Get specific business knowledge by subject path and name.

                Args:
                    subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue']).
                    name: The name of the knowledge entry.

                Returns:
                    Knowledge entry with search_text and explanation.
                """
                result = self.context_tool.get_knowledge(
                    subject_path=subject_path,
                    name=name,
                )
                return self._format_result(result)

    @staticmethod
    def _format_result(result: Union[FuncToolResult, Any]) -> Dict[str, Any]:
        """Convert FuncToolResult to a dictionary for MCP response."""
        if isinstance(result, FuncToolResult):
            return result.model_dump()
        return {"success": 1, "error": None, "result": result}

    def run(
        self,
        transport: Literal["stdio", "sse", "http"] = "http",
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        """
        Start the MCP server.

        Args:
            transport: Transport type:
                - "http": Streamable HTTP (bidirectional, default)
                - "sse": Server-Sent Events over HTTP
                - "stdio": Standard input/output (for Claude Desktop)
            host: Host to bind for HTTP transports (default: 127.0.0.1)
            port: Port to bind for HTTP transports (default: 8000)
        """
        logger.info(f"Starting Datus MCP Server (namespace={self.namespace}, transport={transport})")

        if transport == "http":
            self._run_http_server(self.mcp.streamable_http_app(), host, port, "/mcp")
        elif transport == "sse":
            self._run_http_server(self.mcp.sse_app(), host, port, "/sse")
        elif transport == "stdio":
            self.mcp.run(transport="stdio")

    def _run_http_server(self, app, host: str, port: int, path: str):
        """Run the ASGI app with uvicorn."""
        import uvicorn

        logger.info(f"HTTP server starting on http://{host}:{port}{path}")
        print(f"\n{'='*60}")
        print("  Datus MCP Server (HTTP Mode)")
        print(f"  Namespace: {self.namespace}")
        print(f"  Endpoint:  http://{host}:{port}{path}")
        print(f"{'='*60}\n")

        uvicorn.run(app, host=host, port=port, log_level="info")

    def get_sse_app(self):
        """
        Get the SSE ASGI application for integration with other frameworks.

        This allows mounting the MCP server in an existing FastAPI/Starlette app:

            from fastapi import FastAPI
            from datus.mcp_server import create_server

            app = FastAPI()
            mcp_server = create_server(namespace="demo")
            app.mount("/sse", mcp_server.get_sse_app())

        Returns:
            ASGI application instance for SSE transport
        """
        return self.mcp.sse_app()

    def get_streamable_http_app(self):
        """
        Get the Streamable HTTP ASGI application for integration with other frameworks.

        This allows mounting the MCP server in an existing FastAPI/Starlette app:

            from fastapi import FastAPI
            from datus.mcp_server import create_server

            app = FastAPI()
            mcp_server = create_server(namespace="demo")
            app.mount("/mcp", mcp_server.get_streamable_http_app())

        Returns:
            ASGI application instance for streamable HTTP transport
        """
        return self.mcp.streamable_http_app()


def create_server(
    namespace: str,
    sub_agent: Optional[str] = None,
    database_name: Optional[str] = None,
    config_path: Optional[str] = None,
) -> DatusMCPServer:
    """
    Factory function to create a DatusMCPServer instance.

    Args:
        namespace: The database namespace to use (required)
        sub_agent: Optional sub-agent name for scoped context
        database_name: Optional database name override
        config_path: Optional path to agent configuration file

    Returns:
        Configured DatusMCPServer instance
    """
    return DatusMCPServer(
        namespace=namespace,
        sub_agent=sub_agent,
        database_name=database_name,
        config_path=config_path,
    )


def main():
    """Main entry point for the MCP server CLI."""
    parser = argparse.ArgumentParser(
        description="Datus MCP Server - Expose Datus tools via Model Context Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with uv (recommended for development)
    uv run datus-mcp --namespace demo
    uv run datus-mcp --namespace demo --transport stdio

    # Run with uvx (after installing from PyPI)
    uvx --from datus-agent datus-mcp --namespace demo
    uvx --from datus-agent datus-mcp --namespace demo --transport stdio

    # Run with HTTP streamable mode (default)
    python -m datus.mcp_server --namespace demo
    python -m datus.mcp_server --namespace demo --host 0.0.0.0 --port 8000

    # Run with HTTP SSE mode
    python -m datus.mcp_server --namespace demo --transport sse --port 8000

    # Run with stdio (for Claude Desktop)
    python -m datus.mcp_server --namespace demo --transport stdio

    # Use custom config file
    python -m datus.mcp_server --namespace demo --config /path/to/agent.yml

Claude Desktop Configuration (claude_desktop_config.json):

    {
        "mcpServers": {
            "datus": {
                "command": "uvx",
                "args": ["--from", "datus-agent", "datus-mcp", "--namespace", "demo", "--transport", "stdio"]
            }
        }
    }

HTTP Client Usage:
    # Streamable HTTP (default): http://localhost:8000/mcp
    # SSE transport: http://localhost:8000/sse
        """,
    )

    parser.add_argument(
        "--namespace",
        "-n",
        required=True,
        help="Database namespace to use (required)",
    )
    parser.add_argument(
        "--sub-agent",
        "-s",
        default=None,
        help="Sub-agent name for scoped context",
    )
    parser.add_argument(
        "--database",
        "-d",
        default=None,
        help="Database name override",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to agent configuration file",
    )
    parser.add_argument(
        "--transport",
        "-t",
        choices=["http", "sse", "stdio"],
        default="http",
        help="Transport type: http (default, HTTP bidirectional), sse (HTTP SSE), stdio (for Claude Desktop)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind for HTTP transports (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Port to bind for HTTP transports (default: 8000)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    configure_logging(debug=args.debug)

    # Create and run server
    server = create_server(
        namespace=args.namespace,
        sub_agent=args.sub_agent,
        database_name=args.database,
        config_path=args.config,
    )
    server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
