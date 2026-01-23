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
    # === Dynamic Mode (Multi-namespace HTTP Server) ===
    # Run dynamic server that supports all namespaces via URL path
    python -m datus.mcp_server --dynamic
    python -m datus.mcp_server --dynamic --host 0.0.0.0 --port 8000

    # Connect to specific namespace:
    # http://localhost:8000/mcp/{namespace}
    # http://localhost:8000/mcp/{namespace}?subagent={subagent_name}

    # === Static Mode (Single-namespace) ===
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
    # Static mode: http://localhost:8000/mcp
    # Dynamic mode: http://localhost:8000/mcp/{namespace}
    # SSE: http://localhost:8000/sse (static mode only)
"""

import argparse
import asyncio
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Union

from mcp.server.fastmcp import FastMCP

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import configuration_manager, load_agent_config
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.func_tool.database import DBFuncTool, db_function_tool_instance, db_function_tool_instance_multi
from datus.utils.loggings import configure_logging, get_logger

# Re-export for external use
__all__ = [
    "DatusMCPServer",
    "ToolContext",
    "ToolContextManager",
    "LightweightDynamicMCPServer",
    "create_server",
    "create_dynamic_app",
    "run_dynamic_server",
]

logger = get_logger(__name__)

# Suppress verbose logging for MCP server
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ============================================================================
# Lightweight Dynamic Mode Components
# ============================================================================


@dataclass
class ToolContext:
    """
    Container for namespace-scoped tool instances.

    This class holds the AgentConfig and tool instances for a specific
    namespace/subagent combination, enabling tool reuse across requests.
    """

    namespace: str
    subagent: Optional[str]
    agent_config: AgentConfig
    db_tool: Optional[DBFuncTool]
    context_tool: Optional[ContextSearchTools]

    @property
    def has_db_tools(self) -> bool:
        return self.db_tool is not None

    @property
    def has_context_tools(self) -> bool:
        return self.context_tool is not None

    def close(self):
        """Release resources held by this context."""
        if self.db_tool:
            try:
                if hasattr(self.db_tool, "connector") and self.db_tool.connector:
                    self.db_tool.connector.close()
            except Exception as e:
                logger.warning(f"Error closing db_tool connector: {e}")
        self.db_tool = None
        self.context_tool = None


class ToolContextManager:
    """
    Manages ToolContext instances for multiple namespaces with LRU caching.

    This manager:
    1. Creates AgentConfig copies with specific namespace settings
    2. Initializes DBFuncTool and ContextSearchTools lazily
    3. Caches contexts with LRU eviction policy
    4. Provides validation of namespaces and subagents
    5. Properly closes evicted contexts to release resources
    """

    DEFAULT_MAX_SIZE = 16

    def __init__(self, config_path: Optional[str] = None, max_size: Optional[int] = None):
        """
        Initialize the ToolContextManager.

        Args:
            config_path: Optional path to agent configuration file
            max_size: Maximum number of contexts to cache. When exceeded,
                     the least recently used context is evicted. Default is 16.
                     Set to 0 or None for unlimited cache.
        """
        self.config_path = config_path
        self._max_size = max_size if max_size is not None else self.DEFAULT_MAX_SIZE
        self._contexts: OrderedDict[str, ToolContext] = OrderedDict()
        self._lock = asyncio.Lock()

        # Load base configuration
        self._config_manager = configuration_manager(config_path=config_path or "")
        self._available_namespaces: Set[str] = set(self._config_manager.get("namespace", {}).keys())
        self._available_subagents: Set[str] = set(self._config_manager.get("agentic_nodes", {}).keys())

        logger.info(
            f"ToolContextManager initialized with namespaces: {list(self._available_namespaces)}, "
            f"max_size: {self._max_size or 'unlimited'}"
        )

    @property
    def available_namespaces(self) -> List[str]:
        return list(self._available_namespaces)

    @property
    def available_subagents(self) -> List[str]:
        return list(self._available_subagents)

    def validate_namespace(self, namespace: str) -> bool:
        return namespace in self._available_namespaces

    def validate_subagent(self, subagent: str) -> bool:
        return subagent in self._available_subagents

    def _get_cache_key(
        self,
        namespace: str,
        subagent: Optional[str] = None,
    ) -> str:
        return f"{namespace}:{subagent or ''}"

    async def get_or_create_context(
        self,
        namespace: str,
        subagent: Optional[str] = None,
    ) -> ToolContext:
        """
        Get or create a ToolContext for the given namespace/subagent.

        Uses LRU caching: recently accessed contexts are kept, oldest are evicted
        when cache exceeds max_size.

        Args:
            namespace: The database namespace (required)
            subagent: Optional sub-agent name

        Returns:
            ToolContext with initialized tools
        """
        cache_key = self._get_cache_key(namespace, subagent)

        async with self._lock:
            if cache_key in self._contexts:
                # Cache hit: move to end (most recently used)
                self._contexts.move_to_end(cache_key)
                return self._contexts[cache_key]

            # Cache miss: create new context
            logger.info(f"Creating ToolContext for {cache_key}")
            context = self._create_context(namespace, subagent)

            # Evict oldest if cache is full
            if self._max_size > 0 and len(self._contexts) >= self._max_size:
                evicted_key, evicted_context = self._contexts.popitem(last=False)
                logger.info(f"LRU evicting ToolContext: {evicted_key}")
                try:
                    evicted_context.close()
                except Exception as e:
                    logger.warning(f"Error closing evicted context {evicted_key}: {e}")

            self._contexts[cache_key] = context
            return context

    def _create_context(
        self,
        namespace: str,
        subagent: Optional[str] = None,
    ) -> ToolContext:
        """Create a new ToolContext with initialized tools."""
        # Load agent config with namespace
        config_kwargs = {"namespace": namespace}
        if self.config_path:
            config_kwargs["config"] = self.config_path

        agent_config = load_agent_config(**config_kwargs)

        # Initialize database tools (use multi-connector mode for dynamic server)
        db_tool = None
        try:
            db_tool = db_function_tool_instance_multi(
                agent_config,
                sub_agent_name=subagent,
            )
            logger.info(f"Database tools initialized for namespace: {namespace} (multi-connector mode)")
        except Exception as e:
            logger.warning(f"Failed to initialize database tools for {namespace}: {e}")

        # Initialize context search tools
        context_tool = None
        try:
            context_tool = ContextSearchTools(
                agent_config,
                sub_agent_name=subagent,
            )
            logger.info(f"Context search tools initialized for namespace: {namespace}")
        except Exception as e:
            logger.warning(f"Failed to initialize context tools for {namespace}: {e}")

        return ToolContext(
            namespace=namespace,
            subagent=subagent,
            agent_config=agent_config,
            db_tool=db_tool,
            context_tool=context_tool,
        )

    def close_all(self):
        """Close all managed contexts."""
        for cache_key, context in list(self._contexts.items()):
            try:
                context.close()
                logger.info(f"Closed ToolContext for {cache_key}")
            except Exception as e:
                logger.warning(f"Error closing context {cache_key}: {e}")
        self._contexts.clear()


# Context variable for current request's tool context
_current_tool_context: ContextVar[Optional[ToolContext]] = ContextVar("current_tool_context", default=None)


class LightweightDynamicMCPServer:
    """
    Lightweight MCP server with single FastMCP instance and dynamic tool contexts.

    This server uses a single FastMCP instance shared across all namespaces,
    with tool execution dynamically routed to the appropriate ToolContext
    based on the current request.

    Key benefits:
    - Single FastMCP overhead regardless of namespace count
    - Shared session manager for all requests
    - Per-namespace tool caching (AgentConfig, DBFuncTool, ContextSearchTools)

    Usage:
        server = LightweightDynamicMCPServer(config_path="conf/agent.yml")
        app = server.create_asgi_app()
        # Run with uvicorn
    """

    def __init__(self, config_path: Optional[str] = None, max_cache_size: Optional[int] = 64):
        """
        Initialize the lightweight dynamic MCP server.

        Args:
            config_path: Optional path to agent configuration file
            max_cache_size: Maximum number of ToolContexts to cache (LRU eviction).
                           Default is 16. Set to 0 for unlimited.
        """
        self.config_path = config_path
        self._context_manager = ToolContextManager(config_path, max_size=max_cache_size)

        # Single shared FastMCP instance
        self.mcp = FastMCP(
            name="datus",
            instructions=(
                "Datus is a data engineering agent that provides tools for querying databases, "
                "searching metrics, reference SQL, semantic models, and business knowledge. "
                "Use search_table or list_tables to discover tables, describe_table for schema details, "
                "and read_query to execute SQL queries."
            ),
            stateless_http=True,
        )

        # Register tools once (they use _current_tool_context to get the right instance)
        self._register_tools()

        # Session manager lifecycle
        self._session_manager = None
        self._task_group = None
        self._started = False

    @property
    def available_namespaces(self) -> List[str]:
        return self._context_manager.available_namespaces

    @property
    def available_subagents(self) -> List[str]:
        return self._context_manager.available_subagents

    def validate_namespace(self, namespace: str) -> bool:
        return self._context_manager.validate_namespace(namespace)

    def validate_subagent(self, subagent: str) -> bool:
        return self._context_manager.validate_subagent(subagent)

    @staticmethod
    def _get_context() -> ToolContext:
        """Get current request's tool context."""
        ctx = _current_tool_context.get()
        if ctx is None:
            raise RuntimeError("No tool context set for current request")
        return ctx

    @staticmethod
    def _format_result(result: Union[FuncToolResult, Any]) -> Dict[str, Any]:
        """Convert FuncToolResult to a dictionary for MCP response."""
        if isinstance(result, FuncToolResult):
            return result.model_dump()
        return {"success": 1, "error": None, "result": result}

    def _register_tools(self):
        """Register all MCP tools that use dynamic context."""
        self._register_db_tools()
        self._register_context_tools()

    def _register_db_tools(self):
        """Register database tools with dynamic context lookup."""

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
            ctx = self._get_context()
            if not ctx.has_db_tools:
                return {"success": 0, "error": "Database tools not available", "result": None}
            result = ctx.db_tool.list_databases(catalog=catalog, include_sys=include_sys)
            return self._format_result(result)

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
            ctx = self._get_context()
            if not ctx.has_db_tools:
                return {"success": 0, "error": "Database tools not available", "result": None}
            result = ctx.db_tool.list_schemas(
                catalog=catalog,
                database=database,
                include_sys=include_sys,
            )
            return self._format_result(result)

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
            ctx = self._get_context()
            if not ctx.has_db_tools:
                return {"success": 0, "error": "Database tools not available", "result": None}
            result = ctx.db_tool.list_tables(
                catalog=catalog,
                database=database,
                schema_name=schema_name,
                include_views=include_views,
            )
            return self._format_result(result)

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
            ctx = self._get_context()
            if not ctx.has_db_tools:
                return {"success": 0, "error": "Database tools not available", "result": None}
            if not ctx.db_tool.has_schema:
                return {"success": 0, "error": "Schema search not available", "result": None}
            result = ctx.db_tool.search_table(
                query_text=query_text,
                catalog_name=catalog,
                database_name=database_name,
                schema_name=schema_name or "",
                top_n=top_n,
            )
            return self._format_result(result)

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
            ctx = self._get_context()
            if not ctx.has_db_tools:
                return {"success": 0, "error": "Database tools not available", "result": None}
            result = ctx.db_tool.describe_table(
                table_name=table_name,
                catalog=catalog,
                database=database,
                schema_name=schema_name,
            )
            return self._format_result(result)

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
            ctx = self._get_context()
            if not ctx.has_db_tools:
                return {"success": 0, "error": "Database tools not available", "result": None}
            result = ctx.db_tool.get_table_ddl(
                table_name=table_name,
                catalog=catalog,
                database=database,
                schema_name=schema_name,
            )
            return self._format_result(result)

        @self.mcp.tool()
        def read_query(sql: str) -> Dict[str, Any]:
            """
            Execute SQL query and return the result rows.

            Args:
                sql: SQL text to run against the database.

            Returns:
                Dictionary with query results or error message.
            """
            ctx = self._get_context()
            if not ctx.has_db_tools:
                return {"success": 0, "error": "Database tools not available", "result": None}
            result = ctx.db_tool.read_query(sql=sql)
            return self._format_result(result)

    def _register_context_tools(self):
        """Register context search tools with dynamic context lookup."""

        @self.mcp.tool()
        def list_subject_tree() -> Dict[str, Any]:
            """
            Get the domain-layer taxonomy from subject_tree store.

            Returns a hierarchical structure showing available metrics, reference SQL,
            and knowledge organized by domain and layer.

            Returns:
                Dictionary with hierarchical subject tree structure.
            """
            ctx = self._get_context()
            if not ctx.has_context_tools:
                return {"success": 0, "error": "Context tools not available", "result": None}
            result = ctx.context_tool.list_subject_tree()
            return self._format_result(result)

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
            ctx = self._get_context()
            if not ctx.has_context_tools:
                return {"success": 0, "error": "Context tools not available", "result": None}
            if not ctx.context_tool.has_metrics:
                return {"success": 0, "error": "Metrics search not available", "result": None}
            result = ctx.context_tool.search_metrics(
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
            ctx = self._get_context()
            if not ctx.has_context_tools:
                return {"success": 0, "error": "Context tools not available", "result": None}
            if not ctx.context_tool.has_metrics:
                return {"success": 0, "error": "Metrics not available", "result": None}
            result = ctx.context_tool.get_metrics(
                subject_path=subject_path,
                name=name,
            )
            return self._format_result(result)

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
            ctx = self._get_context()
            if not ctx.has_context_tools:
                return {"success": 0, "error": "Context tools not available", "result": None}
            if not ctx.context_tool.has_reference_sql:
                return {"success": 0, "error": "Reference SQL search not available", "result": None}
            result = ctx.context_tool.search_reference_sql(
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
            ctx = self._get_context()
            if not ctx.has_context_tools:
                return {"success": 0, "error": "Context tools not available", "result": None}
            if not ctx.context_tool.has_reference_sql:
                return {"success": 0, "error": "Reference SQL not available", "result": None}
            result = ctx.context_tool.get_reference_sql(
                subject_path=subject_path,
                name=name,
            )
            return self._format_result(result)

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
            ctx = self._get_context()
            if not ctx.has_context_tools:
                return {"success": 0, "error": "Context tools not available", "result": None}
            if not ctx.context_tool.has_semantic_objects:
                return {"success": 0, "error": "Semantic objects search not available", "result": None}
            result = ctx.context_tool.search_semantic_objects(
                query_text=query_text,
                kinds=kinds,
                top_n=top_n,
            )
            return self._format_result(result)

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
            ctx = self._get_context()
            if not ctx.has_context_tools:
                return {"success": 0, "error": "Context tools not available", "result": None}
            if not ctx.context_tool.has_knowledge:
                return {"success": 0, "error": "Knowledge search not available", "result": None}
            result = ctx.context_tool.search_knowledge(
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
            ctx = self._get_context()
            if not ctx.has_context_tools:
                return {"success": 0, "error": "Context tools not available", "result": None}
            if not ctx.context_tool.has_knowledge:
                return {"success": 0, "error": "Knowledge not available", "result": None}
            result = ctx.context_tool.get_knowledge(
                subject_path=subject_path,
                name=name,
            )
            return self._format_result(result)

    @asynccontextmanager
    async def lifespan_context(self):
        """
        Context manager for server lifespan.
        Starts session manager in background task group.
        """
        import anyio

        # Get session manager from MCP
        _ = self.mcp.streamable_http_app()
        self._session_manager = self.mcp.session_manager

        async with anyio.create_task_group() as tg:
            self._task_group = tg

            # Start session manager
            started_event = anyio.Event()

            async def run_session_manager():
                async with self._session_manager.run():
                    started_event.set()
                    try:
                        await anyio.sleep_forever()
                    except anyio.get_cancelled_exc_class():
                        pass

            tg.start_soon(run_session_manager)
            await started_event.wait()
            self._started = True

            logger.info("LightweightDynamicMCPServer started")
            try:
                yield
            finally:
                logger.info("LightweightDynamicMCPServer shutting down")
                tg.cancel_scope.cancel()
                self._started = False
                self._context_manager.close_all()

    async def handle_request(
        self,
        scope: Dict,
        receive,
        send,
        namespace: str,
        subagent: Optional[str] = None,
    ):
        """
        Handle an MCP request with the specified context.

        Args:
            scope: ASGI scope
            receive: ASGI receive callable
            send: ASGI send callable
            namespace: Target namespace
            subagent: Optional subagent name
        """
        if not self._started:
            raise RuntimeError("Server not started. Use lifespan_context first.")

        # Get or create tool context
        context = await self._context_manager.get_or_create_context(
            namespace=namespace,
            subagent=subagent,
        )

        # Set context for this request
        token = _current_tool_context.set(context)
        try:
            # Modify scope path for MCP
            new_scope = dict(scope)
            new_scope["path"] = "/mcp"
            await self._session_manager.handle_request(new_scope, receive, send)
        finally:
            _current_tool_context.reset(token)

    def create_asgi_app(self):
        """
        Create a Starlette ASGI application with dynamic routing.

        Returns:
            Starlette application instance
        """
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route

        server = self

        class DynamicRouter:
            """ASGI router that parses namespace from path and routes to server."""

            @staticmethod
            def _parse_request(scope: Dict) -> tuple:
                path = scope.get("path", "")
                query_string = scope.get("query_string", b"").decode("utf-8")

                parts = path.strip("/").split("/")
                if not parts or not parts[0]:
                    return None, None

                namespace = parts[-1]

                subagent = None
                if query_string:
                    from urllib.parse import parse_qs

                    params = parse_qs(query_string)
                    subagent = params.get("subagent", [None])[0]

                return namespace, subagent

            async def __call__(self, scope, receive, send):
                if scope["type"] != "http":
                    return

                namespace, subagent = self._parse_request(scope)

                if not namespace:
                    await self._send_error(send, 400, "Bad Request: Missing namespace in path")
                    return

                if not server.validate_namespace(namespace):
                    await self._send_error(
                        send,
                        404,
                        f"Not Found: Namespace '{namespace}' not available. "
                        f"Available: {server.available_namespaces}",
                    )
                    return

                if subagent and not server.validate_subagent(subagent):
                    await self._send_error(
                        send,
                        404,
                        f"Not Found: Subagent '{subagent}' not available. " f"Available: {server.available_subagents}",
                    )
                    return

                try:
                    await server.handle_request(scope, receive, send, namespace, subagent)
                except Exception as e:
                    logger.exception(f"Error handling request for namespace={namespace}")
                    await self._send_error(send, 500, f"Internal Server Error: {str(e)}")

            async def _send_error(self, send, status_code: int, message: str):
                import json

                body = json.dumps({"error": message}).encode("utf-8")
                await send(
                    {
                        "type": "http.response.start",
                        "status": status_code,
                        "headers": [[b"content-type", b"application/json"]],
                    }
                )
                await send({"type": "http.response.body", "body": body})

        @asynccontextmanager
        async def lifespan(_app):
            logger.info("Starting Datus MCP Server (Lightweight Dynamic Mode)")
            logger.info(f"Available namespaces: {server.available_namespaces}")
            if server.available_subagents:
                logger.info(f"Available subagents: {server.available_subagents}")

            async with server.lifespan_context():
                yield

            logger.info("Datus MCP Server stopped")

        async def root(request):
            return JSONResponse(
                {
                    "service": "Datus MCP Server",
                    "mode": "lightweight-dynamic",
                    "available_namespaces": server.available_namespaces,
                    "available_subagents": server.available_subagents,
                    "endpoints": {
                        "mcp": "/mcp/{namespace}",
                        "mcp_with_subagent": "/mcp/{namespace}?subagent={subagent_name}",
                        "health": "/health",
                    },
                }
            )

        async def health(request):
            return JSONResponse(
                {
                    "status": "healthy",
                    "cached_contexts": len(server._context_manager._contexts),
                }
            )

        return Starlette(
            debug=False,
            routes=[
                Route("/", root, methods=["GET"]),
                Route("/health", health, methods=["GET"]),
                Mount("/mcp", app=DynamicRouter()),
            ],
            lifespan=lifespan,
        )


# ============================================================================
# Static Mode Server (Original Implementation)
# ============================================================================


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
        stateless_http: bool = False,
    ):
        """
        Initialize the Datus MCP Server.

        Args:
            namespace: The database namespace to use (required)
            sub_agent: Optional sub-agent name for scoped context
            database_name: Optional database name override
            config_path: Optional path to agent configuration file
            stateless_http: If True, creates a new transport per request (for dynamic routing)
        """
        self.namespace = namespace
        self.sub_agent = sub_agent
        self.database_name = database_name or ""
        self._stateless_http = stateless_http

        # Initialize FastMCP server
        self.mcp = FastMCP(
            name="datus",
            instructions=(
                "Datus is a data engineering agent that provides tools for querying databases, "
                "searching metrics, reference SQL, semantic models, and business knowledge. "
                "Use search_table or list_tables to discover tables, describe_table for schema details, "
                "and read_query to execute SQL queries."
            ),
            stateless_http=stateless_http,  # Enable stateless mode for dynamic routing
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


class DatusMCPServerManager:
    """
    Manages multiple DatusMCPServer instances for dynamic namespace/subagent support.

    This manager provides:
    1. Lazy initialization of MCP servers per namespace/subagent combination
    2. Instance caching to avoid repeated initialization
    3. Validation of namespaces and subagents against configuration
    4. Graceful cleanup of all instances

    Usage:
        manager = DatusMCPServerManager(config_path="conf/agent.yml")
        server = await manager.get_or_create_server("demo", subagent="sales")
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the manager.

        Args:
            config_path: Optional path to agent configuration file
        """
        self.config_path = config_path
        self._servers: Dict[str, DatusMCPServer] = {}
        self._lock = asyncio.Lock()

        # Load configuration to get available namespaces and subagents
        self._config_manager = configuration_manager(config_path=config_path or "")
        self._available_namespaces: Set[str] = set(self._config_manager.get("namespace", {}).keys())
        self._available_subagents: Set[str] = set(self._config_manager.get("agentic_nodes", {}).keys())

        logger.info(f"MCP Server Manager initialized with namespaces: {self._available_namespaces}")
        if self._available_subagents:
            logger.info(f"Available subagents: {self._available_subagents}")

    @property
    def available_namespaces(self) -> List[str]:
        """Get list of available namespaces from configuration."""
        return list(self._available_namespaces)

    @property
    def available_subagents(self) -> List[str]:
        """Get list of available subagents from configuration."""
        return list(self._available_subagents)

    def _get_cache_key(self, namespace: str, subagent: Optional[str] = None) -> str:
        """Generate cache key for namespace/subagent combination."""
        return f"{namespace}:{subagent or ''}"

    def validate_namespace(self, namespace: str) -> bool:
        """Check if namespace exists in configuration."""
        return namespace in self._available_namespaces

    def validate_subagent(self, subagent: str) -> bool:
        """Check if subagent exists in configuration."""
        return subagent in self._available_subagents

    async def get_or_create_server(
        self,
        namespace: str,
        subagent: Optional[str] = None,
    ) -> DatusMCPServer:
        """
        Get or create a DatusMCPServer instance for the given namespace/subagent.

        Args:
            namespace: The database namespace (required, must exist in config)
            subagent: Optional sub-agent name for scoped context

        Returns:
            DatusMCPServer instance

        Raises:
            ValueError: If namespace or subagent is not found in configuration
        """
        if not self.validate_namespace(namespace):
            raise ValueError(f"Namespace '{namespace}' not found. Available: {self.available_namespaces}")

        if subagent and not self.validate_subagent(subagent):
            raise ValueError(f"Subagent '{subagent}' not found. Available: {self.available_subagents}")

        cache_key = self._get_cache_key(namespace, subagent)

        async with self._lock:
            if cache_key not in self._servers:
                logger.info(f"Creating new MCP server for {cache_key}")
                server = DatusMCPServer(
                    namespace=namespace,
                    sub_agent=subagent,
                    config_path=self.config_path,
                    stateless_http=True,  # Enable stateless mode for dynamic routing
                )
                self._servers[cache_key] = server
            return self._servers[cache_key]

    def get_server(self, namespace: str, subagent: Optional[str] = None) -> Optional[DatusMCPServer]:
        """
        Get existing server instance without creating a new one.

        Args:
            namespace: The database namespace
            subagent: Optional sub-agent name

        Returns:
            DatusMCPServer instance if exists, None otherwise
        """
        cache_key = self._get_cache_key(namespace, subagent)
        return self._servers.get(cache_key)

    def close_server(self, namespace: str, subagent: Optional[str] = None) -> bool:
        """
        Close and remove a specific server instance.

        Args:
            namespace: The database namespace
            subagent: Optional sub-agent name

        Returns:
            True if server was found and closed, False otherwise
        """
        cache_key = self._get_cache_key(namespace, subagent)
        if cache_key in self._servers:
            self._servers[cache_key].close()
            del self._servers[cache_key]
            logger.info(f"Closed MCP server for {cache_key}")
            return True
        return False

    def close_all(self):
        """Close all managed server instances."""
        for cache_key, server in list(self._servers.items()):
            try:
                server.close()
                logger.info(f"Closed MCP server for {cache_key}")
            except Exception as e:
                logger.warning(f"Error closing server {cache_key}: {e}")
        self._servers.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures all resources are released."""
        self.close_all()
        return False


class DynamicMCPRouter:
    """
    ASGI application that dynamically routes MCP requests to namespace-specific servers.

    This router parses the URL path to extract namespace and query parameters for subagent,
    then forwards the request to the appropriate MCP server instance.

    URL format: /mcp/{namespace}?subagent={subagent_name}

    Note: Uses stateless_http=True mode for simplicity in dynamic routing, which means
    each request creates a fresh transport without session persistence.
    """

    def __init__(self, manager: DatusMCPServerManager):
        self.manager = manager
        self._session_managers: Dict[str, Any] = {}
        self._session_contexts: Dict[str, Any] = {}  # Store context managers
        self._lock = asyncio.Lock()
        self._task_group = None

    @asynccontextmanager
    async def lifespan_context(self):
        """
        Context manager for router lifespan.
        Manages task group for session managers.
        """
        import anyio

        async with anyio.create_task_group() as tg:
            self._task_group = tg
            logger.info("DynamicMCPRouter started with task group")
            try:
                yield
            finally:
                logger.info("DynamicMCPRouter shutting down")
                # Cancel all tasks
                tg.cancel_scope.cancel()
                self._task_group = None
                self._session_managers.clear()
                self._session_contexts.clear()

    async def _get_session_manager(
        self,
        namespace: str,
        subagent: Optional[str] = None,
    ):
        """Get or create the session manager for a namespace/subagent combination."""
        import anyio

        cache_key = f"{namespace}:{subagent or ''}"

        async with self._lock:
            if cache_key not in self._session_managers:
                if self._task_group is None:
                    raise RuntimeError("Router not started. Use lifespan_context first.")

                server = await self.manager.get_or_create_server(
                    namespace=namespace,
                    subagent=subagent,
                )

                # Get the session manager (creates it if needed)
                _ = server.mcp.streamable_http_app()
                session_manager = server.mcp.session_manager

                # Start the session manager in the task group
                started_event = anyio.Event()

                async def run_session_manager():
                    async with session_manager.run():
                        started_event.set()
                        # Keep running until cancelled
                        try:
                            await anyio.sleep_forever()
                        except anyio.get_cancelled_exc_class():
                            pass

                self._task_group.start_soon(run_session_manager)

                # Wait for the session manager to start
                await started_event.wait()

                self._session_managers[cache_key] = session_manager
                logger.info(f"Started session manager for {cache_key}")

            return self._session_managers[cache_key]

    @staticmethod
    def _parse_request(scope: Dict) -> tuple:
        """Parse namespace and query parameters from the request.

        Note: When mounted via Mount("/mcp", app=mcp_router), the path received
        here is already stripped of the "/mcp" prefix. So for a request to
        "/mcp/bird_sqlite", this method receives "/bird_sqlite".
        """
        path = scope.get("path", "")
        query_string = scope.get("query_string", b"").decode("utf-8")

        # Extract namespace from path: /{namespace}
        # Path format: /namespace or /namespace/
        parts = path.strip("/").split("/")
        if not parts or not parts[0]:
            return None, None

        namespace = parts[-1]

        # Parse query parameters
        subagent = None
        if query_string:
            from urllib.parse import parse_qs

            params = parse_qs(query_string)
            subagent = params.get("subagent", [None])[0]

        return namespace, subagent

    async def __call__(self, scope, receive, send):
        """ASGI application entry point."""
        if scope["type"] != "http":
            return

        namespace, subagent = self._parse_request(scope)

        if not namespace:
            # Return 400 Bad Request
            await self._send_error(send, 400, "Bad Request: Missing namespace in path")
            return

        # Validate namespace
        if not self.manager.validate_namespace(namespace):
            await self._send_error(
                send,
                404,
                f"Not Found: Namespace '{namespace}' not available. " f"Available: {self.manager.available_namespaces}",
            )
            return

        # Validate subagent if provided
        if subagent and not self.manager.validate_subagent(subagent):
            await self._send_error(
                send,
                404,
                f"Not Found: Subagent '{subagent}' not available. " f"Available: {self.manager.available_subagents}",
            )
            return

        try:
            # Get the session manager for this namespace
            session_manager = await self._get_session_manager(namespace, subagent)

            # Modify scope to have the path that MCP expects
            new_scope = dict(scope)
            new_scope["path"] = "/mcp"

            # Use session manager to handle request directly
            # This bypasses the Starlette app and its lifespan requirement
            await session_manager.handle_request(new_scope, receive, send)

        except Exception as e:
            logger.exception(f"Error handling MCP request for namespace={namespace}")
            await self._send_error(send, 500, f"Internal Server Error: {str(e)}")

    async def _send_error(self, send, status_code: int, message: str):
        """Send an error response."""
        import json

        body = json.dumps({"error": message}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [[b"content-type", b"application/json"]],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )


def create_dynamic_app(
    config_path: Optional[str] = None,
    lightweight: bool = True,
    max_cache_size: Optional[int] = None,
):
    """
    Create a Starlette application with dynamic namespace/subagent routing.

    This function creates an ASGI app that:
    1. Loads all available namespaces from configuration at startup
    2. Routes requests to /mcp/{namespace} to the appropriate MCP server
    3. Supports optional subagent parameter via query string
    4. Properly handles MCP's streaming protocol

    Args:
        config_path: Optional path to agent configuration file
        lightweight: If True (default), use LightweightDynamicMCPServer with single
                    FastMCP instance and cached tool contexts. If False, use legacy
                    mode with separate DatusMCPServer per namespace.
        max_cache_size: Maximum number of ToolContexts to cache (LRU eviction).
                       Default is 16. Set to 0 for unlimited. Only applies to lightweight mode.

    Returns:
        Starlette application instance

    Example:
        app = create_dynamic_app()
        # Run with: uvicorn datus.mcp_server:app --host 0.0.0.0 --port 8000

        # Client connects to:
        # http://localhost:8000/mcp/demo
        # http://localhost:8000/mcp/demo?subagent=sales
    """
    if lightweight:
        # Use new lightweight implementation with single FastMCP instance
        server = LightweightDynamicMCPServer(config_path=config_path, max_cache_size=max_cache_size)
        return server.create_asgi_app()

    # Legacy mode: separate DatusMCPServer per namespace
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    # Create manager instance
    manager = DatusMCPServerManager(config_path=config_path)

    # Create dynamic MCP router
    mcp_router = DynamicMCPRouter(manager)

    @asynccontextmanager
    async def lifespan(_app):
        """Application lifespan handler."""
        logger.info("Starting Datus MCP Server (Dynamic Mode - Legacy)")
        logger.info(f"Available namespaces: {manager.available_namespaces}")
        if manager.available_subagents:
            logger.info(f"Available subagents: {manager.available_subagents}")

        # Start the router with its task group for session managers
        async with mcp_router.lifespan_context():
            yield

        logger.info("Shutting down Datus MCP Server")
        manager.close_all()

    # Define info endpoints
    async def root(request):
        """Root endpoint with server info."""
        return JSONResponse(
            {
                "service": "Datus MCP Server",
                "mode": "dynamic-legacy",
                "available_namespaces": manager.available_namespaces,
                "available_subagents": manager.available_subagents,
                "endpoints": {
                    "mcp": "/mcp/{namespace}",
                    "mcp_with_subagent": "/mcp/{namespace}?subagent={subagent_name}",
                    "health": "/health",
                    "namespaces": "/namespaces",
                },
            }
        )

    async def health(request):
        """Health check endpoint."""
        return JSONResponse(
            {
                "status": "healthy",
                "active_servers": len(manager._servers),
            }
        )

    async def list_namespaces(request):
        """List available namespaces and subagents."""
        return JSONResponse(
            {
                "namespaces": manager.available_namespaces,
                "subagents": manager.available_subagents,
            }
        )

    # Create Starlette app with routes
    app = Starlette(
        debug=False,
        routes=[
            Route("/", root, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
            Route("/namespaces", list_namespaces, methods=["GET"]),
            # Mount the dynamic MCP router for all /mcp/* paths
            Mount("/mcp", app=mcp_router),
        ],
        lifespan=lifespan,
    )

    return app


def run_dynamic_server(
    config_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False,
    lightweight: bool = True,
    max_cache_size: Optional[int] = 64,
):
    """
    Run the dynamic MCP server with uvicorn.

    Args:
        config_path: Optional path to agent configuration file
        host: Host to bind (default: 0.0.0.0)
        port: Port to bind (default: 8000)
        debug: Enable debug mode
        lightweight: If True (default), use lightweight mode with single FastMCP instance
        max_cache_size: Maximum ToolContext cache size (LRU). Default 64, 0 for unlimited.
    """
    import uvicorn

    app = create_dynamic_app(
        config_path=config_path,
        lightweight=lightweight,
        max_cache_size=max_cache_size,
    )
    mode_name = "Lightweight" if lightweight else "Legacy"
    cache_info = f", cache_size={max_cache_size or 64}" if lightweight else ""

    print(f"\n{'='*60}")
    print(f"  Datus MCP Server (Dynamic Mode - {mode_name}{cache_info})")
    print(f"  Endpoint: http://{host}:{port}/mcp/{{namespace}}")
    print(f"  With subagent: http://{host}:{port}/mcp/{{namespace}}?subagent={{name}}")
    print(f"  Info: http://{host}:{port}/")
    print(f"{'='*60}\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
    )


def main():
    """Main entry point for the MCP server CLI."""
    parser = argparse.ArgumentParser(
        description="Datus MCP Server - Expose Datus tools via Model Context Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # === Dynamic Mode (Multi-namespace HTTP Server) ===
    # Run dynamic server that supports all namespaces via URL path
    python -m datus.mcp_server --dynamic
    python -m datus.mcp_server --dynamic --host 0.0.0.0 --port 8000

    # Connect to specific namespace:
    # http://localhost:8000/mcp/{namespace}
    # http://localhost:8000/mcp/{namespace}?subagent={subagent_name}

    # === Static Mode (Single-namespace) ===
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
    # Static mode: http://localhost:8000/mcp
    # Dynamic mode: http://localhost:8000/mcp/{namespace}
    # SSE transport: http://localhost:8000/sse (static mode only)
        """,
    )

    # Mode selection: dynamic vs static
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dynamic",
        action="store_true",
        help="Run in dynamic mode: support all namespaces via /mcp/{namespace} URL",
    )
    mode_group.add_argument(
        "--namespace",
        "-n",
        help="Run in static mode with specified namespace",
    )

    parser.add_argument(
        "--database",
        "-d",
        default=None,
        help="Run in static mode with specified database name",
    )

    parser.add_argument(
        "--sub-agent",
        "-s",
        default=None,
        help="Sub-agent name for scoped context (static mode only)",
    )
    parser.add_argument(
        "--database",
        "-d",
        default=None,
        help="Database name override (static mode only)",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to agent configuration file",
    )
    parser.add_argument(
        "--transport",
        "-t",
        choices=["http", "sse", "stdio"],
        default="http",
        help="Transport type (static mode only): http (default), sse, stdio",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy mode with separate FastMCP per namespace (dynamic mode only)",
    )
    parser.add_argument(
        "--max-cache-size",
        type=int,
        default=64,
        help="Max ToolContext cache size with LRU eviction (dynamic mode only, default: 64, 0=unlimited)",
    )

    args = parser.parse_args()

    configure_logging(debug=args.debug)

    if args.dynamic:
        # Dynamic mode: run multi-namespace server
        run_dynamic_server(
            config_path=args.config,
            host=args.host,
            port=args.port,
            debug=args.debug,
            lightweight=not args.legacy,
            max_cache_size=args.max_cache_size,
        )
    else:
        # Static mode: run single-namespace server
        server = create_server(
            namespace=args.namespace,
            sub_agent=args.sub_agent,
            database_name=args.database,
            config_path=args.config,
        )
        server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
