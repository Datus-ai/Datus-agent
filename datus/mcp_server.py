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
    # === Dynamic Mode (Multi-namespace Server) ===
    # Run dynamic server with HTTP streamable (default)
    python -m datus.mcp_server --dynamic
    python -m datus.mcp_server --dynamic --host 0.0.0.0 --port 8000

    # Run dynamic server with SSE transport
    python -m datus.mcp_server --dynamic --transport sse

    # Connect to specific namespace:
    # HTTP: http://localhost:8000/mcp/{namespace}
    # SSE:  http://localhost:8000/sse/{namespace}
    # With subagent: http://localhost:8000/mcp/{namespace}?subagent={subagent_name}

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
    # Static mode:  http://localhost:8000/mcp (HTTP) or http://localhost:8000/sse (SSE)
    # Dynamic mode: http://localhost:8000/mcp/{namespace} (HTTP) or http://localhost:8000/sse/{namespace} (SSE)
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
from datus.tools.func_tool.database import DBFuncTool
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

# Import tool classes to trigger @mcp_tool_class decorator registration
# This ensures tools are registered in the global registry before server initialization
from datus.utils.mcp_decorators import get_tool_registry  # noqa: E402

# These imports trigger the @mcp_tool_class decorators, registering tools automatically
assert DBFuncTool  # Ensure imported and decorator ran
assert ContextSearchTools  # Ensure imported and decorator ran


# ============================================================================
# Lightweight Dynamic Mode Components
# ============================================================================


@dataclass
class ToolContext:
    """
    Container for namespace-scoped tool instances.

    This class holds the AgentConfig and tool instances for a specific
    namespace/subagent combination, enabling tool reuse across requests.

    Tools are automatically discovered via @mcp_tool_class decorated classes.
    """

    namespace: str
    subagent: Optional[str]
    agent_config: AgentConfig
    tools: Dict[str, Any]  # Unified tool storage: {tool_name: tool_instance}

    # Backward compatibility properties
    @property
    def db_tool(self) -> Optional[DBFuncTool]:
        return self.tools.get("db_tool")

    @property
    def context_tool(self) -> Optional[ContextSearchTools]:
        return self.tools.get("context_tool")

    @property
    def has_db_tools(self) -> bool:
        return self.db_tool is not None

    @property
    def has_context_tools(self) -> bool:
        return self.context_tool is not None

    def close(self):
        """Release resources held by this context."""
        for tool_name, tool_instance in list(self.tools.items()):
            if tool_instance:
                try:
                    # Close database connectors if available
                    if hasattr(tool_instance, "connector") and tool_instance.connector:
                        tool_instance.connector.close()
                        logger.debug(f"Closed connector for {tool_name}")
                except Exception as e:
                    logger.warning(f"Error closing {tool_name} connector: {e}")
        self.tools.clear()


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

    DEFAULT_MAX_SIZE = 64

    def __init__(self, config_path: Optional[str] = None, max_size: Optional[int] = None):
        """
        Initialize the ToolContextManager.

        Args:
            config_path: Optional path to agent configuration file
            max_size: Maximum number of contexts to cache. When exceeded,
                     the least recently used context is evicted. Default is 64.
                     Set to 0 for unlimited cache.
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
        """Create a new ToolContext with initialized tools from global registry."""
        # Load agent config with namespace
        config_kwargs = {"namespace": namespace}
        if self.config_path:
            config_kwargs["config"] = self.config_path

        agent_config = load_agent_config(**config_kwargs)

        # Initialize all tools from global registry
        tools = {}
        for tool_config in get_tool_registry():
            try:
                tool_instance = tool_config.tool_class.create_dynamic(agent_config, subagent)
                tools[tool_config.name] = tool_instance
                logger.info(
                    f"{tool_config.tool_class.__name__} initialized for namespace: {namespace} "
                    f"(multi-connector mode)"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize {tool_config.name} for {namespace}: {e}")
                tools[tool_config.name] = None

        return ToolContext(
            namespace=namespace,
            subagent=subagent,
            agent_config=agent_config,
            tools=tools,
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
                           Default is 64. Set to 0 for unlimited.
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

        # Session manager lifecycle (for HTTP streamable)
        self._session_manager = None
        self._task_group = None
        self._started = False

        # SSE app reference (for SSE transport)
        self._sse_app = None

        # SSE session to context mapping (session_id -> ToolContext)
        # This is used to route /messages/ requests to the correct context
        self._sse_sessions: Dict[str, ToolContext] = {}

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
        """Register all MCP tools from global registry."""
        from datus.utils.mcp_decorators import register_dynamic_tools

        # Automatically register all tools from global registry
        for tool_config in get_tool_registry():
            register_dynamic_tools(
                mcp=self.mcp,
                tool_class=tool_config.tool_class,
                context_getter=self._get_context,
                instance_attr=tool_config.name,
                availability_attr=tool_config.availability_property,
                format_result=self._format_result,
            )

    @asynccontextmanager
    async def lifespan_context(self, transport: Literal["http", "sse"] = "http"):
        """
        Context manager for server lifespan.
        Starts session manager in background task group (for HTTP transport).

        Args:
            transport: Transport type - "http" for streamable HTTP, "sse" for Server-Sent Events
        """
        import anyio

        if transport == "sse":
            # SSE transport: use sse_app directly (no session manager needed)
            self._sse_app = self.mcp.sse_app()
            self._started = True

            logger.info(f"LightweightDynamicMCPServer started (transport={transport})")
            try:
                yield
            finally:
                logger.info("LightweightDynamicMCPServer shutting down")
                self._started = False
                self._sse_app = None
                self._context_manager.close_all()
        else:
            # HTTP streamable transport: use session manager
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

                logger.info(f"LightweightDynamicMCPServer started (transport={transport})")
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
        transport: Literal["http", "sse"] = "http",
        subpath: str = "/",
    ):
        """
        Handle an MCP request with the specified context.

        Args:
            scope: ASGI scope
            receive: ASGI receive callable
            send: ASGI send callable
            namespace: Target namespace
            subagent: Optional subagent name
            transport: Transport type - "http" or "sse"
            subpath: Subpath after namespace (for SSE: "/" or "/messages")
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
            if transport == "sse":
                # SSE: delegate to the SSE app with correct path rewriting
                # /sse/{namespace} -> /sse (for SSE connection)
                # /sse/{namespace}/messages -> /messages (for posting messages)
                if self._sse_app is None:
                    raise RuntimeError("SSE app not initialized. Use lifespan_context with transport='sse' first.")

                new_scope = dict(scope)
                new_scope["root_path"] = ""
                if subpath == "/" or subpath == "":
                    # SSE connection endpoint
                    new_scope["path"] = "/sse"

                    # Wrap send to capture session_id from SSE events
                    async def capturing_send(message):
                        if message.get("type") == "http.response.body":
                            body = message.get("body", b"")
                            if body:
                                # Try to extract session_id from SSE event
                                # Format: "event: endpoint\ndata: /messages/?session_id=xxx\n\n"
                                body_str = body.decode("utf-8", errors="ignore")
                                if "session_id=" in body_str:
                                    import re

                                    match = re.search(r"session_id=([a-f0-9]+)", body_str)
                                    if match:
                                        session_id = match.group(1)
                                        self._sse_sessions[session_id] = context
                                        logger.debug(f"Captured SSE session: {session_id} -> {namespace}")
                        await send(message)

                    await self._sse_app(new_scope, receive, capturing_send)
                else:
                    # Other endpoints like /messages
                    new_scope["path"] = subpath
                    await self._sse_app(new_scope, receive, send)
            else:
                # HTTP streamable: use session manager
                new_scope = dict(scope)
                new_scope["path"] = "/mcp"
                await self._session_manager.handle_request(new_scope, receive, send)
        finally:
            _current_tool_context.reset(token)

    def create_asgi_app(self, transport: Literal["http", "sse"] = "http"):
        """
        Create a Starlette ASGI application with dynamic routing.

        Args:
            transport: Transport type - "http" for streamable HTTP, "sse" for Server-Sent Events

        Returns:
            Starlette application instance
        """
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route

        server = self
        current_transport = transport

        class DynamicRouter:
            """ASGI router that parses namespace from path and routes to server."""

            # Mount prefixes that need to be stripped
            MOUNT_PREFIXES = ("/sse/", "/mcp/")

            def __init__(self, transport_type: Literal["http", "sse"] = "http"):
                self.transport_type = transport_type

            @classmethod
            def _parse_request(cls, scope: Dict) -> tuple:
                """
                Parse namespace, subagent, and subpath from request.

                Handles both cases:
                1. Mount stripped the prefix: /bird_sqlite, /bird_sqlite/messages
                2. Mount didn't strip the prefix: /sse/bird_sqlite, /mcp/bird_sqlite/messages

                Returns:
                - /bird_sqlite           -> namespace="bird_sqlite", subpath="/"
                - /bird_sqlite/messages  -> namespace="bird_sqlite", subpath="/messages"
                """
                path = scope.get("path", "")
                query_string = scope.get("query_string", b"").decode("utf-8")

                # Strip mount prefix if present (Starlette Mount may not strip it in some cases)
                for prefix in cls.MOUNT_PREFIXES:
                    if path.startswith(prefix):
                        path = path[len(prefix) - 1 :]  # Keep the leading /
                        break

                parts = path.strip("/").split("/")
                if not parts or not parts[0]:
                    return None, None, "/"

                # First part is always the namespace
                namespace = parts[0]

                # Remaining parts form the subpath (for SSE: /messages, etc.)
                subpath = "/" + "/".join(parts[1:]) if len(parts) > 1 else "/"

                subagent = None
                if query_string:
                    from urllib.parse import parse_qs

                    params = parse_qs(query_string)
                    subagent = params.get("subagent", [None])[0]

                return namespace, subagent, subpath

            async def __call__(self, scope, receive, send):
                if scope["type"] != "http":
                    return

                namespace, subagent, subpath = self._parse_request(scope)

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
                    await server.handle_request(
                        scope, receive, send, namespace, subagent, transport=self.transport_type, subpath=subpath
                    )
                except Exception as e:
                    logger.error(f"Error handling request for namespace={namespace}: {e}")
                    await self._send_error(send, 500, f"Internal Server Error: {str(e)}")

            @staticmethod
            async def _send_error(send, status_code: int, message: str):
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
            logger.info(f"Starting Datus MCP Server (Lightweight Dynamic Mode, transport={current_transport})")
            logger.info(f"Available namespaces: {server.available_namespaces}")
            if server.available_subagents:
                logger.info(f"Available subagents: {server.available_subagents}")

            async with server.lifespan_context(transport=current_transport):
                yield

            logger.info("Datus MCP Server stopped")

        async def root(_request):
            # Determine endpoint based on transport
            if current_transport == "sse":
                endpoints = {
                    "sse": "/sse/{namespace}",
                    "sse_with_subagent": "/sse/{namespace}?subagent={subagent_name}",
                    "health": "/health",
                }
            else:
                endpoints = {
                    "mcp": "/mcp/{namespace}",
                    "mcp_with_subagent": "/mcp/{namespace}?subagent={subagent_name}",
                    "health": "/health",
                }

            return JSONResponse(
                {
                    "service": "Datus MCP Server",
                    "mode": "lightweight-dynamic",
                    "transport": current_transport,
                    "available_namespaces": server.available_namespaces,
                    "available_subagents": server.available_subagents,
                    "endpoints": endpoints,
                }
            )

        async def health(_request):
            return JSONResponse(
                {
                    "status": "healthy",
                    "transport": current_transport,
                    "cached_contexts": len(server._context_manager._contexts),
                }
            )

        async def handle_messages(request):
            """
            Handler for /messages/ endpoint in SSE mode.
            Routes messages to the correct context based on session_id.
            """
            from starlette.responses import JSONResponse

            # Extract session_id from query string
            session_id = request.query_params.get("session_id")

            if not session_id:
                return JSONResponse({"error": "Bad Request: Missing session_id"}, status_code=400)

            # Look up context from session_id
            context = server._sse_sessions.get(session_id)
            if not context:
                logger.warning(f"Unknown session_id: {session_id}, available: {list(server._sse_sessions.keys())}")
                return JSONResponse({"error": "Not Found: Unknown session_id"}, status_code=404)

            # Set context and delegate to SSE app
            token = _current_tool_context.set(context)
            try:
                # Build scope for SSE app
                # Use /messages/ with trailing slash to match SSE app's route and avoid 307 redirect
                scope = dict(request.scope)
                scope["root_path"] = ""
                scope["path"] = "/messages/"

                logger.debug(f"Messages request: session_id={session_id}, namespace={context.namespace}")

                # Create a response by calling the SSE app directly
                # We need to capture the response from the ASGI app
                response_status = 200
                response_headers = []
                response_body = []

                async def receive():
                    body = await request.body()
                    return {"type": "http.request", "body": body, "more_body": False}

                async def send(message):
                    nonlocal response_status, response_headers
                    if message["type"] == "http.response.start":
                        response_status = message["status"]
                        response_headers = message.get("headers", [])
                    elif message["type"] == "http.response.body":
                        body = message.get("body", b"")
                        if body:
                            response_body.append(body)

                await server._sse_app(scope, receive, send)

                # Build response
                from starlette.responses import Response

                return Response(
                    content=b"".join(response_body),
                    status_code=response_status,
                    headers={k.decode(): v.decode() for k, v in response_headers},
                )
            finally:
                _current_tool_context.reset(token)

        # Define routes based on transport
        if current_transport == "sse":
            routes = [
                Route("/", root, methods=["GET"]),
                Route("/health", health, methods=["GET"]),
                Mount("/sse", app=DynamicRouter(transport_type="sse")),
                Route("/messages/", handle_messages, methods=["POST"]),  # Handle /messages/ for SSE
                Route("/messages", handle_messages, methods=["POST"]),  # Handle /messages for SSE (no trailing slash)
            ]
        else:
            routes = [
                Route("/", root, methods=["GET"]),
                Route("/health", health, methods=["GET"]),
                Mount("/mcp", app=DynamicRouter(transport_type="http")),
            ]

        return Starlette(
            debug=False,
            routes=routes,
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

        # Initialize all tools from global registry
        self.tools = {}
        self._init_tools()

        # Register all MCP tools
        self._register_tools()

    def _init_tools(self):
        """Initialize all tools from global registry."""
        for tool_config in get_tool_registry():
            try:
                tool_instance = tool_config.tool_class.create_static(
                    self.agent_config,
                    self.sub_agent,
                    self.database_name,
                )
                self.tools[tool_config.name] = tool_instance
                logger.info(f"{tool_config.tool_class.__name__} initialized for namespace: {self.namespace}")
            except Exception as e:
                logger.warning(f"Failed to initialize {tool_config.name}: {e}")
                self.tools[tool_config.name] = None

    # Backward compatibility properties
    @property
    def db_tool(self) -> Optional[DBFuncTool]:
        return self.tools.get("db_tool")

    @property
    def context_tool(self) -> Optional[ContextSearchTools]:
        return self.tools.get("context_tool")

    def close(self):
        """
        Release all resources held by the MCP server.

        This method should be called when the server is no longer needed,
        especially for HTTP transport modes where the server lifecycle
        is managed manually.
        """
        # Close all tool instances
        for tool_name, tool_instance in list(self.tools.items()):
            if tool_instance:
                try:
                    # Close database connectors if available
                    if hasattr(tool_instance, "connector") and tool_instance.connector:
                        tool_instance.connector.close()
                        logger.info(f"{tool_name} connector closed")
                except Exception as e:
                    logger.warning(f"Error closing {tool_name} connector: {e}")

        # Clear tool references
        self.tools.clear()
        logger.info("MCP server resources released")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resources are released."""
        self.close()
        return False

    def _register_tools(self):
        """Register all available tools from global registry."""
        from datus.utils.mcp_decorators import register_static_tools

        # Automatically register all tools from global registry
        for tool_config in get_tool_registry():
            tool_instance = self.tools.get(tool_config.name)
            if tool_instance:
                register_static_tools(self.mcp, tool_instance, self._format_result)

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


def create_dynamic_app(
    config_path: Optional[str] = None,
    max_cache_size: Optional[int] = None,
    transport: Literal["http", "sse"] = "http",
):
    """
    Create a Starlette application with dynamic namespace/subagent routing.

    This function creates an ASGI app that:
    1. Loads all available namespaces from configuration at startup
    2. Routes requests to /mcp/{namespace} (HTTP) or /sse/{namespace} (SSE)
    3. Supports optional subagent parameter via query string
    4. Properly handles MCP's streaming protocol

    Args:
        config_path: Optional path to agent configuration file
        max_cache_size: Maximum number of ToolContexts to cache (LRU eviction).
                       Default is 64. Set to 0 for unlimited.
        transport: Transport type - "http" for streamable HTTP (default), "sse" for SSE

    Returns:
        Starlette application instance

    Example:
        # HTTP streamable (default)
        app = create_dynamic_app()
        # Client connects to: http://localhost:8000/mcp/demo

        # SSE transport
        app = create_dynamic_app(transport="sse")
        # Client connects to: http://localhost:8000/sse/demo
    """
    server = LightweightDynamicMCPServer(config_path=config_path, max_cache_size=max_cache_size)
    return server.create_asgi_app(transport=transport)


def run_dynamic_server(
    config_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False,
    max_cache_size: Optional[int] = 64,
    transport: Literal["http", "sse"] = "http",
):
    """
    Run the dynamic MCP server with uvicorn.

    Args:
        config_path: Optional path to agent configuration file
        host: Host to bind (default: 0.0.0.0)
        port: Port to bind (default: 8000)
        debug: Enable debug mode
        max_cache_size: Maximum ToolContext cache size (LRU). Default 64, 0 for unlimited.
        transport: Transport type - "http" for streamable HTTP (default), "sse" for SSE
    """
    import uvicorn

    app = create_dynamic_app(
        config_path=config_path,
        max_cache_size=max_cache_size,
        transport=transport,
    )

    # Determine endpoint path based on transport
    endpoint_path = "/sse" if transport == "sse" else "/mcp"

    print(f"\n{'='*60}")
    print(f"  Datus MCP Server (Dynamic Mode, transport={transport})")
    print(f"  Cache size: {max_cache_size or 64}")
    print(f"  Endpoint: http://{host}:{port}{endpoint_path}/{{namespace}}")
    print(f"  With subagent: http://{host}:{port}{endpoint_path}/{{namespace}}?subagent={{name}}")
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
    # === Dynamic Mode (Multi-namespace Server) ===
    # Run dynamic server with HTTP streamable (default)
    python -m datus.mcp_server --dynamic
    python -m datus.mcp_server --dynamic --host 0.0.0.0 --port 8000

    # Run dynamic server with SSE transport
    python -m datus.mcp_server --dynamic --transport sse

    # Connect to specific namespace:
    # HTTP: http://localhost:8000/mcp/{namespace}
    # SSE:  http://localhost:8000/sse/{namespace}
    # With subagent: ?subagent={subagent_name}

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
        help="Transport type: http (default), sse, stdio (stdio is static mode only)",
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
        "--max-cache-size",
        type=int,
        default=64,
        help="Max ToolContext cache size with LRU eviction (dynamic mode only, default: 64, 0=unlimited)",
    )

    args = parser.parse_args()

    configure_logging(debug=args.debug)

    if args.dynamic:
        # Dynamic mode: run multi-namespace server
        # Note: stdio is not supported in dynamic mode
        if args.transport == "stdio":
            parser.error("stdio transport is not supported in dynamic mode. Use http or sse.")

        run_dynamic_server(
            config_path=args.config,
            host=args.host,
            port=args.port,
            debug=args.debug,
            max_cache_size=args.max_cache_size,
            transport=args.transport,
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
