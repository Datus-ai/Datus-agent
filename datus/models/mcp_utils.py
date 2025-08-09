import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Dict

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def _safe_connect_server(server_name: str, server, max_retries: int = 3):
    """Context-managed safe MCP server connection"""
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempting to connect to MCP server {server_name} (attempt {attempt + 1}/{max_retries})")

            provider = server  # assume already created via Provider.from_process(...)
            # async context here ensures lifecycle is tracked
            try:
                async with provider:
                    logger.debug(f"MCP server {server_name} connected successfully")
                    yield provider
                    return  # only yield once; exit after use
            except Exception as exit_exc:
                # Handle any exceptions during context exit (including cancel scope issues)
                logger.debug(f"MCP server {server_name} exit warning (non-critical): {str(exit_exc)}")
                # Re-raise only if it's a critical error, not cleanup-related
                if not any(keyword in str(exit_exc).lower() for keyword in ["cancel", "scope", "task", "cleanup"]):
                    raise

        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to MCP server {server_name} (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                raise
        except Exception as e:
            logger.error(f"Failed to connect MCP server {server_name} (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1.0)


@asynccontextmanager
async def multiple_mcp_servers(mcp_servers: Dict[str, Any]):
    """Context manager for managing multiple MCP servers.

    Args:
        mcp_servers: Dictionary of MCP servers to manage

    Yields:
        Dictionary of connected MCP servers
    """
    connected_servers = {}
    stack = AsyncExitStack()

    try:
        for server_name, server in mcp_servers.items():
            try:
                cm = _safe_connect_server(server_name, server)
                connected_server = await stack.enter_async_context(cm)
                connected_servers[server_name] = connected_server
            except Exception as e:
                logger.error(f"Failed to start MCP server {server_name}: {str(e)}")

        if not connected_servers:
            logger.warning("No MCP servers were successfully connected")

        yield connected_servers

    finally:
        logger.debug("Cleaning up all MCP servers via AsyncExitStack")
        try:
            await stack.aclose()
        except Exception as e:
            # Handle cancel scope errors and other cleanup exceptions silently
            logger.debug(f"MCP server cleanup warning (non-critical): {str(e)}")
            # Suppressing this error as it's typically a cleanup timing issue
