import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Dict

from agents import Agent, RunContextWrapper, Usage

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

LIST_TOOLS = "list_tools"
CALL_TOOL = "call_tool"
LIST_SERVERS = "list_servers"


@asynccontextmanager
async def _safe_connect_server(server_name: str, server, max_retries: int = 3):
    """Context-managed safe MCP server connection"""
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempting to connect to MCP server {server_name} (attempt {attempt + 1}/{max_retries})")

            provider = server  # assume already created via Provider.from_process(...)
            # async context here ensures lifecycle is tracked
            async with provider:
                logger.debug(f"MCP server {server_name} connected successfully")
                yield provider
                return  # only yield once; exit after use

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
        await stack.aclose()


async def func_list_tools(connected_server, agent: Agent, run_context: RunContextWrapper):
    result = await connected_server.list_tools(run_context, agent)
    mcp_tools = [tool.name for tool in result]
    return mcp_tools


async def func_call_tool(connected_server, params: Dict[str, Any]):
    tool_name = params["tool_name"]
    result = await connected_server.call_tool(tool_name, params)
    if result.structuredContent:
        result = result.structuredContent["result"]
    return result


@asynccontextmanager
async def exe_mcp_cli_func(mcp_server_clients: Dict[str, Any], cmd: str, **kwargs) -> Any:
    """Context manager for executing cli commands through MCP server clients."""
    async with multiple_mcp_servers(mcp_server_clients) as connected_servers:
        result = {}
        for server_name, connected_server in connected_servers.items():
            try:
                # Create minimal agent and run context for the new interface
                agent = Agent(name="mcp-tools-agent")
                run_context = RunContextWrapper(context=None, usage=Usage())
                if cmd == LIST_TOOLS:
                    tools = await func_list_tools(connected_server, agent, run_context)
                    result[server_name] = tools
                elif cmd == CALL_TOOL:
                    return_data = await func_call_tool(connected_server, **kwargs)
                    result[server_name] = return_data
                elif cmd == LIST_SERVERS:
                    connected_set = set(connected_servers)
                    client_set = set(mcp_server_clients)
                    diff = client_set - connected_set
                    result.update(dict.fromkeys(connected_set, True))
                    result.update(dict.fromkeys(diff, False))
                return result
            except Exception as e:
                logger.error(f"Failed to execute MCP server {server_name} command is {cmd}: {str(e)}")
        return None
