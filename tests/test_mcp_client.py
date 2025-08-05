# !/usr/bin/env python3
import os
import traceback

import pytest
from agents import Agent, RunContextWrapper, Usage

from datus.configuration.agent_config import DbConfig
from datus.configuration.mcp_config import MCP_CONF_DIR, McpConfig
from datus.models.mcp_utils import multiple_mcp_servers
from datus.tools.mcp_tools.mcp_server_client import McpServerClient


@pytest.mark.asyncio
async def test_mcp_sse_client():
    """
    Note: before test mcp_sse_client, please start an api_server_sse SSE Server
    (url=http://{host:-localhost}:{port:-8000}/sse)
    """
    client = McpServerClient.get_mcp_server_client("api_server_sse", DbConfig(**{}))
    client2 = McpServerClient.get_mcp_server_client("api_server_sse", DbConfig(**{}))
    print(f"client1 == client2: {client is client2}")
    assert client == client2
    mcp_server_clients = {"default": client}
    try:
        all_tools = []
        async with multiple_mcp_servers(mcp_server_clients) as connected_servers:
            for server_name, connected_server in connected_servers.items():
                try:
                    # Create minimal agent and run context for the new interface
                    agent = Agent(name="mcp-tools-agent")
                    run_context = RunContextWrapper(context=None, usage=Usage())
                    mcp_tools = await connected_server.list_tools(run_context, agent)
                    all_tools.extend(mcp_tools)
                    print(f"Retrieved {mcp_tools} tools from {server_name}")

                    for tool in mcp_tools:
                        if tool.name == "echo":
                            result = await connected_server.call_tool(tool.name, {"message": "Hello SSE11111111!"})
                            print(
                                f"call tool: {tool.name}  from {server_name} and get result: "
                                f"{result.structuredContent['result']}"
                            )
                        elif tool.name == "add_numbers":
                            result = await connected_server.call_tool(tool.name, {"a": 15, "b": 27})
                            print(
                                f"call tool: {tool.name}  from {server_name} and get result: "
                                f"{result.structuredContent['result']}"
                            )
                        else:
                            result = await connected_server.call_tool(tool.name, {})
                            print(
                                f"call tool: {tool.name}  from {server_name} and get result: "
                                f"{result.structuredContent['result']}"
                            )
                except Exception as e:
                    print(f"Error getting tools from {server_name}: {str(e)}")
                    continue
            print(f"Retrieved {len(all_tools)} tools from MCP servers")
        print(f"connected to sse_mcp_server_client: {mcp_server_clients}")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ OpenAI Agent SDK SSE Client error: {e}")


@pytest.mark.asyncio
async def test_mcp_streamable_http_client():
    """
    Note: before test mcp_sse_client, please start an api_server_http StreamedHTTP Server
    (url=http://{host:-localhost}:{port:-8000}/mcp)
    """
    client = McpServerClient.get_mcp_server_client("api_server_http", DbConfig(**{}))
    mcp_server_clients = {"default": client}
    try:
        all_tools = []
        async with multiple_mcp_servers(mcp_server_clients) as connected_servers:
            for server_name, connected_server in connected_servers.items():
                try:
                    # Create minimal agent and run context for the new interface
                    agent = Agent(name="mcp-tools-agent")
                    run_context = RunContextWrapper(context=None, usage=Usage())
                    mcp_tools = await connected_server.list_tools(run_context, agent)
                    all_tools.extend(mcp_tools)
                    print(f"Retrieved {mcp_tools} tools from {server_name}")

                    for tool in mcp_tools:
                        if tool.name == "echo":
                            result = await connected_server.call_tool(tool.name, {"message": "Hello SSE11111111!"})
                            print(
                                f"call tool: {tool.name}  from {server_name} and get result: "
                                f"{result.structuredContent['result']}"
                            )
                        elif tool.name == "add_numbers":
                            result = await connected_server.call_tool(tool.name, {"a": 15, "b": 27})
                            print(
                                f"call tool: {tool.name}  from {server_name} and get result: "
                                f"{result.structuredContent['result']}"
                            )
                        else:
                            result = await connected_server.call_tool(tool.name, {})
                            print(
                                f"call tool: {tool.name}  from {server_name} and get result: "
                                f"{result.structuredContent['result']}"
                            )
                except Exception as e:
                    print(f"Error getting tools from {server_name}: {str(e)}")
                    continue
            print(f"Retrieved {len(all_tools)} tools from MCP servers")
        print(f"connected to sse_mcp_server_client: {mcp_server_clients}")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ OpenAI Agent SDK SSE Client error: {e}")


def test_save_mcp_config():
    mcp_config1 = McpConfig.get_instance()
    mcp_config2 = McpConfig.get_instance()
    print(mcp_config1 is mcp_config2)
    assert mcp_config1 is mcp_config2
    new_servers = {
        "test_stdio": {"type": "stdio", "command": "echo test", "args": ["-d", "xxx.py"], "env": {"KEY": "VALUE"}},
        "test_sse": {"type": "sse", "url": "http://localhost:8080/mcp", "headers": {}, "timeout": 30},
        "test_http": {"type": "http", "url": "http://localhost:8080/mcp", "headers": {}, "timeout": 30},
    }
    mcp_config1.update_then_save_config(new_servers, "mcp_test.json")
    file = os.path.join(MCP_CONF_DIR, "mcp_test.json")
    assert os.path.exists(file)
    if os.path.exists(file):
        os.remove(file)
