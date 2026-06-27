# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import pytest_asyncio
import uvicorn
import yaml
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from sse_starlette.sse import AppStatus

from datus.mcp_server import DatusMCPServer

MCP_ACCEPTANCE_HOST = os.getenv("DATUS_MCP_ACCEPTANCE_HOST", "127.0.0.1")
CONFIG_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "tests" / "conf" / "agent.yml"


async def start_uvicorn(app, port: int):
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None

    config = uvicorn.Config(
        app,
        host=MCP_ACCEPTANCE_HOST,
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    if not server.started:
        raise RuntimeError(f"uvicorn server failed to start on port {port}")
    servers = getattr(server, "servers", None)
    if not servers or not servers[0].sockets:
        raise RuntimeError("uvicorn server started without a bound socket")
    bound_port = servers[0].sockets[0].getsockname()[1]
    return server, task, bound_port


async def stop_uvicorn(server, task):
    server.should_exit = True
    try:
        await asyncio.wait_for(task, timeout=10)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        AppStatus.should_exit = False
        AppStatus.should_exit_event = None
        await drain_sse_shutdown_watchers()


async def drain_sse_shutdown_watchers():
    current_task = asyncio.current_task()
    watchers = []
    for task in asyncio.all_tasks():
        if task is current_task or task.done():
            continue
        qualname = getattr(task.get_coro(), "__qualname__", "")
        if qualname.endswith("_shutdown_watcher"):
            watchers.append(task)

    for task in watchers:
        task.cancel()
    if watchers:
        await asyncio.gather(*watchers, return_exceptions=True)


@asynccontextmanager
async def mcp_http_session(url: str):
    async with streamablehttp_client(url=url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def parse_tool_result(result) -> dict:
    assert result.isError is False
    assert len(result.content) == 1
    payload = json.loads(result.content[0].text)
    assert set(payload) >= {"success", "error", "result"}
    return payload


@pytest_asyncio.fixture()
async def mcp_url(tmp_path):
    home = tmp_path / ".datus_home"
    workspace = tmp_path / "workspace"
    config_path = tmp_path / "agent.yml"
    config = yaml.safe_load(CONFIG_TEMPLATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("agent"), dict):
        pytest.fail("agent.yml template must contain an agent mapping")
    config["agent"]["home"] = str(home)
    config["agent"]["project_root"] = str(workspace)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    server = DatusMCPServer(datasource="ssb_sqlite", config_path=str(config_path), stateless_http=True)
    uvicorn_server, task, port = await start_uvicorn(server.get_streamable_http_app(), 0)
    try:
        yield f"http://{MCP_ACCEPTANCE_HOST}:{port}/mcp"
    finally:
        try:
            await stop_uvicorn(uvicorn_server, task)
        finally:
            server.close()


@pytest.mark.acceptance
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_mcp_http_registration_read_only_call_and_error_envelope(mcp_url):
    async with mcp_http_session(mcp_url) as session:
        tools = await session.list_tools()
        tool_by_name = {tool.name: tool for tool in tools.tools}

        assert {"list_subject_tree", "execute_sql"}.issubset(tool_by_name)
        assert tool_by_name["execute_sql"].inputSchema["type"] == "object"
        assert "sql" in tool_by_name["execute_sql"].inputSchema["properties"]

        subject_tree = parse_tool_result(await session.call_tool("list_subject_tree", {}))
        assert subject_tree["success"] == 1
        assert subject_tree["error"] is None
        assert isinstance(subject_tree["result"], dict)

        invalid_query = parse_tool_result(
            await session.call_tool("execute_sql", {"sql": "SELECT * FROM nonexistent_mcp_acceptance_table"})
        )
        assert invalid_query["success"] == 0
        assert invalid_query["result"] is None
        assert "nonexistent_mcp_acceptance_table" in invalid_query["error"]
