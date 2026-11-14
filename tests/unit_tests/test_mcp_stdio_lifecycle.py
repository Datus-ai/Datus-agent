"""Exercise stdio fixture failures with real child processes and MCP messages."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import pytest
from mcp.client.stdio import StdioServerParameters

from tests.helpers.mcp_stdio import StdioTestServer

pytestmark = [pytest.mark.component, pytest.mark.asyncio]

_SERVER = """
import os
import time
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stdio-fixture-regression")

@mcp.tool()
def child_pid() -> int:
    return os.getpid()

@mcp.tool()
async def unresponsive() -> int:
    time.sleep(60)
    return os.getpid()

mcp.run(transport="stdio")
"""


def server_for(tmp_path: Path, *, source: str = _SERVER, **budgets) -> StdioTestServer:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "server.py"
    script.write_text(source)
    params = StdioServerParameters(command=sys.executable, args=[str(script)], env=os.environ.copy())
    return StdioTestServer(params, tmp_path / "diagnostics", startup_timeout=10, **budgets)


def assert_reaped(server: StdioTestServer) -> None:
    assert server.pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(server.pid, 0)


async def test_ready_session_survives_separate_fixture_setup_and_teardown_tasks(tmp_path):
    server = server_for(tmp_path)
    lifecycle = server.running()
    # pytest-asyncio resumes yield fixtures on a different task at teardown.
    await asyncio.create_task(lifecycle.__aenter__())
    try:
        async with server.session() as first:
            result = await first.call_tool("child_pid", {})
            assert int(result.content[0].text) == server.pid
        async with server.session() as second:
            assert first is second
            result = await second.call_tool("child_pid", {})
            assert int(result.content[0].text) == server.pid
    finally:
        await asyncio.create_task(lifecycle.__aexit__(None, None, None))
    if os.name == "posix":
        assert_reaped(server)


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="Child stack dumps require SIGUSR1")
async def test_startup_timeout_records_child_stack_and_reaps_process(tmp_path):
    script = tmp_path / "stalled_import.py"
    script.write_text("import time\ntime.sleep(60)\n")
    params = StdioServerParameters(command=sys.executable, args=[str(script)])
    server = StdioTestServer(params, tmp_path / "diagnostics", startup_timeout=1)
    with pytest.raises(TimeoutError, match="MCP stdio startup exceeded"):
        async with server.running():
            pytest.fail("A stalled server must never reach the test body")
    assert_reaped(server)
    assert "stalled_import.py" in (server.diagnostics_dir / "stderr.log").read_text()
    records = [json.loads(line) for line in (server.diagnostics_dir / "phases.jsonl").read_text().splitlines()]
    assert any(row["phase"] == "startup" and row["event"] == "timeout" for row in records)


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="Child stack dumps require SIGUSR1")
async def test_tool_timeout_reaps_child_and_next_server_can_run(tmp_path):
    stalled = server_for(tmp_path / "stalled", call_timeout=0.2)
    with pytest.raises(TimeoutError, match="MCP stdio test exceeded"):
        async with stalled.running():
            async with stalled.session() as session:
                await session.call_tool("unresponsive", {})
    assert_reaped(stalled)
    assert "unresponsive" in (stalled.diagnostics_dir / "stderr.log").read_text()
    healthy = server_for(tmp_path / "next")
    async with healthy.running():
        async with healthy.session() as session:
            result = await session.call_tool("child_pid", {})
            assert not result.isError
    assert_reaped(healthy)


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="Child stack dumps require SIGUSR1")
async def test_shutdown_watchdog_kills_and_reaps_unresponsive_child(tmp_path):
    server = server_for(tmp_path, source=_SERVER + "\ntime.sleep(60)\n", shutdown_timeout=0.2)
    with pytest.raises(TimeoutError, match="MCP stdio shutdown exceeded"):
        async with server.running():
            pass
    assert_reaped(server)
    assert "server.py" in (server.diagnostics_dir / "stderr.log").read_text()
