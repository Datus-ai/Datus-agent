# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for proxy_tool module."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from agents import FunctionTool

from datus.tools.proxy.proxy_tool import (
    _matches,
    _parse_patterns,
    apply_proxy_tools,
    create_proxy_tool,
)
from datus.tools.proxy.tool_result_channel import ToolResultChannel
from datus.tools.registry.tool_registry import ToolRegistry

# ---------------------------------------------------------------------------
# Tests: _parse_patterns
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestParsePatterns:
    def test_category_dot_method(self):
        result = _parse_patterns(["filesystem_tools.*"])
        assert result == [("filesystem_tools", "*")]

    def test_bare_tool_name(self):
        result = _parse_patterns(["read_file"])
        assert result == [(None, "read_file")]

    def test_wildcard(self):
        result = _parse_patterns(["*"])
        assert result == [(None, "*")]

    def test_multiple_patterns(self):
        result = _parse_patterns(["db_tools.*", "read_file", "skills.load_*"])
        assert result == [
            ("db_tools", "*"),
            (None, "read_file"),
            ("skills", "load_*"),
        ]


# ---------------------------------------------------------------------------
# Tests: _matches
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestMatches:
    def test_category_wildcard_match(self):
        registry = {"read_file": "filesystem_tools"}
        patterns = [("filesystem_tools", "*")]
        assert _matches("read_file", registry, patterns) is True

    def test_category_wildcard_no_match(self):
        registry = {"read_file": "filesystem_tools"}
        patterns = [("db_tools", "*")]
        assert _matches("read_file", registry, patterns) is False

    def test_bare_name_match(self):
        registry = {}
        patterns = [(None, "read_file")]
        assert _matches("read_file", registry, patterns) is True

    def test_bare_glob_match(self):
        registry = {}
        patterns = [(None, "read_*")]
        assert _matches("read_file", registry, patterns) is True

    def test_bare_glob_no_match(self):
        registry = {}
        patterns = [(None, "write_*")]
        assert _matches("read_file", registry, patterns) is False

    def test_wildcard_matches_all(self):
        registry = {}
        patterns = [(None, "*")]
        assert _matches("anything", registry, patterns) is True

    def test_no_category_in_registry_for_category_pattern(self):
        registry = {}
        patterns = [("filesystem_tools", "*")]
        assert _matches("read_file", registry, patterns) is False


# ---------------------------------------------------------------------------
# Tests: create_proxy_tool
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestCreateProxyTool:
    @pytest.mark.asyncio
    async def test_proxy_tool_waits_for_channel(self):
        channel = ToolResultChannel()
        original = FunctionTool(
            name="test_tool",
            description="A test tool",
            params_json_schema={"type": "object", "properties": {}},
            on_invoke_tool=lambda ctx, args: {"success": 1},
        )
        proxy = create_proxy_tool(original, channel)

        assert proxy.name == "test_tool"
        assert proxy.description == "A test tool"

        ctx = SimpleNamespace(tool_call_id="call_123")

        async def publisher():
            await asyncio.sleep(0.01)
            await channel.publish("call_123", {"success": 1, "result": "proxied"})

        task = asyncio.create_task(publisher())
        result = await proxy.on_invoke_tool(ctx, "{}")
        await task

        assert result == {"success": 1, "result": "proxied"}

    @pytest.mark.asyncio
    async def test_proxy_tool_returns_error_on_channel_cancel(self):
        """Verify that RuntimeError from channel.cancel_all is caught and returns error dict."""
        channel = ToolResultChannel()
        original = FunctionTool(
            name="test_tool",
            description="A test tool",
            params_json_schema={"type": "object", "properties": {}},
            on_invoke_tool=lambda ctx, args: {"success": 1},
        )
        proxy = create_proxy_tool(original, channel)

        ctx = SimpleNamespace(tool_call_id="call_cancel")

        async def cancel_after_delay():
            await asyncio.sleep(0.01)
            channel.cancel_all("stream ended")

        task = asyncio.create_task(cancel_after_delay())
        result = await proxy.on_invoke_tool(ctx, "{}")
        await task

        assert result["success"] == 0
        assert "stream ended" in result["error"]
        assert result["result"] is None

    @pytest.mark.asyncio
    async def test_proxy_tool_uses_tool_call_id(self):
        channel = ToolResultChannel()
        original = FunctionTool(
            name="tool",
            description="Tool",
            params_json_schema={"type": "object"},
            on_invoke_tool=lambda ctx, args: {},
        )
        proxy = create_proxy_tool(original, channel)

        ctx = SimpleNamespace(tool_call_id="call_abc123")

        async def publisher():
            await asyncio.sleep(0.05)
            await channel.publish("call_abc123", {"success": 1})

        task = asyncio.create_task(publisher())
        result = await proxy.on_invoke_tool(ctx, "{}")
        await task
        assert result == {"success": 1}


# ---------------------------------------------------------------------------
# Tests: apply_proxy_tools
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestApplyProxyTools:
    def test_replaces_matching_tools(self):
        original_invoke = MagicMock()
        tool_a = FunctionTool(
            name="read_file",
            description="Read a file",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )
        tool_b = FunctionTool(
            name="execute_sql",
            description="Execute SQL",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )

        node = SimpleNamespace(
            tools=[tool_a, tool_b],
            tool_channel=ToolResultChannel(),
            tool_registry=ToolRegistry({"read_file": "filesystem_tools", "execute_sql": "db_tools"}),
        )

        apply_proxy_tools(node, ["filesystem_tools.*"])

        # read_file should be proxied (different on_invoke_tool)
        assert node.tools[0].name == "read_file"
        assert node.tools[0].on_invoke_tool is not original_invoke

        # execute_sql should be unchanged
        assert node.tools[1].name == "execute_sql"
        assert node.tools[1].on_invoke_tool is original_invoke

    def test_non_function_tools_preserved(self):
        non_func_tool = MagicMock(spec=["name"])
        non_func_tool.name = "mcp_tool"

        func_tool = FunctionTool(
            name="read_file",
            description="Read",
            params_json_schema={"type": "object"},
            on_invoke_tool=MagicMock(),
        )

        node = SimpleNamespace(
            tools=[non_func_tool, func_tool],
            tool_channel=ToolResultChannel(),
            tool_registry=ToolRegistry(),
        )

        apply_proxy_tools(node, ["*"])

        # non-FunctionTool should be preserved as-is
        assert node.tools[0] is non_func_tool
        # FunctionTool should be proxied
        assert node.tools[1].name == "read_file"
