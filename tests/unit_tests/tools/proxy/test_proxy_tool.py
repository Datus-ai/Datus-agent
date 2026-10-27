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
    _FS_DEPENDENT_NODES,
    _matches,
    _parse_patterns,
    apply_proxy_tools,
    create_proxy_tool,
    prepare_web_tool_action,
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
    async def test_proxy_tool_returns_error_on_timeout(self, monkeypatch):
        """A client that never reports must time out into a failure result."""
        monkeypatch.setattr("datus.tools.proxy.proxy_tool.DEFAULT_RESULT_TIMEOUT", 0.01)
        channel = ToolResultChannel()
        original = FunctionTool(
            name="write_file",
            description="A test tool",
            params_json_schema={"type": "object", "properties": {}},
            on_invoke_tool=lambda ctx, args: {"success": 1},
        )
        proxy = create_proxy_tool(original, channel)

        ctx = SimpleNamespace(tool_call_id="call_timeout")
        # No publisher — the wait must elapse and surface a failure, not hang.
        result = await proxy.on_invoke_tool(ctx, "{}")

        assert result["success"] == 0
        error = result["error"]
        # The model is the first reader of this string and the only actor that
        # can recover, so it has to be accurate about what is and is not known.
        # The client executes and only then reports: the work may have happened
        # and the report arrived after the waiter settled, or the user may press
        # Accept later and it runs without ever reaching us. Claiming it did not
        # run would send the model off to redo a write that already landed.
        assert "write_file" in error
        assert "UNKNOWN" in error
        assert "expired" in error
        # ``publish`` drops a result whose waiter already settled, so pointing
        # the user at the stale card would promise a recovery that cannot work.
        assert "dropped" in error
        assert "fresh call" in error
        assert result["result"] is None
        # Telemetry distinguishes this from an ordinary tool failure.
        assert ctx._datus_tool_status == "proxy_timeout"

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

    def test_proxy_tool_preserves_strict_json_schema(self) -> None:
        channel = ToolResultChannel()
        original = FunctionTool(
            name="tool",
            description="Tool",
            params_json_schema={"type": "object"},
            on_invoke_tool=lambda ctx, args: {},
            strict_json_schema=False,
        )

        proxy = create_proxy_tool(original, channel)

        assert proxy.strict_json_schema is False


# ---------------------------------------------------------------------------
# Tests: apply_proxy_tools
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestApplyProxyToolsWithExplicitChannel:
    """Tests for apply_proxy_tools with the optional channel parameter."""

    def test_uses_explicit_channel(self):
        """When channel is passed, proxy tools use that channel instead of node.tool_channel."""
        original_invoke = MagicMock()
        tool_a = FunctionTool(
            name="read_file",
            description="Read a file",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )

        node_channel = ToolResultChannel()
        explicit_channel = ToolResultChannel()

        node = SimpleNamespace(
            tools=[tool_a],
            tool_channel=node_channel,
            tool_registry=ToolRegistry(),
            proxy_tool_patterns=None,
        )

        apply_proxy_tools(node, ["*"], channel=explicit_channel)

        # Tool should be proxied
        assert node.tools[0].on_invoke_tool is not original_invoke

    @pytest.mark.asyncio
    async def test_proxy_tool_receives_from_explicit_channel(self):
        """Proxy tool created with explicit channel can receive results from it."""
        original_invoke = MagicMock()
        tool_a = FunctionTool(
            name="read_file",
            description="Read a file",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )

        node_channel = ToolResultChannel()
        explicit_channel = ToolResultChannel()

        node = SimpleNamespace(
            tools=[tool_a],
            tool_channel=node_channel,
            tool_registry=ToolRegistry(),
            proxy_tool_patterns=None,
        )

        apply_proxy_tools(node, ["*"], channel=explicit_channel)

        ctx = SimpleNamespace(tool_call_id="call_explicit")

        async def publisher():
            await asyncio.sleep(0.01)
            await explicit_channel.publish("call_explicit", {"success": 1, "result": "from_explicit"})

        task = asyncio.create_task(publisher())
        result = await node.tools[0].on_invoke_tool(ctx, "{}")
        await task

        assert result == {"success": 1, "result": "from_explicit"}

    def test_none_channel_falls_back_to_node_channel(self):
        """When channel=None, proxy tools use node.tool_channel (backward compatible)."""
        original_invoke = MagicMock()
        tool_a = FunctionTool(
            name="read_file",
            description="Read a file",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )

        node = SimpleNamespace(
            tools=[tool_a],
            tool_channel=ToolResultChannel(),
            tool_registry=ToolRegistry(),
            proxy_tool_patterns=None,
        )

        apply_proxy_tools(node, ["*"], channel=None)

        # Tool should still be proxied
        assert node.tools[0].on_invoke_tool is not original_invoke

    def test_stores_patterns_on_node(self):
        """apply_proxy_tools stores the patterns on node.proxy_tool_patterns."""
        node = SimpleNamespace(
            tools=[],
            tool_channel=ToolResultChannel(),
            tool_registry=ToolRegistry(),
            proxy_tool_patterns=None,
        )

        apply_proxy_tools(node, ["filesystem_tools.*", "read_file"])

        assert node.proxy_tool_patterns == ["filesystem_tools.*", "read_file"]


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

    def test_web_source_proxies_all_filesystem_writes(self):
        """The 'web' source proxies write_file/edit_file/delete_file (the
        client-owned write subset) while leaving read-only fs tools server-side.
        Mirrors the pattern list applied in ChatTaskManager._run_loop."""
        original_invoke = MagicMock()

        def _fs_tool(name):
            return FunctionTool(
                name=name,
                description=name,
                params_json_schema={"type": "object"},
                on_invoke_tool=original_invoke,
            )

        tools = [_fs_tool(n) for n in ("write_file", "edit_file", "delete_file", "read_file")]
        node = SimpleNamespace(
            tools=tools,
            tool_channel=ToolResultChannel(),
            tool_registry=ToolRegistry(
                {
                    "write_file": "filesystem_tools",
                    "edit_file": "filesystem_tools",
                    "delete_file": "filesystem_tools",
                    "read_file": "filesystem_tools",
                }
            ),
            proxy_tool_patterns=None,
        )

        apply_proxy_tools(node, ["write_file", "edit_file", "delete_file"])

        by_name = {t.name: t for t in node.tools}
        assert by_name["write_file"].on_invoke_tool is not original_invoke
        assert by_name["edit_file"].on_invoke_tool is not original_invoke
        assert by_name["delete_file"].on_invoke_tool is not original_invoke
        # read_file is not in the write subset and must stay server-side.
        assert by_name["read_file"].on_invoke_tool is original_invoke
        assert node.proxied_tool_names == {"write_file", "edit_file", "delete_file"}

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


# ---------------------------------------------------------------------------
# Tests: FS-dependent node exclusion
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestFsDependentNodeExclusion:
    """Filesystem-dependent nodes auto-exclude filesystem_tools from proxying."""

    @pytest.mark.parametrize("node_name", sorted(_FS_DEPENDENT_NODES))
    def test_filesystem_tools_not_proxied_on_fs_dependent_node(self, node_name):
        """filesystem_tools category tools are NOT proxied on fs-dependent nodes."""
        original_invoke = MagicMock()
        write_file = FunctionTool(
            name="write_file",
            description="Write a file",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )
        read_file = FunctionTool(
            name="read_file",
            description="Read a file",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )

        node = SimpleNamespace(
            tools=[write_file, read_file],
            tool_channel=ToolResultChannel(),
            tool_registry=ToolRegistry({"write_file": "filesystem_tools", "read_file": "filesystem_tools"}),
            proxy_tool_patterns=None,
            get_node_name=lambda: node_name,
        )

        apply_proxy_tools(node, ["*"])

        # Both filesystem tools should remain un-proxied
        assert node.tools[0].on_invoke_tool is original_invoke
        assert node.tools[1].on_invoke_tool is original_invoke

    @pytest.mark.parametrize("node_name", sorted(_FS_DEPENDENT_NODES))
    def test_non_filesystem_tools_still_proxied_on_fs_dependent_node(self, node_name):
        """Non-filesystem tools are still proxied on fs-dependent nodes."""
        original_invoke = MagicMock()
        write_file = FunctionTool(
            name="write_file",
            description="Write a file",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )
        execute_sql = FunctionTool(
            name="execute_sql",
            description="Execute SQL",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )

        node = SimpleNamespace(
            tools=[write_file, execute_sql],
            tool_channel=ToolResultChannel(),
            tool_registry=ToolRegistry({"write_file": "filesystem_tools", "execute_sql": "db_tools"}),
            proxy_tool_patterns=None,
            get_node_name=lambda: node_name,
        )

        apply_proxy_tools(node, ["*"])

        # write_file (filesystem_tools) should NOT be proxied
        assert node.tools[0].on_invoke_tool is original_invoke
        # execute_sql (db_tools) should be proxied
        assert node.tools[1].on_invoke_tool is not original_invoke

    def test_filesystem_tools_proxied_on_non_fs_dependent_node(self):
        """On non-fs-dependent nodes, filesystem_tools are proxied normally (regression)."""
        original_invoke = MagicMock()
        write_file = FunctionTool(
            name="write_file",
            description="Write a file",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )

        node = SimpleNamespace(
            tools=[write_file],
            tool_channel=ToolResultChannel(),
            tool_registry=ToolRegistry({"write_file": "filesystem_tools"}),
            proxy_tool_patterns=None,
            get_node_name=lambda: "gen_sql",
        )

        apply_proxy_tools(node, ["filesystem_tools.*"])

        # Should be proxied on a non-fs-dependent node
        assert node.tools[0].on_invoke_tool is not original_invoke

    def test_node_without_get_node_name_proxied_normally(self):
        """Nodes without get_node_name method have all tools proxied normally."""
        original_invoke = MagicMock()
        write_file = FunctionTool(
            name="write_file",
            description="Write a file",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )

        node = SimpleNamespace(
            tools=[write_file],
            tool_channel=ToolResultChannel(),
            tool_registry=ToolRegistry({"write_file": "filesystem_tools"}),
            proxy_tool_patterns=None,
        )

        apply_proxy_tools(node, ["*"])

        # Should be proxied since node has no get_node_name
        assert node.tools[0].on_invoke_tool is not original_invoke


# ---------------------------------------------------------------------------
# Tests: proxied_tool_names exposure for PermissionHooks bypass
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestProxiedToolNames:
    """``apply_proxy_tools`` records actually-wrapped tool names on the node."""

    def _make_node(self, tools, registry=None, node_name=None, proxied_attr="missing"):
        kwargs = {
            "tools": tools,
            "tool_channel": ToolResultChannel(),
            "tool_registry": registry or ToolRegistry(),
            "proxy_tool_patterns": None,
        }
        if node_name is not None:
            kwargs["get_node_name"] = lambda: node_name
        if proxied_attr == "set":
            kwargs["proxied_tool_names"] = set()
        node = SimpleNamespace(**kwargs)
        return node

    def test_records_wrapped_tool_names(self):
        original_invoke = MagicMock()
        read_file = FunctionTool(
            name="read_file",
            description="Read",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )
        execute_sql = FunctionTool(
            name="execute_sql",
            description="SQL",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )

        node = self._make_node(
            [read_file, execute_sql],
            registry=ToolRegistry({"read_file": "filesystem_tools", "execute_sql": "db_tools"}),
        )

        apply_proxy_tools(node, ["filesystem_tools.*"])

        # Only the wrapped tool's name is recorded.
        assert node.proxied_tool_names == {"read_file"}

    def test_excludes_fs_dependent_skipped_tools(self):
        original_invoke = MagicMock()
        write_file = FunctionTool(
            name="write_file",
            description="Write",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )
        execute_sql = FunctionTool(
            name="execute_sql",
            description="SQL",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )

        # semantic_modeling is in _FS_DEPENDENT_NODES — write_file must NOT be
        # proxied or recorded, but execute_sql still is.
        node = self._make_node(
            [write_file, execute_sql],
            registry=ToolRegistry({"write_file": "filesystem_tools", "execute_sql": "db_tools"}),
            node_name="semantic_modeling",
        )

        apply_proxy_tools(node, ["*"])

        assert node.proxied_tool_names == {"execute_sql"}

    def test_excludes_non_function_tools(self):
        non_func_tool = MagicMock(spec=["name"])
        non_func_tool.name = "mcp_tool"
        read_file = FunctionTool(
            name="read_file",
            description="Read",
            params_json_schema={"type": "object"},
            on_invoke_tool=MagicMock(),
        )

        node = self._make_node(
            [non_func_tool, read_file],
            registry=ToolRegistry({"read_file": "filesystem_tools"}),
        )

        apply_proxy_tools(node, ["*"])

        # Non-FunctionTool instances are kept as-is and therefore NOT recorded.
        assert node.proxied_tool_names == {"read_file"}

    def test_subsequent_calls_overwrite(self):
        original_invoke = MagicMock()
        read_file = FunctionTool(
            name="read_file",
            description="Read",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )
        execute_sql = FunctionTool(
            name="execute_sql",
            description="SQL",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )

        node = self._make_node(
            [read_file, execute_sql],
            registry=ToolRegistry({"read_file": "filesystem_tools", "execute_sql": "db_tools"}),
        )

        apply_proxy_tools(node, ["filesystem_tools.*"])
        assert node.proxied_tool_names == {"read_file"}

        apply_proxy_tools(node, ["db_tools.*"])
        # The second call resets the set: only db_tools matches now.
        assert node.proxied_tool_names == {"execute_sql"}

    def test_mutates_existing_set_reference(self):
        """When node already exposes a set, ``apply_proxy_tools`` updates it in
        place so PermissionHooks built earlier still observes the latest names.
        """
        original_invoke = MagicMock()
        read_file = FunctionTool(
            name="read_file",
            description="Read",
            params_json_schema={"type": "object"},
            on_invoke_tool=original_invoke,
        )

        node = self._make_node(
            [read_file],
            registry=ToolRegistry({"read_file": "filesystem_tools"}),
            proxied_attr="set",
        )
        held_reference = node.proxied_tool_names

        apply_proxy_tools(node, ["filesystem_tools.*"])

        assert node.proxied_tool_names is held_reference
        assert held_reference == {"read_file"}


@pytest.mark.asyncio
@pytest.mark.parametrize("node_name", ["chat", "semantic_modeling", "gen_sql_summary"])
@pytest.mark.parametrize("operation", ["write_file", "edit_file", "delete_file"])
async def test_web_main_and_children_share_existing_proxy_lifecycle(tmp_path, node_name, operation):
    import json
    from unittest.mock import AsyncMock, Mock

    from datus.api.services.action_sse_converter import _build_tool_call_content
    from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
    from datus.tools.permission.permission_hooks import PermissionHooks
    from datus.tools.permission.permission_manager import PermissionManager
    from datus.tools.permission.profiles import get_profile
    from datus.tools.proxy.proxy_tool import stamp_tool_execution

    target = tmp_path / "artifact.txt"
    target.write_text("before")
    guard, changed = Mock(), Mock()
    fs = FilesystemFuncTool(root_path=str(tmp_path), strict=True, mutation_guard=guard, mutation_callback=changed)
    channel = ToolResultChannel()
    node = SimpleNamespace(
        tools=fs.available_tools(),
        tool_channel=ToolResultChannel(),
        tool_registry=ToolRegistry({t.name: "filesystem_tools" for t in fs.available_tools()}),
        proxied_tool_names=set(),
        get_node_name=lambda: node_name,
        scope="owner",
        agent_config=SimpleNamespace(_client_source="web"),
    )
    apply_proxy_tools(node, ["write_file", "edit_file", "delete_file"], channel=channel)
    assert node.tool_channel is channel
    tool = next(t for t in node.tools if t.name == operation)
    args = {"path": "artifact.txt"}
    args.update({"content": "after"} if operation == "write_file" else {})
    args.update({"old_string": "before", "new_string": "after"} if operation == "edit_file" else {})
    ctx = SimpleNamespace(tool_call_id="call-web", tool_arguments=json.dumps(args))
    broker = SimpleNamespace(request=AsyncMock())
    hooks = PermissionHooks(
        broker=broker,
        permission_manager=PermissionManager(get_profile("normal")),
        node_name=node_name,
        tool_registry=node.tool_registry,
        proxied_tool_names=node.proxied_tool_names,
        web_filesystem_proxy_names=node.web_filesystem_proxy_names,
    )
    await hooks.on_tool_start(ctx, None, tool)
    broker.request.assert_not_called()
    action = ActionHistory(
        action_id="call-web",
        role=ActionRole.TOOL,
        action_type="tool_call",
        input={"function_name": operation, "arguments": args},
        status=ActionStatus.PROCESSING,
    )
    await prepare_web_tool_action(action, node)
    stamp_tool_execution(action, node)
    assert _build_tool_call_content(action, set())[0].payload["proxied"] is True
    guard.assert_called_once()
    changed.assert_not_called()
    running = asyncio.create_task(tool.on_invoke_tool(ctx, json.dumps(args)))
    await asyncio.sleep(0)
    assert target.read_text() == "before"
    # The existing IDE executor performs the mutation, then tool_result resumes
    # the proxy. No Agent tool-dispatch endpoint participates in this lifecycle.
    import json
    from pathlib import Path

    client_args = json.loads(action.input["arguments"])
    target = Path(client_args["path"])
    if operation == "write_file":
        target.write_text(client_args["content"])
    elif operation == "edit_file":
        target.write_text(target.read_text().replace(client_args["old_string"], client_args["new_string"], 1))
    else:
        target.unlink()
    assert not running.done()  # Finish Edit still owns the final report.
    assert target.exists() == (operation != "delete_file")
    await channel.publish("call-web", {"success": 1, "result": "user finished editing"})
    assert (await running)["result"] == "user finished editing"
    changed.assert_called_once()
    guard.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [False, True])
async def test_web_cancel_or_stop_never_executes_the_original_tool(stop):
    from unittest.mock import AsyncMock

    original = AsyncMock(return_value={"success": 1})
    tool = FunctionTool(
        name="write_file", description="write", params_json_schema={"type": "object"}, on_invoke_tool=original
    )
    channel = ToolResultChannel()
    proxy = create_proxy_tool(tool, channel, source="web")
    running = asyncio.create_task(proxy.on_invoke_tool(SimpleNamespace(tool_call_id="cancel"), "{}"))
    await asyncio.sleep(0)
    if stop:
        channel.cancel_all("stop")
    else:
        await channel.publish("cancel", {"success": 0, "error": "user canceled"})
    assert (await running)["success"] == 0
    original.assert_not_awaited()


@pytest.mark.asyncio
async def test_web_proxy_preserves_original_validation_failure(tmp_path):
    from unittest.mock import Mock

    from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

    fs = FilesystemFuncTool(
        root_path=str(tmp_path), mutation_guard=Mock(side_effect=ValueError("outside the semantic authoring target"))
    )
    channel = ToolResultChannel()
    node = SimpleNamespace(
        tools=fs.available_tools(),
        tool_registry=ToolRegistry({"write_file": "filesystem_tools"}),
        tool_channel=channel,
        agent_config=SimpleNamespace(_client_source="web"),
    )
    apply_proxy_tools(node, ["write_file"])
    action = ActionHistory(
        action_id="blocked",
        role=ActionRole.TOOL,
        action_type="write_file",
        input={"function_name": "write_file", "arguments": {"path": "forbidden", "content": "after"}},
        status=ActionStatus.PROCESSING,
    )
    await prepare_web_tool_action(action, node)
    assert action.input["_proxied"] is False
    tool = next(t for t in node.tools if t.name == "write_file")
    result = await tool.on_invoke_tool(SimpleNamespace(tool_call_id="blocked"), "{}")
    assert result["success"] == 0
    assert "outside the semantic authoring target" in result["error"]
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.asyncio
async def test_nested_web_nodes_reuse_the_parent_result_channel():
    from unittest.mock import AsyncMock

    shared = ToolResultChannel()
    nodes = []
    for _ in range(3):
        tool = FunctionTool(
            name="write_file",
            description="write",
            params_json_schema={"type": "object"},
            on_invoke_tool=AsyncMock(return_value={"success": 1}),
        )
        node = SimpleNamespace(
            tools=[tool],
            tool_channel=ToolResultChannel(),
            tool_registry=ToolRegistry({"write_file": "filesystem_tools"}),
            proxied_tool_names=set(),
            agent_config=SimpleNamespace(_client_source="web"),
            get_node_name=lambda: "custom",
        )
        apply_proxy_tools(node, ["write_file"], channel=shared)
        nodes.append(node)
        shared = node.tool_channel
    running = asyncio.create_task(nodes[-1].tools[0].on_invoke_tool(SimpleNamespace(tool_call_id="nested"), "{}"))
    await asyncio.sleep(0)
    await nodes[0].tool_channel.publish("nested", {"success": 1})
    assert (await running)["success"] == 1


@pytest.mark.parametrize("source", [None, "cli", "vscode"])
@pytest.mark.parametrize("node_name", sorted(_FS_DEPENDENT_NODES))
def test_cli_and_vscode_keep_native_filesystem_and_existing_action_shape(source, node_name):
    from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
    from datus.tools.proxy.proxy_tool import stamp_tool_execution

    original = MagicMock()
    tool = FunctionTool(
        name="write_file", description="write", params_json_schema={"type": "object"}, on_invoke_tool=original
    )
    own_channel, parent_channel = ToolResultChannel(), ToolResultChannel()
    node = SimpleNamespace(
        tools=[tool],
        tool_channel=own_channel,
        tool_registry=ToolRegistry({"write_file": "filesystem_tools"}),
        proxied_tool_names=set(),
        agent_config=SimpleNamespace(_client_source=source),
        get_node_name=lambda: node_name,
    )
    apply_proxy_tools(node, ["filesystem_tools.*"], channel=parent_channel)
    assert node.tools[0].on_invoke_tool is original
    assert node.tool_channel is own_channel
    action = ActionHistory(
        action_id="cli",
        role=ActionRole.TOOL,
        action_type="tool_call",
        input={"function_name": "write_file", "arguments": {}},
        status=ActionStatus.PROCESSING,
    )
    before = action.model_dump()
    stamp_tool_execution(action, node)
    assert action.model_dump() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["report", "dashboard"])
@pytest.mark.parametrize("operation", ["write_file", "edit_file", "delete_file"])
async def test_web_native_artifacts_preserve_path_policy(tmp_path, kind, operation):
    import json

    from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
    from datus.tools.func_tool.dashboard_artifact_tools import DashboardFilesystemFuncTool
    from datus.tools.func_tool.report_artifact_tools import ReportFilesystemFuncTool
    from datus.tools.proxy.proxy_tool import stamp_tool_execution

    cls = ReportFilesystemFuncTool if kind == "report" else DashboardFilesystemFuncTool
    fs = cls(root_path=str(tmp_path))
    channel = ToolResultChannel()
    node = SimpleNamespace(
        tools=fs.available_tools(),
        tool_channel=channel,
        tool_registry=ToolRegistry({operation: "filesystem_tools"}),
        agent_config=SimpleNamespace(_client_source="web"),
        get_node_name=lambda: f"gen_visual_{kind}",
    )
    apply_proxy_tools(node, [operation])
    args = {"path": f"{kind}s/demo/queries/protected.sql"}
    if operation == "write_file":
        args["content"] = "SELECT 1"
    if operation == "edit_file":
        args.update(old_string="SELECT 1", new_string="SELECT 2")
    action = ActionHistory(
        action_id="protected",
        role=ActionRole.TOOL,
        action_type=operation,
        input={"function_name": operation, "arguments": args},
        status=ActionStatus.PROCESSING,
    )
    await prepare_web_tool_action(action, node)
    stamp_tool_execution(action, node)
    assert action.input["_proxied"] is False
    tool = next(t for t in node.tools if t.name == operation)
    result = await tool.on_invoke_tool(SimpleNamespace(tool_call_id="protected"), json.dumps(args))
    assert result["success"] == 0
    assert "save_query" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["report", "dashboard"])
@pytest.mark.parametrize("operation", ["write_file", "edit_file", "delete_file"])
@pytest.mark.parametrize(
    "relative_path,before,after",
    [
        ("render/app.jsx", "export default () => 'before';", "export default () => 'after';"),
        ("manifest.json", '{"name":"before"}', '{"name":"after"}'),
        ("analysis/notes.sql", "SELECT 1", "SELECT 2"),
    ],
)
async def test_web_visual_subagents_execute_files_natively_without_client_confirmation(
    tmp_path, kind, operation, relative_path, before, after
):
    import json
    from unittest.mock import AsyncMock

    from datus.api.services.action_sse_converter import _build_tool_call_content
    from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
    from datus.tools.func_tool.dashboard_artifact_tools import DashboardFilesystemFuncTool
    from datus.tools.func_tool.report_artifact_tools import ReportFilesystemFuncTool
    from datus.tools.permission.permission_hooks import FilesystemPolicy, PermissionHooks
    from datus.tools.permission.permission_manager import PermissionManager
    from datus.tools.permission.profiles import get_profile
    from datus.tools.proxy.proxy_tool import stamp_tool_execution

    node_name = f"gen_visual_{kind}"
    cls = ReportFilesystemFuncTool if kind == "report" else DashboardFilesystemFuncTool
    fs = cls(root_path=str(tmp_path), strict=True, current_node=node_name)
    channel = ToolResultChannel()
    node = SimpleNamespace(
        tools=fs.available_tools(),
        tool_channel=channel,
        tool_registry=ToolRegistry({t.name: "filesystem_tools" for t in fs.available_tools()}),
        agent_config=SimpleNamespace(_client_source="web"),
        get_node_name=lambda: node_name,
        scope="owner",
    )
    apply_proxy_tools(node, ["write_file", "edit_file", "delete_file"])
    target = tmp_path / f"{kind}s/demo/{relative_path}"
    target.parent.mkdir(parents=True)
    target.write_text(before)
    args = {"path": str(target)}
    args.update({"content": after} if operation == "write_file" else {})
    args.update({"old_string": before, "new_string": after} if operation == "edit_file" else {})
    action = ActionHistory(
        action_id="native-artifact",
        role=ActionRole.TOOL,
        action_type=operation,
        input={"function_name": operation, "arguments": args},
        status=ActionStatus.PROCESSING,
    )
    await prepare_web_tool_action(action, node)
    stamp_tool_execution(action, node)
    # Even when the parent proxies file writes, this producing node owns them;
    # the frontend receives a display-only card and cannot execute it twice.
    assert _build_tool_call_content(action, {operation})[0].payload["proxied"] is False
    broker = SimpleNamespace(request=AsyncMock())
    hooks = PermissionHooks(
        broker=broker,
        permission_manager=PermissionManager(get_profile("normal")),
        node_name=node_name,
        tool_registry=node.tool_registry,
        proxied_tool_names=node.proxied_tool_names,
        web_filesystem_proxy_names=node.web_filesystem_proxy_names,
        fs_policy=FilesystemPolicy(root_path=tmp_path, current_node=node_name, strict=True),
    )
    tool = next(t for t in node.tools if t.name == operation)
    ctx = SimpleNamespace(tool_call_id="native-artifact", tool_arguments=json.dumps(args))
    await hooks.on_tool_start(ctx, None, tool)
    broker.request.assert_not_called()
    result = await asyncio.wait_for(tool.on_invoke_tool(ctx, json.dumps(args)), timeout=1)
    assert result["success"] == 1
    assert target.exists() == (operation != "delete_file")
    assert (target.read_text() if target.exists() else None) == (None if operation == "delete_file" else after)


@pytest.mark.asyncio
async def test_web_sql_summary_preserves_snapshots_finalization_and_rollback(tmp_path, monkeypatch):
    import json
    import os
    from pathlib import Path
    from unittest.mock import Mock

    from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
    from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

    summary = SqlSummaryAgenticNode.__new__(SqlSummaryAgenticNode)
    summary.configured_node_name = "gen_sql_summary"
    summary.knowledge_base_dir = str(tmp_path / "subject")
    summary._mutation_snapshots = {}
    summary._mutated_files = set()
    summary.storage_type = "reference_sql"
    summary.agent_config = SimpleNamespace(_client_source="web")
    target = tmp_path / "subject/sql_summaries/query.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("sql: SELECT 1\n")
    original = target.read_bytes()
    fs = FilesystemFuncTool(
        root_path=str(tmp_path),
        mutation_guard=summary._sql_summary_mutation_guard,
        mutation_callback=summary._record_sql_summary_mutation,
        client_mutation_guard=summary._validate_sql_summary_mutation_path,
        client_mutation_callback=summary._record_sql_summary_client_mutation,
    )
    summary.tools = fs.available_tools()
    summary.tool_channel = ToolResultChannel()
    summary.tool_registry = ToolRegistry({"write_file": "filesystem_tools"})
    apply_proxy_tools(summary, ["write_file"])
    action = ActionHistory(
        action_id="summary",
        role=ActionRole.TOOL,
        action_type="write_file",
        input={"function_name": "write_file", "arguments": {"path": str(target), "content": "sql: SELECT 2\n"}},
        status=ActionStatus.PROCESSING,
    )
    await prepare_web_tool_action(action, summary)
    assert not summary._mutation_snapshots
    assert not summary._mutated_files
    client_args = json.loads(action.input["arguments"])
    Path(client_args["path"]).write_text(client_args["content"])
    await summary.tool_channel.publish("summary", {"success": 1})
    tool = next(t for t in summary.tools if t.name == "write_file")
    assert (await tool.on_invoke_tool(SimpleNamespace(tool_call_id="summary"), "{}"))["success"] == 1
    assert os.path.realpath(target) in summary._mutated_files
    assert summary._mutation_snapshots[os.path.realpath(target)][0] == original
    sync = Mock(return_value={"success": True})
    monkeypatch.setattr("datus.storage.reference_sql.artifact_sync.sync_reference_sql_artifact_to_kb", sync)
    summary._finalize_sql_summary_artifact(str(target))
    sync.assert_called_once_with(str(target), summary.agent_config)
    summary._rollback_sql_summary_mutations()
    assert target.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [False, True])
async def test_web_prepared_file_cancel_keeps_file_and_callbacks_unchanged(tmp_path, stop):
    from unittest.mock import Mock

    from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

    target = tmp_path / "artifact.txt"
    target.write_text("before")
    changed = Mock()
    fs = FilesystemFuncTool(root_path=str(tmp_path), mutation_callback=changed)
    channel = ToolResultChannel()
    node = SimpleNamespace(
        tools=fs.available_tools(),
        tool_channel=channel,
        tool_registry=ToolRegistry({"write_file": "filesystem_tools"}),
        agent_config=SimpleNamespace(_client_source="web"),
    )
    apply_proxy_tools(node, ["write_file"])
    action = ActionHistory(
        action_id="cancel",
        role=ActionRole.TOOL,
        action_type="write_file",
        input={"function_name": "write_file", "arguments": {"path": str(target), "content": "after"}},
        status=ActionStatus.PROCESSING,
    )
    await prepare_web_tool_action(action, node)
    tool = next(t for t in node.tools if t.name == "write_file")
    running = asyncio.create_task(tool.on_invoke_tool(SimpleNamespace(tool_call_id="cancel"), "{}"))
    await asyncio.sleep(0)
    if stop:
        channel.cancel_all("stop")
    else:
        await channel.publish("cancel", {"success": 0, "error": "user canceled"})
    assert (await running)["success"] == 0
    assert target.read_text() == "before"
    changed.assert_not_called()


@pytest.mark.asyncio
async def test_web_denied_file_never_exposes_an_executable_card(tmp_path):
    from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
    from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
    from datus.tools.permission.permission_manager import PermissionManager

    fs = FilesystemFuncTool(root_path=str(tmp_path))
    node = SimpleNamespace(
        tools=fs.available_tools(),
        tool_channel=ToolResultChannel(),
        tool_registry=ToolRegistry({"write_file": "filesystem_tools"}),
        agent_config=SimpleNamespace(_client_source="web"),
        get_node_name=lambda: "chat",
        permission_manager=PermissionManager(
            global_config=PermissionConfig(
                rules=[PermissionRule(tool="filesystem_tools", pattern="write_file", permission=PermissionLevel.DENY)]
            )
        ),
    )
    apply_proxy_tools(node, ["write_file"])
    action = ActionHistory(
        action_id="deny",
        role=ActionRole.TOOL,
        action_type="write_file",
        input={"function_name": "write_file", "arguments": {"path": "artifact.txt", "content": "after"}},
        status=ActionStatus.PROCESSING,
    )
    await prepare_web_tool_action(action, node)
    from datus.tools.proxy.proxy_tool import stamp_tool_execution

    stamp_tool_execution(action, node)
    assert action.input["_proxied"] is False
    assert not (tmp_path / "artifact.txt").exists()


@pytest.mark.parametrize("existed", [False, True])
def test_unaccepted_web_summary_never_rolls_back_user_edits(tmp_path, existed):
    from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

    node = SqlSummaryAgenticNode.__new__(SqlSummaryAgenticNode)
    node.knowledge_base_dir = str(tmp_path / "subject")
    node._mutation_snapshots = {}
    node._mutated_files = set()
    target = tmp_path / "subject/sql_summaries/query.yaml"
    target.parent.mkdir(parents=True)
    if existed:
        target.write_text("sql: SELECT 1\n")
    fs = FilesystemFuncTool(
        root_path=str(tmp_path),
        mutation_guard=node._sql_summary_mutation_guard,
        mutation_callback=node._record_sql_summary_mutation,
        client_mutation_guard=node._validate_sql_summary_mutation_path,
        client_mutation_callback=node._record_sql_summary_client_mutation,
    )
    fs._prepare_client_mutation("write_file", {"path": str(target), "content": "sql: SELECT 2\n"})
    target.write_text("sql: SELECT 'user saved while pending'\n")
    # Cancel/Stop never calls complete(); later node failure must preserve the
    # IDE save, whether the file was new or already present.
    node._rollback_sql_summary_mutations()
    assert target.read_text() == "sql: SELECT 'user saved while pending'\n"
    assert not node._mutated_files


@pytest.mark.parametrize("operation", ["write_file", "edit_file", "delete_file"])
def test_web_summary_rollback_only_reverts_its_last_owned_revision(tmp_path, operation):
    from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

    node = SqlSummaryAgenticNode.__new__(SqlSummaryAgenticNode)
    node.knowledge_base_dir = str(tmp_path / "subject")
    node._mutation_snapshots = {}
    node._mutated_files = set()
    target = tmp_path / "subject/sql_summaries/query.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("sql: SELECT 1\n")
    fs = FilesystemFuncTool(
        root_path=str(tmp_path),
        mutation_guard=node._sql_summary_mutation_guard,
        mutation_callback=node._record_sql_summary_mutation,
        client_mutation_guard=node._validate_sql_summary_mutation_path,
        client_mutation_callback=node._record_sql_summary_client_mutation,
    )
    args = {"path": str(target)}
    if operation == "write_file":
        args["content"] = "sql: SELECT 2\n"
    elif operation == "edit_file":
        args.update(old_string="SELECT 1", new_string="SELECT 2")
    mutation = fs._prepare_client_mutation(operation, args)
    if operation == "delete_file":
        target.unlink()
    else:
        target.write_text("sql: SELECT 2\n")
    assert mutation.complete() is None
    # A later unaccepted card must not take ownership of this subsequent save.
    fs._prepare_client_mutation("write_file", {"path": str(target), "content": "sql: SELECT 3\n"})
    target.write_text("sql: SELECT 'new user save'\n")
    node._rollback_sql_summary_mutations()
    assert target.read_text() == "sql: SELECT 'new user save'\n"
