# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/mcp_tools/mcp_manager.py"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.tools.mcp_tools.mcp_config import MCPServerType, SSEServerConfig, STDIOServerConfig, ToolFilterConfig
from datus.tools.mcp_tools.mcp_manager import MCPManager, _validate_server_exists, create_static_tool_filter

# ---------------------------------------------------------------------------
# Helper: build a manager bypassing __init__ path manager calls
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path) -> MCPManager:
    """Create MCPManager with config in tmp_path, bypassing real path_manager."""
    mock_path_manager = MagicMock()
    config_file = tmp_path / "conf" / ".mcp.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    mock_path_manager.mcp_config_path.return_value = config_file
    mock_path_manager.ensure_dirs.return_value = None

    with patch("datus.utils.path_manager.get_path_manager", return_value=mock_path_manager):
        manager = MCPManager()

    return manager


# ---------------------------------------------------------------------------
# create_static_tool_filter
# ---------------------------------------------------------------------------


class TestCreateStaticToolFilter:
    def test_creates_with_allowlist(self):
        tf = create_static_tool_filter(allowed_tool_names=["read", "write"])
        assert tf.allowed_tool_names == ["read", "write"]
        assert tf.blocked_tool_names is None
        assert tf.enabled is True

    def test_creates_with_blocklist(self):
        tf = create_static_tool_filter(blocked_tool_names=["delete"])
        assert tf.blocked_tool_names == ["delete"]
        assert tf.allowed_tool_names is None

    def test_creates_disabled(self):
        tf = create_static_tool_filter(enabled=False)
        assert tf.enabled is False

    def test_filter_is_toolfilterconfig(self):
        tf = create_static_tool_filter()
        assert isinstance(tf, ToolFilterConfig)


# ---------------------------------------------------------------------------
# _validate_server_exists
# ---------------------------------------------------------------------------


class TestValidateServerExists:
    def test_server_exists(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config.add_server(STDIOServerConfig(name="test", command="python"))
        valid, msg, cfg = _validate_server_exists(manager, "test")
        assert valid is True
        assert msg == ""
        assert cfg is not None

    def test_server_not_exists(self, tmp_path):
        manager = _make_manager(tmp_path)
        valid, msg, cfg = _validate_server_exists(manager, "nonexistent")
        assert valid is False
        assert "not found" in msg
        assert cfg is None


class TestMCPManagerInit:
    def test_init_uses_explicit_path_manager(self, tmp_path):
        config_file = tmp_path / "conf" / ".mcp.json"
        path_manager = MagicMock()
        path_manager.mcp_config_path.return_value = config_file

        manager = MCPManager(path_manager=path_manager)

        path_manager.ensure_dirs.assert_called_once_with("conf")
        assert manager.config_path == config_file

    def test_init_uses_agent_config_path_manager(self, tmp_path):
        config_file = tmp_path / "conf" / ".mcp.json"
        path_manager = MagicMock()
        path_manager.mcp_config_path.return_value = config_file
        agent_config = SimpleNamespace(path_manager=path_manager)

        manager = MCPManager(agent_config=agent_config)

        path_manager.ensure_dirs.assert_called_once_with("conf")
        assert manager.config_path == config_file


# ---------------------------------------------------------------------------
# MCPManager - config CRUD
# ---------------------------------------------------------------------------


class TestMCPManagerCRUD:
    def test_add_server_success(self, tmp_path):
        manager = _make_manager(tmp_path)
        srv = STDIOServerConfig(name="my-srv", command="python")
        success, msg = manager.add_server(srv)
        assert success is True
        assert "my-srv" in manager.config.servers

    def test_add_server_duplicate_fails(self, tmp_path):
        manager = _make_manager(tmp_path)
        srv = STDIOServerConfig(name="dup", command="python")
        manager.add_server(srv)
        success, msg = manager.add_server(srv)
        assert success is False
        assert "already exists" in msg

    def test_remove_server_success(self, tmp_path):
        manager = _make_manager(tmp_path)
        srv = STDIOServerConfig(name="to-remove", command="python")
        manager.add_server(srv)
        success, msg = manager.remove_server("to-remove")
        assert success is True
        assert "to-remove" not in manager.config.servers

    def test_remove_server_not_found(self, tmp_path):
        manager = _make_manager(tmp_path)
        success, msg = manager.remove_server("nonexistent")
        assert success is False
        assert "not found" in msg

    def test_list_servers_empty(self, tmp_path):
        manager = _make_manager(tmp_path)
        assert manager.list_servers() == []

    def test_list_servers_all(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config.add_server(STDIOServerConfig(name="s1", command="python"))
        manager.config.add_server(SSEServerConfig(name="s2", url="http://example.com"))
        servers = manager.list_servers()
        assert len(servers) == 2

    def test_list_servers_filtered(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config.add_server(STDIOServerConfig(name="s1", command="python"))
        manager.config.add_server(SSEServerConfig(name="s2", url="http://example.com"))
        stdio_servers = manager.list_servers(server_type=MCPServerType.STDIO)
        assert len(stdio_servers) == 1
        assert stdio_servers[0].name == "s1"

    def test_get_server_config_existing(self, tmp_path):
        manager = _make_manager(tmp_path)
        srv = STDIOServerConfig(name="srv", command="echo")
        manager.config.add_server(srv)
        result = manager.get_server_config("srv")
        assert result is not None
        assert result.name == "srv"

    def test_get_server_config_missing(self, tmp_path):
        manager = _make_manager(tmp_path)
        assert manager.get_server_config("missing") is None


# ---------------------------------------------------------------------------
# MCPManager - save/load config
# ---------------------------------------------------------------------------


class TestMCPManagerPersistence:
    def test_save_and_load_config(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config.add_server(STDIOServerConfig(name="persist-srv", command="node"))
        saved = manager.save_config()
        assert saved is True
        assert manager.config_path.exists()

        # Load it back with a fresh manager pointing to same file
        manager2 = _make_manager(tmp_path)
        assert "persist-srv" in manager2.config.servers

    def test_load_config_file_missing(self, tmp_path):
        manager = _make_manager(tmp_path)
        # Remove config file
        if manager.config_path.exists():
            manager.config_path.unlink()
        result = manager.load_config()
        assert result is True

    def test_load_config_empty_file(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config_path.parent.mkdir(parents=True, exist_ok=True)
        manager.config_path.write_text("")
        result = manager.load_config()
        assert result is True

    def test_load_config_invalid_format(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config_path.write_text(json.dumps({"otherKey": {}}))
        result = manager.load_config()
        assert result is True
        assert manager.config.servers == {}

    def test_save_config_failure(self, tmp_path):
        manager = _make_manager(tmp_path)
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = manager.save_config()
        assert result is False


# ---------------------------------------------------------------------------
# MCPManager - tool filter management
# ---------------------------------------------------------------------------


class TestMCPManagerToolFilter:
    def test_set_tool_filter_success(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config.add_server(STDIOServerConfig(name="srv", command="python"))
        tf = ToolFilterConfig(allowed_tool_names=["read"])
        success, msg = manager.set_tool_filter("srv", tf)
        assert success is True
        assert manager.config.servers["srv"].tool_filter is tf

    def test_set_tool_filter_server_not_found(self, tmp_path):
        manager = _make_manager(tmp_path)
        tf = ToolFilterConfig(allowed_tool_names=["read"])
        success, msg = manager.set_tool_filter("nonexistent", tf)
        assert success is False
        assert "not found" in msg

    def test_get_tool_filter_existing(self, tmp_path):
        manager = _make_manager(tmp_path)
        tf = ToolFilterConfig(allowed_tool_names=["write"])
        manager.config.add_server(STDIOServerConfig(name="srv", command="python", tool_filter=tf))
        success, msg, result = manager.get_tool_filter("srv")
        assert success is True
        assert result is tf

    def test_get_tool_filter_server_not_found(self, tmp_path):
        manager = _make_manager(tmp_path)
        success, msg, result = manager.get_tool_filter("missing")
        assert success is False
        assert result is None

    def test_get_tool_filter_no_filter_set(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config.add_server(STDIOServerConfig(name="srv", command="python"))
        success, msg, result = manager.get_tool_filter("srv")
        assert success is True
        assert result is None


# ---------------------------------------------------------------------------
# MCPManager - _create_server_instance
# ---------------------------------------------------------------------------


class TestCreateServerInstance:
    def test_create_stdio_server(self, tmp_path):
        manager = _make_manager(tmp_path)
        cfg = STDIOServerConfig(name="stdio", command="python", args=["-m", "app"])
        with patch("datus.tools.mcp_tools.mcp_manager.SilentMCPServerStdio") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            instance, details = manager._create_server_instance(cfg)
        assert instance is mock_instance
        assert details["command"] == "python"

    def test_create_sse_server_missing_url(self, tmp_path):
        manager = _make_manager(tmp_path)
        # Use _create_sse_server directly with empty url
        instance, details = manager._create_sse_server({"url": None})
        assert instance is None
        assert "error" in details

    def test_create_http_server_missing_url(self, tmp_path):
        manager = _make_manager(tmp_path)
        instance, details = manager._create_http_server({"url": None})
        assert instance is None
        assert "error" in details

    def test_create_sse_server_with_url(self, tmp_path):
        manager = _make_manager(tmp_path)
        with patch("datus.tools.mcp_tools.mcp_manager.MCPServerSse") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            with patch("datus.tools.mcp_tools.mcp_manager.MCPServerSseParams"):
                instance, details = manager._create_sse_server({"url": "http://example.com"})
        assert instance is not None

    def test_create_http_server_with_url(self, tmp_path):
        manager = _make_manager(tmp_path)
        with patch("datus.tools.mcp_tools.mcp_manager.MCPServerStreamableHttp") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            with patch("datus.tools.mcp_tools.mcp_manager.MCPServerStreamableHttpParams"):
                instance, details = manager._create_http_server({"url": "http://example.com"})
        assert instance is not None

    def test_create_server_instance_unsupported_type(self, tmp_path):
        manager = _make_manager(tmp_path)
        cfg = STDIOServerConfig(name="s", command="echo")
        # Patch cfg.type to return an unexpected string via object attribute
        try:
            object.__setattr__(cfg, "type", "unsupported_type")
        except Exception:
            pytest.skip("Cannot override pydantic field for this test")
        instance, details = manager._create_server_instance(cfg)
        assert instance is None


# ---------------------------------------------------------------------------
# MCPManager - async operations (mocked server instances)
# ---------------------------------------------------------------------------


class TestMCPManagerAsync:
    @pytest.mark.asyncio
    async def test_check_connectivity_server_not_found(self, tmp_path):
        manager = _make_manager(tmp_path)
        success, msg, details = await manager.check_connectivity("nonexistent")
        assert success is False
        assert "not found" in msg

    @pytest.mark.asyncio
    async def test_check_connectivity_instance_creation_fails(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config.add_server(STDIOServerConfig(name="srv", command="python"))
        with patch.object(manager, "_create_server_instance", return_value=(None, {"error": "fail"})):
            success, msg, details = await manager.check_connectivity("srv")
        assert success is False
        assert "fail" in msg

    @pytest.mark.asyncio
    async def test_list_tools_server_not_found(self, tmp_path):
        manager = _make_manager(tmp_path)
        success, msg, tools = await manager.list_tools("nonexistent")
        assert success is False
        assert tools == []

    @pytest.mark.asyncio
    async def test_list_tools_instance_creation_fails(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config.add_server(STDIOServerConfig(name="srv", command="python"))
        with patch.object(manager, "_create_server_instance", return_value=(None, {"error": "can't connect"})):
            success, msg, tools = await manager.list_tools("srv")
        assert success is False
        assert tools == []

    @pytest.mark.asyncio
    async def test_list_tools_success_with_filter(self, tmp_path):
        manager = _make_manager(tmp_path)
        tf = ToolFilterConfig(allowed_tool_names=["read"])
        manager.config.add_server(STDIOServerConfig(name="srv", command="python", tool_filter=tf))
        mock_server = MagicMock()
        with patch.object(manager, "_create_server_instance", return_value=(mock_server, {})):
            with patch.object(
                manager,
                "_run_tools_operation_async",
                return_value=(True, {"tools": [{"name": "read"}, {"name": "write"}]}),
            ):
                success, msg, tools = await manager.list_tools("srv", apply_filter=True)
        assert success is True
        # Only "read" passes filter
        assert len(tools) == 1
        assert tools[0]["name"] == "read"

    @pytest.mark.asyncio
    async def test_list_tools_no_filter_applied(self, tmp_path):
        manager = _make_manager(tmp_path)
        tf = ToolFilterConfig(allowed_tool_names=["read"])
        manager.config.add_server(STDIOServerConfig(name="srv", command="python", tool_filter=tf))
        mock_server = MagicMock()
        with patch.object(manager, "_create_server_instance", return_value=(mock_server, {})):
            with patch.object(
                manager,
                "_run_tools_operation_async",
                return_value=(True, {"tools": [{"name": "read"}, {"name": "write"}]}),
            ):
                success, msg, tools = await manager.list_tools("srv", apply_filter=False)
        assert success is True
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_call_tool_server_not_found(self, tmp_path):
        manager = _make_manager(tmp_path)
        success, msg, data = await manager.call_tool("nonexistent", "my_tool", {})
        assert success is False

    @pytest.mark.asyncio
    async def test_call_tool_blocked_by_filter(self, tmp_path):
        manager = _make_manager(tmp_path)
        tf = ToolFilterConfig(blocked_tool_names=["dangerous"])
        manager.config.add_server(STDIOServerConfig(name="srv", command="python", tool_filter=tf))
        success, msg, data = await manager.call_tool("srv", "dangerous", {})
        assert success is False
        assert "blocked" in msg

    @pytest.mark.asyncio
    async def test_call_tool_instance_creation_fails(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config.add_server(STDIOServerConfig(name="srv", command="python"))
        with patch.object(manager, "_create_server_instance", return_value=(None, {"error": "no server"})):
            success, msg, data = await manager.call_tool("srv", "my_tool", {})
        assert success is False

    @pytest.mark.asyncio
    async def test_call_tool_success(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config.add_server(STDIOServerConfig(name="srv", command="python"))
        mock_server = MagicMock()
        result_data = {"content": [{"type": "text", "text": "result"}], "isError": False}
        with patch.object(manager, "_create_server_instance", return_value=(mock_server, {})):
            with patch.object(manager, "_run_tools_operation_async", return_value=(True, result_data)):
                success, msg, data = await manager.call_tool("srv", "my_tool", {"arg": "val"})
        assert success is True
        assert data == result_data

    @pytest.mark.asyncio
    async def test_list_filtered_tools_delegates(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.config.add_server(STDIOServerConfig(name="srv", command="python"))
        mock_server = MagicMock()
        with patch.object(manager, "_create_server_instance", return_value=(mock_server, {})):
            with patch.object(manager, "_run_tools_operation_async", return_value=(True, {"tools": [{"name": "t1"}]})):
                success, msg, tools = await manager.list_filtered_tools("srv")
        assert success is True


# ---------------------------------------------------------------------------
# MCPManager - _dispatch_operation and handlers
# ---------------------------------------------------------------------------


class TestDispatchOperation:
    @pytest.mark.asyncio
    async def test_dispatch_unknown_operation(self, tmp_path):
        manager = _make_manager(tmp_path)
        server_instance = MagicMock()
        agent = MagicMock()
        run_context = MagicMock()
        success, data = await manager._dispatch_operation(server_instance, "srv", "unknown_op", agent, run_context)
        assert success is False
        assert "Unknown operation" in data["error"]

    @pytest.mark.asyncio
    async def test_handle_list_tools(self, tmp_path):
        manager = _make_manager(tmp_path)
        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read a file"
        mock_tool.inputSchema = {"type": "object"}
        server_instance = AsyncMock()
        server_instance.list_tools.return_value = [mock_tool]
        agent = MagicMock()
        run_context = MagicMock()
        success, data = await manager._handle_list_tools(server_instance, "srv", run_context, agent)
        assert success is True
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_handle_list_tools_empty(self, tmp_path):
        manager = _make_manager(tmp_path)
        server_instance = AsyncMock()
        server_instance.list_tools.return_value = []
        agent = MagicMock()
        run_context = MagicMock()
        success, data = await manager._handle_list_tools(server_instance, "srv", run_context, agent)
        assert success is True
        assert data["tools"] == []

    @pytest.mark.asyncio
    async def test_handle_list_tools_with_model_dump_schema(self, tmp_path):
        manager = _make_manager(tmp_path)
        mock_tool = MagicMock()
        mock_tool.name = "my_tool"
        mock_tool.description = "desc"
        # inputSchema has model_dump
        schema_mock = MagicMock()
        schema_mock.model_dump.return_value = {"type": "object", "properties": {}}
        mock_tool.inputSchema = schema_mock
        server_instance = AsyncMock()
        server_instance.list_tools.return_value = [mock_tool]
        success, data = await manager._handle_list_tools(server_instance, "srv", MagicMock(), MagicMock())
        assert data["tools"][0]["inputSchema"] == {"type": "object", "properties": {}}

    @pytest.mark.asyncio
    async def test_handle_call_tool_no_name(self, tmp_path):
        manager = _make_manager(tmp_path)
        server_instance = AsyncMock()
        success, data = await manager._handle_call_tool(server_instance, "srv")
        assert success is False
        assert "required" in data["error"]

    @pytest.mark.asyncio
    async def test_handle_call_tool_success(self, tmp_path):
        manager = _make_manager(tmp_path)
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = "hello"
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [mock_content]
        server_instance = AsyncMock()
        server_instance.call_tool.return_value = mock_result
        success, data = await manager._handle_call_tool(server_instance, "srv", tool_name="my_tool", arguments={"a": 1})
        assert success is True
        assert data["isError"] is False
        assert data["content"][0]["text"] == "hello"

    @pytest.mark.asyncio
    async def test_handle_call_tool_data_content(self, tmp_path):
        manager = _make_manager(tmp_path)
        mock_content = MagicMock(spec=["type", "data"])
        mock_content.type = "image"
        mock_content.data = b"imagedata"
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [mock_content]
        server_instance = AsyncMock()
        server_instance.call_tool.return_value = mock_result
        success, data = await manager._handle_call_tool(server_instance, "srv", tool_name="screenshot", arguments={})
        assert success is True
        assert data["content"][0]["data"] == b"imagedata"

    @pytest.mark.asyncio
    async def test_handle_connectivity_test(self, tmp_path):
        manager = _make_manager(tmp_path)
        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        server_instance = AsyncMock()
        server_instance.list_tools.return_value = [mock_tool]
        agent = MagicMock()
        run_context = MagicMock()
        success, data = await manager._handle_connectivity_test(server_instance, "srv", run_context, agent)
        assert success is True
        assert data["connected"] is True
        assert data["tool_count"] == 1

    @pytest.mark.asyncio
    async def test_handle_connectivity_test_list_tools_fails(self, tmp_path):
        manager = _make_manager(tmp_path)
        server_instance = AsyncMock()
        server_instance.list_tools.side_effect = Exception("timeout")
        agent = MagicMock()
        run_context = MagicMock()
        success, data = await manager._handle_connectivity_test(server_instance, "srv", run_context, agent)
        assert success is True
        assert data["tools_available"] is False

    @pytest.mark.asyncio
    async def test_handle_connectivity_test_no_list_tools(self, tmp_path):
        manager = _make_manager(tmp_path)
        # Server instance without list_tools attribute
        server_instance = MagicMock(spec=[])
        success, data = await manager._handle_connectivity_test(server_instance, "srv", MagicMock(), MagicMock())
        assert success is True
        assert data["tools_available"] is False


# ---------------------------------------------------------------------------
# MCPManager - _cleanup_server_instance
# ---------------------------------------------------------------------------


class TestCleanupServerInstance:
    @pytest.mark.asyncio
    async def test_cleanup_with_cleanup_method(self, tmp_path):
        manager = _make_manager(tmp_path)
        mock_srv = AsyncMock()
        await manager._cleanup_server_instance(mock_srv)
        mock_srv.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_disconnect_method(self, tmp_path):
        manager = _make_manager(tmp_path)
        mock_srv = AsyncMock(spec=["disconnect"])
        await manager._cleanup_server_instance(mock_srv)
        mock_srv.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_close_method(self, tmp_path):
        manager = _make_manager(tmp_path)
        mock_srv = AsyncMock(spec=["close"])
        await manager._cleanup_server_instance(mock_srv)
        mock_srv.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_exception_suppressed(self, tmp_path):
        manager = _make_manager(tmp_path)
        mock_srv = AsyncMock()
        mock_srv.cleanup.side_effect = Exception("cleanup failed")
        # Should not raise
        await manager._cleanup_server_instance(mock_srv)


# ---------------------------------------------------------------------------
# MCPManager - cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_manager(self, tmp_path):
        manager = _make_manager(tmp_path)
        # Should not raise
        manager.cleanup()
