# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Extended unit tests for datus/tools/mcp_tools/mcp_tool.py"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from datus.tools.mcp_tools.mcp_config import ToolFilterConfig
from datus.tools.mcp_tools.mcp_tool import MCPTool, _parse_header_from_parts, parse_command_string
from datus.utils.exceptions import DatusException, ErrorCode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mcp_tool(tmp_path: Path) -> MCPTool:
    """Create an MCPTool with a mocked MCPManager."""
    mock_manager = MagicMock()
    mock_manager.config_path = tmp_path / "conf" / ".mcp.json"
    with patch("datus.tools.mcp_tools.mcp_tool.MCPManager", return_value=mock_manager):
        tool = MCPTool()
    tool.manager = mock_manager
    return tool


# ---------------------------------------------------------------------------
# _parse_header_from_parts
# ---------------------------------------------------------------------------


class TestParseHeaderFromParts:
    def test_valid_json_object(self):
        result = _parse_header_from_parts(['{"Token": "abc", "X-Key": "val"}'])
        assert result == {"Token": "abc", "X-Key": "val"}

    def test_single_quotes_normalized(self):
        result = _parse_header_from_parts(["{'Token': 'abc'}"])
        assert result.get("Token") == "abc"

    def test_token_level_parse_key_value(self):
        result = _parse_header_from_parts(["Token:", "abc123"])
        assert result.get("Token") == "abc123"

    def test_token_level_parse_inline_value(self):
        result = _parse_header_from_parts(["Token: abc123"])
        assert result.get("Token") == "abc123"

    def test_empty_parts(self):
        result = _parse_header_from_parts([])
        assert result == {}

    def test_braces_stripped(self):
        result = _parse_header_from_parts(["{Token:", "abc}"])
        assert "Token" in result

    def test_comma_separated(self):
        result = _parse_header_from_parts(["Token: abc, Other: xyz"])
        assert result.get("Token") == "abc"
        assert result.get("Other") == "xyz"


# ---------------------------------------------------------------------------
# parse_command_string
# ---------------------------------------------------------------------------


class TestParseCommandString:
    def test_stdio_basic(self):
        cmd = "--transport stdio myserver python -m app"
        t, name, params = parse_command_string(cmd)
        assert t == "stdio"
        assert name == "myserver"
        assert params["command"] == "python"
        assert "-m" in params["args"]
        assert "app" in params["args"]

    def test_stdio_with_env(self):
        cmd = "--transport stdio myserver node server.js --env FOO=bar --env BAR=baz"
        t, name, params = parse_command_string(cmd)
        assert t == "stdio"
        assert params["env"] == {"FOO": "bar", "BAR": "baz"}

    def test_studio_alias(self):
        cmd = "--transport studio myserver python"
        t, name, params = parse_command_string(cmd)
        assert t == "studio"
        assert params["command"] == "python"

    def test_sse_with_url_and_header(self):
        cmd = '--transport sse my-sse https://example.com/stream --header {"Token":"abc"} --timeout 5'
        t, name, params = parse_command_string(cmd)
        assert t == "sse"
        assert name == "my-sse"
        assert params["url"] == "https://example.com/stream"
        assert params["headers"].get("Token") == "abc"
        assert params["timeout"] == 5.0

    def test_http_with_url(self):
        cmd = "--transport http my-http https://api.example.com/mcp"
        t, name, params = parse_command_string(cmd)
        assert t == "http"
        assert params["url"] == "https://api.example.com/mcp"

    def test_sse_default_timeout(self):
        cmd = "--transport sse my-sse https://example.com"
        t, name, params = parse_command_string(cmd)
        assert params["timeout"] == 10.0

    def test_sse_url_fallback_http_token(self):
        cmd = "--transport sse my-sse --url https://api.example.com"
        t, name, params = parse_command_string(cmd)
        assert params["url"] == "https://api.example.com"

    def test_sse_url_fallback_bare_https(self):
        cmd = "--transport sse my-sse https://bare.example.com"
        t, name, params = parse_command_string(cmd)
        assert "bare.example.com" in params["url"]

    def test_no_transport_raises(self):
        with pytest.raises(DatusException) as exc_info:
            parse_command_string("python -m app")
        assert exc_info.value.code == ErrorCode.COMMON_FIELD_INVALID

    def test_unsupported_transport_raises(self):
        with pytest.raises(DatusException, match="Unsupported transport"):
            parse_command_string("--transport grpc my-server python")

    def test_timeout_parsed_for_sse(self):
        cmd = "--transport sse my-sse https://example.com --timeout 30"
        t, name, params = parse_command_string(cmd)
        assert params["timeout"] == 30.0

    def test_args_between_command_and_env(self):
        cmd = "--transport stdio myserver python -m datus.main arg1 arg2 --env KEY=VAL"
        t, name, params = parse_command_string(cmd)
        assert "arg1" in params["args"]
        assert "arg2" in params["args"]
        assert "KEY" in params["env"]

    def test_stdio_no_name(self):
        # Only one token after --transport (the transport type itself)
        cmd = "--transport stdio python"
        t, name, params = parse_command_string(cmd)
        assert t == "stdio"
        # name == "python" or None depending on token count
        assert params["command"] is not None


# ---------------------------------------------------------------------------
# MCPTool - wraps MCPManager methods
# ---------------------------------------------------------------------------


class TestMCPToolAddServer:
    def test_add_server_success(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.add_server.return_value = (True, "Successfully added server 'srv'")
        result = tool.add_server(name="srv", type="stdio", command="python")
        assert result.success is True

    def test_add_server_failure(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.add_server.return_value = (False, "Server 'srv' already exists")
        result = tool.add_server(name="srv", type="stdio", command="python")
        assert result.success is False
        assert "already exists" in result.message

    def test_add_server_exception(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.add_server.side_effect = Exception("unexpected")
        result = tool.add_server(name="srv", type="stdio", command="python")
        assert result.success is False

    def test_add_server_invalid_type(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        result = tool.add_server(name="srv", type="grpc")
        assert result.success is False


class TestMCPToolRemoveServer:
    def test_remove_server_success(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.remove_server.return_value = (True, "Successfully removed server 'srv'")
        result = tool.remove_server("srv")
        assert result.success is True
        assert result.result["removed_server"] == "srv"

    def test_remove_server_not_found(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.remove_server.return_value = (False, "Server 'srv' not found")
        result = tool.remove_server("srv")
        assert result.success is False

    def test_remove_server_exception(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.remove_server.side_effect = Exception("error")
        result = tool.remove_server("srv")
        assert result.success is False


class TestMCPToolListServers:
    def test_list_servers_empty(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.list_servers.return_value = []
        result = tool.list_servers()
        assert result.success is True
        assert result.result["total_count"] == 0

    def test_list_servers_with_data(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        mock_srv = MagicMock()
        mock_srv.model_dump.return_value = {"name": "srv", "type": "stdio"}
        tool.manager.list_servers.return_value = [mock_srv]
        result = tool.list_servers()
        assert result.success is True
        assert result.result["total_count"] == 1

    def test_list_servers_with_type_filter(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.list_servers.return_value = []
        tool.list_servers(server_type="stdio")
        tool.manager.list_servers.assert_called_once_with(server_type="stdio")

    def test_list_servers_exception(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.list_servers.side_effect = Exception("fail")
        result = tool.list_servers()
        assert result.success is False


class TestMCPToolGetServer:
    def test_get_server_found(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        mock_cfg = MagicMock()
        mock_cfg.model_dump.return_value = {"name": "srv", "type": "stdio"}
        tool.manager.get_server_config.return_value = mock_cfg
        result = tool.get_server("srv")
        assert result.success is True
        assert result.result["name"] == "srv"

    def test_get_server_not_found(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.get_server_config.return_value = None
        result = tool.get_server("nonexistent")
        assert result.success is False
        assert "not found" in result.message

    def test_get_server_exception(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.get_server_config.side_effect = Exception("err")
        result = tool.get_server("srv")
        assert result.success is False


class TestMCPToolCheckConnectivity:
    def test_check_connectivity_success(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        with patch("datus.tools.mcp_tools.mcp_tool.run_async", return_value=(True, "OK", {"connected": True})):
            result = tool.check_connectivity("srv")
        assert result.success is True
        assert result.result["connectivity"] is True

    def test_check_connectivity_failure(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        with patch("datus.tools.mcp_tools.mcp_tool.run_async", return_value=(False, "Connection refused", {})):
            result = tool.check_connectivity("srv")
        assert result.success is False

    def test_check_connectivity_exception(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        with patch("datus.tools.mcp_tools.mcp_tool.run_async", side_effect=Exception("timeout")):
            result = tool.check_connectivity("srv")
        assert result.success is False


class TestMCPToolListTools:
    def test_list_tools_success(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tools_list = [{"name": "read_file"}, {"name": "write_file"}]
        with patch(
            "datus.tools.mcp_tools.mcp_tool.run_async",
            return_value=(True, "Found 2 tools", tools_list),
        ):
            result = tool.list_tools("srv")
        assert result.success is True
        assert result.result["tools_count"] == 2
        assert result.result["filtered"] is True

    def test_list_tools_failure(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        with patch("datus.tools.mcp_tools.mcp_tool.run_async", return_value=(False, "Error", [])):
            result = tool.list_tools("srv")
        assert result.success is False
        assert result.result["tools_count"] == 0

    def test_list_tools_no_filter(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        with patch(
            "datus.tools.mcp_tools.mcp_tool.run_async",
            return_value=(True, "OK", [{"name": "t1"}]),
        ):
            result = tool.list_tools("srv", apply_filter=False)
        assert result.result["filtered"] is False

    def test_list_filtered_tools_delegates(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        with patch.object(tool, "list_tools", return_value=MagicMock(success=True)) as mock_list:
            tool.list_filtered_tools("srv")
        mock_list.assert_called_once_with("srv", apply_filter=True)

    def test_list_tools_exception(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        with patch("datus.tools.mcp_tools.mcp_tool.run_async", side_effect=Exception("boom")):
            result = tool.list_tools("srv")
        assert result.success is False


class TestMCPToolCallTool:
    def test_call_tool_success(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        result_data = {"content": [{"type": "text", "text": "ok"}], "isError": False}
        with patch("datus.tools.mcp_tools.mcp_tool.run_async", return_value=(True, "OK", result_data)):
            result = tool.call_tool("srv", "my_tool", {"arg": "val"})
        assert result.success is True
        assert result.result["tool_name"] == "my_tool"

    def test_call_tool_failure(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        with patch("datus.tools.mcp_tools.mcp_tool.run_async", return_value=(False, "Error", {})):
            result = tool.call_tool("srv", "my_tool", {})
        assert result.success is False

    def test_call_tool_none_arguments(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        with patch("datus.tools.mcp_tools.mcp_tool.run_async", return_value=(True, "OK", {})):
            result = tool.call_tool("srv", "my_tool", None)
        assert result.success is True

    def test_call_tool_exception(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        with patch("datus.tools.mcp_tools.mcp_tool.run_async", side_effect=Exception("crash")):
            result = tool.call_tool("srv", "my_tool", {})
        assert result.success is False


class TestMCPToolFilterMethods:
    def test_set_tool_filter_success(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.set_tool_filter.return_value = (True, "Updated")
        result = tool.set_tool_filter("srv", allowed_tools=["read"], blocked_tools=None, enabled=True)
        assert result.success is True
        assert result.result["server_name"] == "srv"

    def test_set_tool_filter_failure(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.set_tool_filter.return_value = (False, "Server 'srv' not found")
        result = tool.set_tool_filter("srv", allowed_tools=["read"])
        assert result.success is False
        assert result.result is None

    def test_set_tool_filter_exception(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.set_tool_filter.side_effect = Exception("err")
        result = tool.set_tool_filter("srv")
        assert result.success is False

    def test_get_tool_filter_with_filter(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tf = ToolFilterConfig(allowed_tool_names=["read"])
        tool.manager.get_tool_filter.return_value = (True, "OK", tf)
        result = tool.get_tool_filter("srv")
        assert result.success is True
        assert result.result["has_filter"] is True
        assert result.result["filter_config"] is not None

    def test_get_tool_filter_no_filter(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.get_tool_filter.return_value = (True, "OK", None)
        result = tool.get_tool_filter("srv")
        assert result.success is True
        assert result.result["has_filter"] is False
        assert result.result["filter_config"] is None

    def test_get_tool_filter_failure(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.get_tool_filter.return_value = (False, "Not found", None)
        result = tool.get_tool_filter("srv")
        assert result.success is False
        assert result.result is None

    def test_get_tool_filter_exception(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.get_tool_filter.side_effect = Exception("crash")
        result = tool.get_tool_filter("srv")
        assert result.success is False

    def test_remove_tool_filter_success(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.set_tool_filter.return_value = (True, "Updated")
        result = tool.remove_tool_filter("srv")
        assert result.success is True
        assert result.result["filter_removed"] is True

    def test_remove_tool_filter_failure(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.set_tool_filter.return_value = (False, "Not found")
        result = tool.remove_tool_filter("srv")
        assert result.success is False
        assert result.result is None

    def test_remove_tool_filter_exception(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager.set_tool_filter.side_effect = Exception("boom")
        result = tool.remove_tool_filter("srv")
        assert result.success is False


class TestMCPToolCleanup:
    def test_cleanup_calls_manager_cleanup(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.cleanup()
        tool.manager.cleanup.assert_called_once()

    def test_cleanup_none_manager(self, tmp_path):
        tool = _make_mcp_tool(tmp_path)
        tool.manager = None
        # Should not raise
        tool.cleanup()
