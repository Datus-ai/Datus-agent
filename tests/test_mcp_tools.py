import pytest

from datus.tools.mcp_tools import MCPTool
from datus.tools.mcp_tools.mcp_config import ToolFilterConfig
from datus.tools.mcp_tools.mcp_manager import create_static_tool_filter
from datus.tools.mcp_tools.mcp_tool import parse_command_string
from datus.utils.exceptions import DatusException, ErrorCode


def test_tools():
    tool = MCPTool()
    server_result = tool.list_servers()
    assert server_result.success
    servers = server_result.result["servers"]

    assert len(servers) > 0
    for server in servers:
        print(server["name"], server["type"])
        con_result = tool.check_connectivity(server["name"])
        print(server["name"], "connect status:", con_result.success, "; error=", con_result.message)
        if con_result.success:
            print("tools:", tool.list_tools(server["name"]))


def test_parse_cmd():
    cmd = '--transport sse my-sse https://example.com/stream --header {"Token":"abc"} --timeout 5'
    transport_type, name, params = parse_command_string(cmd)
    assert transport_type == "sse"
    assert name == "my-sse"
    assert params == {
        "url": "https://example.com/stream",
        "headers": {"Token": "abc"},
        "timeout": 5,
    }

    cmd = (
        "--transport stdio my-studio python -m datus.main --directory foo run svc -abc"
        " --env DEBUG=1 --env a=b --timeout 5"
    )
    transport_type, name, params = parse_command_string(cmd)
    print(params)
    assert transport_type == "stdio"
    assert name == "my-studio"
    assert params == {
        "command": "python",
        "args": ["-m", "datus.main", "--directory", "foo", "run", "svc", "-abc"],
        "env": {"DEBUG": "1", "a": "b"},
    }

    with pytest.raises(DatusException, match="Unsupported transport protocols") as exc_info:
        parse_command_string(
            "--transport no_type my-studio python -m datus.main --directory foo run svc -abc"
            " --env DEBUG=1 --env a=b --timeout 5 --invalid-param"
        )
    assert exc_info.value.code == ErrorCode.COMMON_FIELD_INVALID


def test_tool_filtering():
    """Test tool filtering functionality."""
    # Test creating static tool filters
    allowlist_filter = create_static_tool_filter(allowed_tool_names=["read_file", "write_file", "list_directory"])
    assert allowlist_filter.allowed_tool_names == ["read_file", "write_file", "list_directory"]
    assert allowlist_filter.blocked_tool_names is None
    assert allowlist_filter.enabled is True

    blocklist_filter = create_static_tool_filter(blocked_tool_names=["delete_file", "execute_command"])
    assert blocklist_filter.allowed_tool_names is None
    assert blocklist_filter.blocked_tool_names == ["delete_file", "execute_command"]
    assert blocklist_filter.enabled is True

    # Test filter logic
    assert allowlist_filter.is_tool_allowed("read_file") is True
    assert allowlist_filter.is_tool_allowed("delete_file") is False
    assert allowlist_filter.is_tool_allowed("write_file") is True

    assert blocklist_filter.is_tool_allowed("read_file") is True
    assert blocklist_filter.is_tool_allowed("delete_file") is False
    assert blocklist_filter.is_tool_allowed("execute_command") is False

    # Test disabled filter
    disabled_filter = create_static_tool_filter(blocked_tool_names=["everything"], enabled=False)
    assert disabled_filter.is_tool_allowed("everything") is True  # Should be allowed when disabled

    # Test filter configuration model
    filter_config = ToolFilterConfig(allowed_tool_names=["tool1", "tool2"], blocked_tool_names=["tool3"], enabled=True)

    # Serialize and deserialize
    filter_dict = filter_config.model_dump()
    restored_filter = ToolFilterConfig(**filter_dict)

    assert restored_filter.allowed_tool_names == ["tool1", "tool2"]
    assert restored_filter.blocked_tool_names == ["tool3"]
    assert restored_filter.enabled is True

    # Test filtering logic on restored filter
    assert restored_filter.is_tool_allowed("tool1") is True
    assert restored_filter.is_tool_allowed("tool3") is False  # blocked takes precedence
    assert restored_filter.is_tool_allowed("tool4") is False  # not in allowlist


def test_mcp_tool_filter_methods():
    """Test MCPTool filter management methods."""
    tool = MCPTool()

    # Test filter methods on non-existent server (should fail gracefully)
    result = tool.get_tool_filter("non_existent_server")
    assert result.success is False
    assert "not found" in result.message.lower()

    result = tool.set_tool_filter("non_existent_server", ["tool1"], None, True)
    assert result.success is False
    assert "not found" in result.message.lower()

    result = tool.remove_tool_filter("non_existent_server")
    assert result.success is False
    assert "not found" in result.message.lower()

    # Test list_tools with filtering parameter
    servers = tool.list_servers()
    if servers.success and servers.result["servers"]:
        server_name = servers.result["servers"][0]["name"]

        # Test listing tools with filtering enabled
        result_filtered = tool.list_tools(server_name, apply_filter=True)
        assert "filtered" in result_filtered.result
        assert result_filtered.result["filtered"] is True

        # Test listing tools with filtering disabled
        result_unfiltered = tool.list_tools(server_name, apply_filter=False)
        assert "filtered" in result_unfiltered.result
        assert result_unfiltered.result["filtered"] is False
