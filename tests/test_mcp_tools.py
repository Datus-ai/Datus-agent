import pytest

from datus.tools.mcp_tools import MCPTool
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
