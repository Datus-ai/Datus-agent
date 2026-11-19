# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Native catalog and MCP selection use actual tool and server contracts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from datus.agent.node.configurable_tools import build_mcp_servers, configurable_tool_catalog, resolve_native_tool_ids
from datus.agent.node.node_factory import create_interactive_node, create_node_input
from datus.cli.execution_state import InteractionCancelled
from datus.schemas.agent_spec import MCPToolSelection
from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
from datus.tools.permission.permission_hooks import PermissionDeniedException
from datus.tools.permission.permission_manager import PermissionManager
from datus.utils.exceptions import DatusException


def _manager(*rules):
    return PermissionManager(
        global_config=PermissionConfig(default_permission=PermissionLevel.ALLOW, rules=list(rules))
    )


def test_native_catalog_resolves_real_method_contracts_without_default_expansion():
    catalog = configurable_tool_catalog()
    assert catalog["filesystem_tools.read_file"]["access"] == "read"
    assert catalog["db_tools.execute_ddl"]["access"] == "write"
    assert resolve_native_tool_ids(["execute_sql", "filesystem_tools.read_file"]) == [
        "db_tools.execute_sql",
        "filesystem_tools.read_file",
    ]
    assert resolve_native_tool_ids([]) == []
    with pytest.raises(DatusException, match="unknown or unavailable"):
        resolve_native_tool_ids(["db_tools.*"])


@pytest.mark.asyncio
@pytest.mark.parametrize("signal", [InteractionCancelled, PermissionDeniedException])
async def test_native_error_boundary_preserves_cancellation_and_permission_denial(
    real_agent_config, monkeypatch, signal
):
    real_agent_config.agentic_nodes["reader"] = {
        "node_class": "configurable",
        "tools": ["filesystem_tools.read_file"],
        "scoped_context": {"project_files": ["docs/**"]},
    }
    node = create_interactive_node("reader", real_agent_config, execution_mode="workflow")
    node.input = create_node_input("Read the project documentation.", node)

    def stop(identity, arguments):
        raise signal("Stopped by the caller")

    try:
        await node._before_stream(SimpleNamespace(user_input=node.input))
        tool = next(item for item in node.tools if item.name == "read_file")
        monkeypatch.setattr(node.execution_scope, "require_tool_scope", stop)
        with pytest.raises(signal, match="Stopped by the caller"):
            await tool.on_invoke_tool(None, "{}")
    finally:
        await node.aclose()


def test_mcp_tool_filter_reaches_the_real_sdk_server_without_mutating_project_connection(real_agent_config):
    connection = {
        "type": "http",
        "url": "http://localhost:1/mcp",
        "tool_filter": {"allowed_tool_names": ["search", "write"]},
    }
    real_agent_config.services.mcp_servers = {"knowledge": deepcopy(connection)}
    selection = MCPToolSelection(connection_ref="knowledge", tools=["search", "write"])
    manager = _manager(PermissionRule(tool="mcp.knowledge", pattern="write", permission=PermissionLevel.DENY))
    servers = build_mcp_servers(real_agent_config, [selection], permission_manager=manager, node_name="research")
    assert servers["knowledge"].tool_filter == {"allowed_tool_names": ["search"]}
    assert real_agent_config.services.mcp_servers["knowledge"] == connection


def test_denying_every_selected_mcp_tool_never_produces_an_unfiltered_server(real_agent_config):
    real_agent_config.services.mcp_servers = {"knowledge": {"type": "http", "url": "http://localhost:1/mcp"}}
    servers = build_mcp_servers(
        real_agent_config,
        [MCPToolSelection(connection_ref="knowledge", tools=["search"])],
        permission_manager=_manager(PermissionRule(tool="mcp.knowledge", pattern="*", permission=PermissionLevel.DENY)),
        node_name="research",
    )
    assert servers == {}


@pytest.mark.parametrize("reference,tools", [("missing", ["search"]), ("knowledge", ["write"])])
def test_mcp_selection_cannot_exceed_the_project_connection(real_agent_config, reference, tools):
    real_agent_config.services.mcp_servers = {
        "knowledge": {
            "type": "http",
            "url": "http://localhost:1/mcp",
            "tool_filter": {"allowed_tool_names": ["search"]},
        }
    }
    with pytest.raises(DatusException, match="MCP"):
        build_mcp_servers(
            real_agent_config,
            [MCPToolSelection(connection_ref=reference, tools=tools)],
            permission_manager=_manager(),
            node_name="research",
        )
