"""A new tool cannot silently ship without a restricted-host access contract."""

import pytest

from datus.tools.registry.capabilities import TOOL_CAPABILITIES, tool_capability


@pytest.mark.acceptance
def test_generated_catalog_matches_source():
    """Every PR validates all native declarations, regardless of changed-file selection."""
    import json
    from pathlib import Path

    from datus.tools.registry.catalog_builder import collect_contracts

    package = Path(__file__).resolve().parents[4] / "datus"
    committed = json.loads((package / "tools/registry/capabilities.json").read_text())
    assert collect_contracts(package) == committed


def test_every_configurable_agent_tool_declares_a_capability():
    """Require capability declarations for every configurable family method."""
    from datus.api.services.agent_service import VALID_TOOL_METHODS

    missing = [
        f"{category}.{name}"
        for category, names in VALID_TOOL_METHODS.items()
        for name in names
        if tool_capability(f"{category}.{name}") is None
    ]
    assert not missing, f"Declare capabilities for new tools: {sorted(missing)}"


def test_every_mcp_tool_declares_a_capability():
    """Check registered MCP methods against their canonical family contracts."""
    import datus.mcp_server  # noqa: F401
    from datus.utils.mcp_decorators import get_mcp_tools, get_tool_registry

    missing = [
        f"{cfg.tool_class.permission_category}.{name}"
        for cfg in get_tool_registry()
        for name, _, _ in get_mcp_tools(cfg.tool_class)
        if tool_capability(f"{cfg.tool_class.permission_category}.{name}") is None
    ]
    assert not missing, f"Declare capabilities for new MCP tools: {sorted(missing)}"


def test_companions_have_existing_owners_and_no_cycles():
    """Ensure paging companions share access with an existing, non-companion owner."""
    companions = [(tool_id, capability) for tool_id, capability in TOOL_CAPABILITIES.items() if capability.companion_of]
    assert companions, "Exercise at least one real paging contract"
    owners = [(tool_id, capability, TOOL_CAPABILITIES[capability.companion_of]) for tool_id, capability in companions]
    assert all(owner.companion_of is None for _, _, owner in owners)
    assert all(owner.access == capability.access for _, capability, owner in owners)


def test_unknown_tools_are_not_inferred_from_their_names():
    assert tool_capability("semantic_tools.read_and_publish_something") is None
    assert tool_capability("new_tools.read") is None
    assert tool_capability("bash_tools.new_command").access == "restricted"
