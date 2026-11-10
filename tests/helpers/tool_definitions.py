"""Registered SDK test doubles for tests of proxies and argument middleware.

These tests exercise wrapping, not capability declaration. Reuse native fixture
contracts where possible; production registration never infers a contract by name.
"""

from agents import FunctionTool

from datus.tools.registry.capabilities import TOOL_CAPABILITIES
from datus.tools.registry.contracts import ToolCapability, ToolDefinition, register_tool_definition


def registered_test_tool(*, category=None, tool_type=FunctionTool, **kwargs):
    """Construct a declared test double while preserving the actual SDK type."""
    name = kwargs["name"]
    candidates = [
        (identity, capability)
        for identity, capability in TOOL_CAPABILITIES.items()
        if identity.split(".", 1)[1] == name and (category is None or identity.split(".", 1)[0] == category)
    ]
    if len(candidates) > 1:
        raise ValueError(f"Specify a fixture category for ambiguous tool {name!r}")
    identity, capability = (
        candidates[0] if candidates else (f"{category or 'tools'}.{name}", ToolCapability("restricted", False))
    )
    return register_tool_definition(tool_type(**kwargs), ToolDefinition(identity, capability))
