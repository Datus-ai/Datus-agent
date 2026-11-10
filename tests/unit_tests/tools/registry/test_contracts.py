"""Mandatory registration, inherited implementations and wrapper preservation."""

from types import SimpleNamespace

import pytest
from agents import FunctionTool

from datus.agent.node.agentic_node import AgenticNode
from datus.tools.func_tool.base import trans_to_function_tool
from datus.tools.registry.capabilities import TOOL_CAPABILITIES
from datus.tools.registry.contracts import (
    ToolCapability,
    ToolDefinition,
    ToolDefinitionError,
    clone_registered_tool,
    definition_from_callable,
    registered_function_tool,
    tool_contract,
    tool_definition,
    validate_registered_tools,
)
from datus.tools.registry.tool_registry import ToolRegistry
from datus.utils.mcp_decorators import get_mcp_tools, mcp_tool


@pytest.mark.parametrize("options", [{}, {"access": "read"}, {"direct_call": True}])
def test_declaration_requires_both_fields(options):
    """A new registration cannot silently inherit an access or exposure default."""
    with pytest.raises(TypeError):
        tool_contract(**options)


@pytest.mark.parametrize("access,direct", [("reader", True), ("read", "yes"), ("read", 1)])
def test_declaration_rejects_invalid_contracts(access, direct):
    """Invalid effect values and truthy non-booleans are definition errors."""
    with pytest.raises(ToolDefinitionError):
        tool_contract(access=access, direct_call=direct)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_missing_declaration_fails_at_conversion(asynchronous):
    """Neither synchronous nor asynchronous methods can bypass declaration."""

    def sync_method():
        return 1

    async def async_method():
        return 1

    with pytest.raises(ToolDefinitionError, match="Missing @tool_contract"):
        trans_to_function_tool(async_method if asynchronous else sync_method)


@pytest.mark.asyncio
async def test_inherited_async_contract_and_override():
    """Inherited declarations work, but an overriding implementation must redeclare."""

    class Parent:
        permission_category = "custom_tools"

        @tool_contract(access="read", direct_call=False)
        async def inspect(self, value: int) -> int:
            return value

    class Child(Parent):
        pass

    class UndeclaredOverride(Parent):
        async def inspect(self, value: int) -> int:
            return value + 1

    tool = trans_to_function_tool(Child().inspect)
    assert tool_definition(tool) == ToolDefinition("custom_tools.inspect", ToolCapability("read", False))
    assert await tool.on_invoke_tool(None, '{"value":7}') == 7
    with pytest.raises(ToolDefinitionError, match="Missing @tool_contract"):
        trans_to_function_tool(UndeclaredOverride().inspect)


def test_missing_category_and_stale_native_inventory(monkeypatch):
    """New native contracts cannot silently run against an older generated inventory."""

    @tool_contract(access="read", direct_call=False)
    def no_category():
        return 1

    with pytest.raises(ToolDefinitionError, match="category"):
        definition_from_callable(no_category)

    @tool_contract(access="read", direct_call=False, category="future_tools")
    def future_reader():
        return 1

    monkeypatch.setattr(future_reader, "__module__", "datus.tools.future")
    with pytest.raises(ToolDefinitionError, match="Stale native tool catalog"):
        trans_to_function_tool(future_reader)
    monkeypatch.setitem(TOOL_CAPABILITIES, "future_tools.future_reader", ToolCapability("read", False))
    assert tool_definition(trans_to_function_tool(future_reader)).tool_id == "future_tools.future_reader"


def test_mcp_registration_requires_the_actual_method_contract():
    """MCP does not get a second optional or name-inferred capability path."""

    class Undeclared:
        permission_category = "custom_tools"

        @mcp_tool()
        async def inspect(self):
            return 1

    with pytest.raises(ToolDefinitionError, match="Missing @tool_contract"):
        get_mcp_tools(Undeclared)

    class Declared:
        permission_category = "custom_tools"

        @mcp_tool()
        @tool_contract(access="read", direct_call=True)
        async def inspect(self):
            return 1

    name, method, _ = get_mcp_tools(Declared)[0]
    assert name == "inspect"
    assert definition_from_callable(method, owner=Declared).capability == ToolCapability("read", True)


def test_dynamic_factory_requires_definition_and_preserves_sdk_settings():
    """Dynamic tools and their clones retain declarations and SDK guardrail fields."""
    definition = ToolDefinition("custom_tools.inspect", ToolCapability("read", False))

    async def invoke(context, arguments):
        return arguments

    values = dict(name="inspect", description="Inspect", params_json_schema={}, on_invoke_tool=invoke)
    with pytest.raises(TypeError):
        registered_function_tool(**values)
    original = registered_function_tool(definition=definition, needs_approval=True, is_enabled=False, **values)
    wrapped = clone_registered_tool(original, description="Wrapped")
    assert tool_definition(wrapped) is definition
    assert wrapped.needs_approval is True
    assert wrapped.is_enabled is False
    assert wrapped.description == "Wrapped"
    assert wrapped.params_json_schema == original.params_json_schema
    with pytest.raises(ToolDefinitionError, match="name"):
        clone_registered_tool(original, name="different_tool")


def test_final_assembly_rejects_unregistered_tools_before_visibility():
    """Even a tool that DENY would hide must first have a valid registration."""

    async def invoke(context, arguments):
        return None

    raw = FunctionTool(name="write_file", description="", params_json_schema={}, on_invoke_tool=invoke)
    node = SimpleNamespace(tools=[raw], agent_config=SimpleNamespace(restrict_tool_visibility=True))
    with pytest.raises(ToolDefinitionError, match="Unregistered tool"):
        AgenticNode._ensure_tool_transformers(node)


def test_canonical_identity_cannot_be_overwritten_by_another_group():
    """A mounted tool's definition, rather than its name, determines its category."""
    definition = ToolDefinition("custom_tools.inspect", ToolCapability("read", False))

    async def invoke(context, arguments):
        return None

    tool = registered_function_tool(
        definition=definition, name="inspect", description="", params_json_schema={}, on_invoke_tool=invoke
    )
    registry = ToolRegistry()
    with pytest.raises(ToolDefinitionError, match="under"):
        registry.register_tools("db_tools", [tool])
    registry.register_tools("custom_tools", [tool])
    assert registry.get("inspect") == "custom_tools"
    second = registered_function_tool(
        definition=ToolDefinition("other_tools.inspect", ToolCapability("read", False)),
        name="inspect",
        description="",
        params_json_schema={},
        on_invoke_tool=invoke,
    )
    with pytest.raises(ToolDefinitionError, match="Ambiguous tool name"):
        validate_registered_tools([tool, second])


def test_known_identity_cannot_claim_a_different_effect():
    """Dynamic registration cannot reclassify an existing writer as a reader."""

    async def invoke(context, arguments):
        return None

    with pytest.raises(ToolDefinitionError, match="Conflicting capability"):
        registered_function_tool(
            definition=ToolDefinition("filesystem_tools.write_file", ToolCapability("read", True)),
            name="write_file",
            description="",
            params_json_schema={},
            on_invoke_tool=invoke,
        )


def test_method_cannot_change_its_owning_family():
    """An explicit category cannot bypass the tool group's permission namespace."""

    class Tool:
        permission_category = "bash_tools"

        @tool_contract(access="read", direct_call=True, category="filesystem_tools")
        def inspect(self):
            return 1

    with pytest.raises(ToolDefinitionError, match="declaring class"):
        trans_to_function_tool(Tool().inspect)


def test_mcp_alias_requires_a_declared_compatibility_name():
    """An attribute alias cannot create an unclassified transport name."""

    class Tool:
        permission_category = "custom_tools"

        @mcp_tool()
        @tool_contract(access="read", direct_call=True)
        def inspect(self):
            return 1

        alias = inspect

    with pytest.raises(ToolDefinitionError, match="MCP alias"):
        get_mcp_tools(Tool)
