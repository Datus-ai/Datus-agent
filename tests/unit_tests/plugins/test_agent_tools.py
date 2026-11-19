# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Instance-bound plugin tools preserve state and validate input before business execution."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from datus.plugins.agent_tools import bind_plugin_tools
from datus.plugins.agents import PluginToolDeclaration
from datus.tools.registry.contracts import tool_definition
from datus.utils.exceptions import DatusException


def _declaration(tool_id="research_tools.remember"):
    return PluginToolDeclaration(
        tool_id=tool_id,
        description="Remember one business observation.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        access="write",
    )


@pytest.mark.asyncio
async def test_plain_bound_function_executes_with_registered_effect_metadata():
    observations = []

    def remember(value):
        observations.append(value)
        return {"observations": list(observations)}

    tool = bind_plugin_tools([_declaration()], {"research_tools.remember": remember})[0]
    result = await tool.on_invoke_tool(None, json.dumps({"value": "sales"}))

    assert result == {"observations": ["sales"]}
    assert observations == ["sales"]
    assert tool_definition(tool).tool_id == "research_tools.remember"
    assert tool_definition(tool).access == "write"


@pytest.mark.asyncio
async def test_proxy_wildcard_keeps_instance_bound_business_callback_local():
    from datus.tools.proxy.proxy_tool import apply_proxy_tools
    from datus.tools.proxy.tool_result_channel import ToolResultChannel
    from datus.tools.registry.tool_registry import ToolRegistry

    evidence = []

    def remember(value):
        evidence.append(value)
        return {"events": list(evidence)}

    tool = bind_plugin_tools([_declaration()], {"research_tools.remember": remember})[0]
    registry = ToolRegistry()
    registry.register_tools("research_tools", [tool])
    node = SimpleNamespace(tools=[tool], tool_registry=registry, tool_channel=ToolResultChannel())
    apply_proxy_tools(node, ["*"])
    result = await node.tools[0].on_invoke_tool(None, '{"value": "sale"}')
    assert result == {"events": ["sale"]}
    assert evidence == ["sale"]
    assert node.proxied_tool_names == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [{"value": 42}, {"other": "value"}, {"value": "sales", "extra": True}])
async def test_invalid_arguments_do_not_reach_the_business_callback(arguments):
    observations = []

    def remember(value):
        observations.append(value)
        return {"observations": list(observations)}

    tool = bind_plugin_tools([_declaration()], {"research_tools.remember": remember})[0]
    result = await tool.on_invoke_tool(None, json.dumps(arguments))

    assert result["success"] == 0
    assert "Invalid arguments" in result["error"]
    assert observations == []


def test_undeclared_callbacks_cannot_add_tools():
    with pytest.raises(DatusException, match="undeclared"):
        bind_plugin_tools([_declaration()], {"research_tools.other": lambda: {"done": True}})


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["#/$defs/text", "#/$defs/missing"], ids=["local-reference", "missing-reference"])
async def test_plugin_input_references_return_an_error_before_invoking_callback(reference):
    observations = []

    def remember(value):
        observations.append(value)
        return {"observations": list(observations)}

    declaration = PluginToolDeclaration(
        **{
            **_declaration().model_dump(),
            "input_schema": {
                "type": "object",
                "$defs": {"text": {"type": "string"}},
                "properties": {"value": {"$ref": reference}},
            },
        }
    )
    tool = bind_plugin_tools([declaration], {declaration.tool_id: remember})[0]
    result = await tool.on_invoke_tool(None, '{"value":"sales"}')

    if reference == "#/$defs/text":
        assert result == {"observations": ["sales"]}
        assert observations == ["sales"]
    else:
        assert result["success"] == 0
        assert "unresolvable reference" in result["error"]
        assert observations == []


def test_plugin_callback_cannot_replace_native_implementation():
    with pytest.raises(DatusException, match="cannot replace a native tool"):
        bind_plugin_tools([_declaration("db_tools.execute_sql")], {"db_tools.execute_sql": lambda value: value})


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_parallel_model_tool_calls_do_not_race_shared_runtime_state():
    entered = asyncio.Event()
    release = asyncio.Event()
    second_attempted = asyncio.Event()
    counter = 0

    async def remember(value):
        nonlocal counter
        previous = counter
        entered.set()
        await release.wait()
        counter = previous + 1
        return {"count": counter, "value": value}

    first_tool, second_tool = bind_plugin_tools(
        [_declaration(), _declaration("research_tools.remember_second")],
        {"research_tools.remember": remember, "research_tools.remember_second": remember},
    )
    first = asyncio.create_task(first_tool.on_invoke_tool(None, '{"value":"first"}'))
    await entered.wait()

    async def second_call():
        second_attempted.set()
        return await second_tool.on_invoke_tool(None, '{"value":"second"}')

    second = asyncio.create_task(second_call())
    await second_attempted.wait()
    release.set()
    results = await asyncio.gather(first, second)

    assert results == [{"count": 1, "value": "first"}, {"count": 2, "value": "second"}]
    assert counter == 2
