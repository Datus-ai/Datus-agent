# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0.

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents import RunConfig, Runner, function_tool
from agents.exceptions import ModelBehaviorError
from agents.mcp import MCPServerStdio
from mcp.types import Tool

from datus.agent.node.node_factory import create_interactive_node, create_node_input
from datus.cli.execution_state import InteractionBroker
from datus.configuration.agent_config import ModelConfig
from datus.models.codex_model import CodexModel
from datus.models.openai_model import OpenAIModel
from datus.models.tool_preflight import ToolPreflightAgent
from datus.models.tool_result_stream import ToolResultStream
from datus.observability.tool_calls import observe_tool_hooks
from datus.schemas.action_history import ActionRole, ActionStatus
from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
from datus.tools.permission.permission_hooks import PermissionHooks
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.registry.tool_registry import ToolRegistry


def _start_run(model, tools, hooks):
    stream = ToolResultStream(observe_tool_hooks(hooks))
    agent = ToolPreflightAgent(name="tool-check", model=model, tools=tools, hooks=stream)
    run = Runner.run_streamed(agent, "Run tools", run_config=RunConfig(tracing_disabled=True))
    return run, stream.stream_events(run.stream_events(), run)


def _model_results(model, index):
    return {
        item["call_id"]: item["output"] for item in model.inputs[index] if item.get("type") == "function_call_output"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("remains_enabled", [False, True], ids=["disabled-after-discovery", "still-enabled"])
async def test_sdk_rechecks_configured_tool_before_execution(scripted_agents_model, remains_enabled):
    enabled = True
    executions = []

    def is_enabled(context, agent):
        return enabled

    async def on_llm_end(context, agent, response):
        nonlocal enabled
        enabled = remains_enabled

    @function_tool(is_enabled=is_enabled)
    def probe_tool() -> str:
        executions.append("read")
        return "result"

    original_guardrails = probe_tool.tool_input_guardrails
    hooks = SimpleNamespace(on_llm_end=on_llm_end, on_tool_start=AsyncMock())
    script = ("tool", "tool", "final") if remains_enabled else ("tool", "final")
    model = scripted_agents_model(script=script)
    run, events = _start_run(model, [probe_tool], hooks)

    async def consume():
        async with asyncio.timeout(5):
            async for _ in events:
                pass

    if remains_enabled:
        await consume()
        assert run.final_output == "done"
        assert executions == ["read", "read"]
        assert hooks.on_tool_start.await_count == 2
    else:
        with pytest.raises(ModelBehaviorError, match="Tool probe_tool is currently disabled"):
            await consume()
        assert executions == []
        hooks.on_tool_start.assert_not_awaited()
        assert len(model.inputs) == 1
    assert probe_tool.tool_input_guardrails is original_guardrails


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai-compatible", "codex"])
async def test_sdk_rejects_mcp_call_before_rpc_and_preserves_batch(scripted_agents_model, monkeypatch, provider):
    server = MCPServerStdio(params={"command": "unused"}, name="demo")
    server.connect = AsyncMock()
    server.cleanup = AsyncMock()
    server.list_tools = AsyncMock(
        return_value=[Tool(name="mcp_read", inputSchema={"type": "object", "properties": {}})]
    )
    server.call_tool = AsyncMock()
    permission = PermissionHooks(
        broker=InteractionBroker(),
        permission_manager=PermissionManager(
            global_config=PermissionConfig(
                default_permission=PermissionLevel.ALLOW,
                rules=[PermissionRule(tool="mcp.demo", pattern="mcp_read", permission=PermissionLevel.DENY)],
            )
        ),
        node_name="chat",
        tool_registry=ToolRegistry({"mcp_read": "mcp.demo"}),
    )
    executions = []

    @function_tool
    def read_project() -> str:
        executions.append("read")
        return "authorized result"

    model = scripted_agents_model(tool_calls=[("mcp_read", "{}"), ("read_project", "{}")])
    monkeypatch.setattr("datus.cli.provider_model_catalog.resolve_model_details", lambda **kwargs: {})
    monkeypatch.setattr(
        "datus.utils.trace_context.build_agents_run_config_kwargs", lambda **kwargs: {"tracing_disabled": True}
    )
    if provider == "codex":
        runtime = CodexModel(ModelConfig(type="codex", model="gpt-5.3-codex", api_key="test"))
        monkeypatch.setattr(runtime, "_get_responses_model", lambda: model)
    else:
        runtime = OpenAIModel(ModelConfig(type="openai", model="gpt-4.1-mini", api_key="test", max_retry=1))
        monkeypatch.setattr(runtime.litellm_adapter, "get_agents_sdk_model", lambda: model)

    async with asyncio.timeout(5):
        actions = [
            action
            async for action in runtime.generate_with_tools_stream(
                prompt="Run tools", tools=[read_project], mcp_servers={"demo": server}, hooks=permission
            )
        ]

    server.call_tool.assert_not_awaited()
    assert executions == ["read"]
    assert any(action.role == ActionRole.ASSISTANT and action.output.get("raw_output") == "done" for action in actions)
    completions = [
        (action.action_id, action.status)
        for action in actions
        if action.role == ActionRole.TOOL and action.status != ActionStatus.PROCESSING
    ]
    assert sorted(completions) == [
        ("complete_call_1", ActionStatus.FAILED),
        ("complete_call_1_2", ActionStatus.SUCCESS),
    ]
    assert len(model.inputs) == 2
    assert sorted(item["call_id"] for item in model.inputs[1] if item.get("type") == "function_call_output") == [
        "call_1",
        "call_1_2",
    ]
    next_inputs = _model_results(model, 1)
    refusal = json.loads(next_inputs["call_1"])
    assert refusal["success"] == 0
    assert "mcp_read" in refusal["error"]
    assert next_inputs["call_1_2"] == "authorized result"


@pytest.mark.asyncio
async def test_sdk_scope_rejection_keeps_authorized_host_read(real_agent_config, scripted_agents_model, monkeypatch):
    real_agent_config.agentic_nodes["reader"] = {
        "node_class": "configurable",
        "tools": ["filesystem_tools.grep", "filesystem_tools.read_file"],
        "scoped_context": {"project_files": ["docs/**"]},
    }
    node = create_interactive_node("reader", real_agent_config, execution_mode="workflow")
    try:
        user_input = create_node_input("Read the spec.", node)
        await node._before_stream(SimpleNamespace(user_input=user_input))
        host_dir = Path(node._resolve_session_data_dir()) / "semantic_authoring"
        host_dir.mkdir(parents=True, exist_ok=True)
        ref = host_dir / "reference.md"
        ref.write_text("Host specification", encoding="utf-8")
        node.execution_scope.host_read_roots = (host_dir,)
        model = scripted_agents_model(
            tool_calls=[
                ("grep", json.dumps({"pattern": "Host specification", "path": str(host_dir)})),
                ("read_file", json.dumps({"path": str(ref)})),
            ]
        )
        hooks = node._compose_hooks()
        check_permission = AsyncMock(wraps=node.permission_hooks.on_tool_start)
        monkeypatch.setattr(node.permission_hooks, "on_tool_start", check_permission)
        run, events = _start_run(model, node.tools, hooks)
        async with asyncio.timeout(5):
            async for _ in events:
                pass

        assert [call.args[2].name for call in check_permission.await_args_list] == ["read_file"]
        assert run.final_output == "done"
        outputs = _model_results(model, 1)
        refusal = json.loads(outputs["call_1"])
        assert refusal["success"] == 0
        assert "File search is outside scoped_context.project_files" in refusal["error"]
        assert "Host specification" in outputs["call_1_2"]
    finally:
        await node.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["y", "n", "", None], ids=["allow", "deny", "dismiss", "workflow"])
async def test_sdk_checks_approval_once_and_records_response(scripted_agents_model, answer):
    broker = InteractionBroker()
    hooks = PermissionHooks(
        broker=broker,
        permission_manager=PermissionManager(global_config=PermissionConfig(default_permission=PermissionLevel.ASK)),
        node_name="chat",
        tool_registry=ToolRegistry({"write_project": "project"}),
        non_interactive=answer is None,
    )
    executions = []

    @function_tool
    def write_project() -> str:
        executions.append("write")
        return "write result"

    model = scripted_agents_model(tool_name="write_project")
    run, events = _start_run(model, [write_project], hooks)
    async with asyncio.timeout(5):
        if answer is not None:
            requests = broker.fetch()
            request = await anext(requests)
            await broker.submit(request.action_id, [[answer]])
            await requests.aclose()
        async for _ in events:
            pass

    assert run.final_output == "done"
    output = _model_results(model, 1)["call_1"]
    if answer == "y":
        assert executions == ["write"]
        assert output == "write result"
    else:
        assert executions == []
        rejection = json.loads(output)
        assert rejection["success"] == 0
        reason = "requires user confirmation" if answer is None else "User rejected execution"
        assert reason in rejection["error"]


@pytest.mark.asyncio
async def test_cancelling_sdk_run_during_approval_stops_before_next_model_request(scripted_agents_model):
    broker = InteractionBroker()
    hooks = PermissionHooks(
        broker=broker,
        permission_manager=PermissionManager(global_config=PermissionConfig(default_permission=PermissionLevel.ASK)),
        node_name="chat",
        tool_registry=ToolRegistry({"write_project": "project"}),
    )
    executions = []

    @function_tool
    def write_project() -> str:
        executions.append("write")
        return "write result"

    model = scripted_agents_model(tool_name="write_project")
    run, events = _start_run(model, [write_project], hooks)
    requests = broker.fetch()
    async with asyncio.timeout(5):
        await anext(requests)
        run.cancel(mode="immediate")
        async for _ in events:
            pass
        await requests.aclose()
    assert len(model.inputs) == 1
    assert executions == []
