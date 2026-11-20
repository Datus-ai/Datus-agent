# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Exercise the actual common loop, factories, scoped tools and runtime lifecycle."""

import asyncio
import json
from pathlib import Path

import pytest

from datus.agent.node.node import Node
from datus.agent.node.node_factory import create_interactive_node, create_node_input
from datus.schemas.configurable_agentic_node_models import ConfigurableNodeInput
from tests.unit_tests.mock_llm_model import MockLLMResponse, MockToolCall
from tests.unit_tests.plugins.conftest import plugin_env as plugin_env


def _configure(config, **overrides):
    config.agentic_nodes["research"] = {
        "node_class": "configurable",
        "instructions": "Answer only using supplied evidence.",
        **overrides,
    }
    return config


async def _run(node, message="Research sales"):
    node.input = ConfigurableNodeInput(user_message=message)
    return [action async for action in node.execute_stream()]


@pytest.mark.asyncio
async def test_chat_refreshes_same_name_plugin_capabilities_without_replacing_proxy_invocation(
    real_agent_config, mock_llm_create, plugin_env
):
    from datus.tools.func_tool.sub_agent_task_tool import BUILTIN_SUBAGENT_DESCRIPTIONS
    from datus.tools.proxy.proxy_tool import apply_proxy_tools
    from datus.tools.proxy.tool_result_channel import ToolResultChannel

    config = real_agent_config
    config.agentic_nodes["chat"].update(tools="sub_agent_tools.task", subagents="semantic_modeling")
    node = create_interactive_node(None, config, execution_mode="workflow")
    apply_proxy_tools(node, ["sub_agent_tools.task"], channel=ToolResultChannel())
    tool = next(item for item in node.tools if item.name == "task")
    invoke = tool.on_invoke_tool
    mock_llm_create.reset(responses=[MockLLMResponse(content="Ready") for _ in range(3)])

    async def run():
        node.input = create_node_input("Explain the available modeling capabilities.", node)
        return [action async for action in node.execute_stream()]

    try:
        await run()
        assert node.result.success is True, node.result.error
        assert BUILTIN_SUBAGENT_DESCRIPTIONS["semantic_modeling"] in tool.description
        plugin_env(
            "datus-semantic-plugin",
            """manifest_version: 1
agents:
  semantic_modeling:
    description: Author semantic models, ontology concepts and mappings together.
    instructions: Review the requested modeling change.
""",
        )
        config.agentic_nodes["semantic_modeling"] = {
            "node_class": "configurable",
            "source": {"plugin": "datus-semantic-plugin", "agent": "semantic_modeling"},
        }
        await run()
        assert node.result.success is True, node.result.error
        assert tool.description.count("- semantic_modeling:") == 1
        assert (
            "- semantic_modeling: Author semantic models, ontology concepts and mappings together." in tool.description
        )
        child = node.sub_agent_task_tool._create_node("semantic_modeling")
        try:
            assert child.get_node_class_name() == "configurable"
            assert child.spec.description == "Author semantic models, ontology concepts and mappings together."
        finally:
            await child.aclose()
        config.agentic_nodes["semantic_modeling"]["description"] = "Review joint modeling proposals without publishing."
        await run()
        assert node.result.success is True, node.result.error
        assert "- semantic_modeling: Review joint modeling proposals without publishing." in tool.description
        assert next(item for item in node.tools if item.name == "task") is tool
        assert tool.on_invoke_tool is invoke
        assert "task" in node.proxied_tool_names
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_pure_config_runs_without_implicit_tools_or_business_resources(real_agent_config, mock_llm_create):
    config = _configure(real_agent_config)
    mock_llm_create.reset(responses=[MockLLMResponse(content="Supplied evidence is sufficient.")])
    node = create_interactive_node("research", config, execution_mode="workflow")
    try:
        actions = await _run(node)
        assert node.result.success is True, node.result.error
        assert node.result.response == "Supplied evidence is sufficient."
        assert node.tools == []
        assert node.agent_config.services.datasources == {}
        assert config.current_datasource == "california_schools"
        assert actions[-1].output["response"] == node.result.response
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_selected_sql_tool_runs_against_real_scoped_sqlite(real_agent_config, mock_llm_create):
    config = _configure(
        real_agent_config, tools=["db_tools.execute_sql"], scoped_context={"datasource": "california_schools"}
    )
    mock_llm_create.reset(
        responses=[
            MockLLMResponse(
                content="Counted the schools.",
                tool_calls=[MockToolCall("execute_sql", {"sql": "SELECT COUNT(*) AS n FROM schools"})],
            )
        ]
    )
    node = create_interactive_node("research", config, execution_mode="workflow")
    try:
        await _run(node)
        assert node.result.success is True, node.result.error
        assert [tool.name for tool in node.tools] == ["execute_sql"]
        output = mock_llm_create.tool_results[0]["output"]
        assert output["success"] == 1, output
        assert int(output["result"]["compressed_data"].splitlines()[-1].split(",")[-1]) > 0
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_missing_resource_declaration_stops_database_access_before_model(real_agent_config, mock_llm_create):
    node = create_interactive_node("research", _configure(real_agent_config, tools=["execute_sql"]))
    try:
        await _run(node)
        assert not node.result.success
        assert "scoped_context.datasource" in node.result.error
        assert mock_llm_create.call_history == []
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_runtime_scope_narrowing_rebinds_real_sql_tools_on_each_turn(
    real_agent_config, mock_llm_create, plugin_env
):
    plugin_env(
        "private",
        """manifest_version: 1
agents:
  research:
    runtime_factory: '{pkg}.runtime:create_runtime'
""",
        files={
            "runtime.py": """
class Runtime:
    def __init__(self, host):
        self.host = host
        self.turns = 0
    def prepare_turn(self, context):
        self.turns += 1
        if self.turns == 1:
            return {"scoped_context": {"tables": "schools"}}
        return {}
    async def finalize(self, result, context):
        checked = await self.host.call_tool("db_tools.execute_sql", {"sql": "SELECT COUNT(*) FROM satscores"})
        return {"host_query": checked, "scope": self.host.scoped_context, "turns": self.turns}
def create_runtime(config, host):
    return Runtime(host)
"""
        },
    )
    config = _configure(
        real_agent_config,
        source={"plugin": "private", "agent": "research"},
        tools=["db_tools.execute_sql"],
        scoped_context={"datasource": "california_schools", "tables": "schools,satscores"},
    )
    mock_llm_create.reset(responses=[MockLLMResponse(content="First"), MockLLMResponse(content="Second")])
    node = create_interactive_node("research", config, execution_mode="workflow")
    try:
        await _run(node)
        first = node.result.model_dump()
        assert first["success"], first["error"]
        assert first["host_query"]["success"] == 0
        assert "outside scoped context" in first["host_query"]["error"]
        assert first["scope"]["tables"] == "schools"
        await _run(node)
        second = node.result.model_dump()
        assert second["success"], second["error"]
        assert second["host_query"]["success"] == 1
        assert second["scope"]["tables"] == "schools,satscores"
        assert second["turns"] == 2
        assert config.agentic_nodes["research"]["scoped_context"]["tables"] == "schools,satscores"
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_factory_input_and_serialization_preserve_agent_name_and_business_result(
    real_agent_config, mock_llm_create
):
    config = _configure(real_agent_config, output={"format": "json"})
    node = create_interactive_node("research", config, node_id="opaque-id", execution_mode="workflow")
    mock_llm_create.reset(responses=[MockLLMResponse(content='{"concepts": ["sale"]}')])
    try:
        node.input = create_node_input("Return concepts", node)
        assert isinstance(node.input, ConfigurableNodeInput)
        await _run(node)
        node.result.__pydantic_extra__["validation"] = {"valid": True}
        restored = Node.from_dict(json.loads(json.dumps(node.to_dict())), config)
        try:
            assert restored.get_node_name() == "research"
            assert restored.result.data == {"concepts": ["sale"]}
            assert restored.result.model_dump()["validation"] == {"valid": True}
        finally:
            await restored.aclose()
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_configurable_input_activates_plan_mode(real_agent_config, mock_llm_create):
    node = create_interactive_node("research", _configure(real_agent_config), execution_mode="workflow")
    mock_llm_create.reset(responses=[MockLLMResponse(content="I'll prepare a plan.")])
    node.input = create_node_input("Plan an analysis", node, plan_mode=True)
    try:
        _ = [action async for action in node.execute_stream()]
        assert node.input.plan_mode is True
        assert node.is_in_plan_mode()
        assert node.plan_file_path
        assert "confirm_plan" in {tool.name for tool in node.tools}
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_malformed_native_arguments_return_retryable_tool_error(real_agent_config, mock_llm_create):
    from agents.tool_context import ToolContext

    config = _configure(
        real_agent_config,
        tools=["db_tools.execute_sql"],
        scoped_context={"datasource": real_agent_config.current_datasource},
    )
    node = create_interactive_node("research", config, execution_mode="workflow")
    mock_llm_create.reset(responses=[MockLLMResponse(content="Ready.")])
    try:
        await _run(node)
        tool = next(item for item in node.tools if item.name == "execute_sql")
        arguments = '{"sql":"SELECT 1"'
        context = ToolContext(context=None, tool_name=tool.name, tool_call_id="repaired", tool_arguments=arguments)
        hooks = node._compose_hooks()

        await hooks.on_tool_start(context, None, tool)
        result = await tool.on_invoke_tool(context, context.tool_arguments)
        assert result["success"] == 0
        assert "Retry the tool" in result["error"]

        valid = ToolContext(
            context=None,
            tool_name=tool.name,
            tool_call_id="valid",
            tool_arguments='{"sql":"SELECT 1"}',
        )
        await hooks.on_tool_start(valid, None, tool)
        actual = await tool.on_invoke_tool(valid, valid.tool_arguments)
        assert actual["success"] == 1
    finally:
        await node.aclose()


_RUNTIME = """
class Runtime:
    def __init__(self, config, host):
        self.events = []
        self.host = host
        self.close_path = config["close_path"]
    def prepare_turn(self, context):
        self.events.append(context["input"]["user_message"])
        return {"instructions": "Current count: " + str(len(self.events)),
                "tool_bindings": {"research_tools.remember": self.remember}}
    def remember(self, value):
        self.events.append(value)
        return {"events": list(self.events)}
    async def finalize(self, result, context):
        return {"business": {"events": list(self.events)}}
    def close(self):
        from pathlib import Path
        path = Path(self.close_path)
        with path.open("a") as stream:
            stream.write("closed\\n")
def create_runtime(config, host):
    return Runtime(config, host)
"""


@pytest.mark.asyncio
async def test_private_runtime_state_is_shared_with_callbacks_but_isolated_between_nodes(
    real_agent_config, mock_llm_create, plugin_env, tmp_path
):
    close_path = tmp_path / "runtime-closed.txt"
    plugin_env(
        "private",
        """
manifest_version: 1
agents:
  research:
    instructions: Research supplied concepts.
    runtime_factory: '{pkg}.runtime:create_runtime'
    tool_definitions:
      - tool_id: research_tools.remember
        description: Remember supplied evidence.
        access: read
        input_schema:
          type: object
          properties:
            value: {type: string}
          required: [value]
          additionalProperties: false
""",
        files={"runtime.py": _RUNTIME},
    )
    config = _configure(
        real_agent_config,
        source={"plugin": "private", "agent": "research"},
        runtime_config={"close_path": str(close_path)},
    )
    mock_llm_create.reset(
        responses=[
            MockLLMResponse(content="One", tool_calls=[MockToolCall("remember", {"value": "sale"})]),
            MockLLMResponse(content="Two"),
            MockLLMResponse(content="Other"),
        ]
    )
    first = create_interactive_node("research", config, execution_mode="workflow")
    second = create_interactive_node("research", config, execution_mode="workflow")
    # Read effects do not automatically grant a publisher's category. The
    # operator explicitly authorizes this installed plugin in the normal profile.
    from datus.tools.permission.permission_config import PermissionLevel, PermissionRule

    for node in (first, second):
        node.permission_manager.global_config.rules.append(
            PermissionRule(tool="research_tools", pattern="remember", permission=PermissionLevel.ALLOW)
        )
    try:
        await _run(first, "First")
        assert first.result.success is True, first.result.error
        assert first.result.model_dump()["business"]["events"] == ["First", "sale"]
        await _run(first, "Continue")
        assert "Current count: 3" in mock_llm_create.call_history[-1]["instruction"]
        await _run(second, "Separate")
        assert first.result.model_dump()["business"]["events"] == ["First", "sale", "Continue"]
        assert second.result.model_dump()["business"]["events"] == ["Separate"]
        assert not close_path.exists()
    finally:
        await first.aclose()
        await first.aclose()
        await second.aclose()
    assert close_path.read_text().splitlines() == ["closed", "closed"]


@pytest.mark.asyncio
async def test_skill_selection_does_not_implicitly_grant_other_skills_or_file_tools(real_agent_config, mock_llm_create):
    root = Path(real_agent_config.project_root) / ".datus" / "skills"
    for name in ("selected", "other"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Research guidance.\n---\nUse the {name} research rules.\n"
        )
    config = _configure(real_agent_config, skills=["selected"])
    mock_llm_create.reset(
        responses=[
            MockLLMResponse(
                content="Done",
                tool_calls=[
                    MockToolCall("load_skill", {"skill_name": "other"}),
                    MockToolCall("load_skill", {"skill_name": "selected"}),
                ],
            )
        ]
    )
    node = create_interactive_node("research", config, execution_mode="workflow")
    try:
        await _run(node)
        assert node.result.success is True, node.result.error
        assert [tool.name for tool in node.tools] == ["load_skill"]
        assert mock_llm_create.tool_results[0]["output"]["success"] == 0
        assert "selected research rules" in mock_llm_create.tool_results[1]["output"]["result"]
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_selected_skill_reference_can_be_read_with_an_explicit_file_resource(real_agent_config, mock_llm_create):
    skill = Path(real_agent_config.project_root) / ".datus" / "skills" / "selected"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: selected\ndescription: Research rules.\n---\nRead references/rules.md for the review rules.\n"
    )
    (skill / "references" / "rules.md").write_text("Compare revenue with the preceding fiscal quarter.")
    config = _configure(
        real_agent_config,
        skills=["selected"],
        tools=["filesystem_tools.read_file"],
        scoped_context={"project_files": [".datus/skills/selected/references/**"]},
    )
    mock_llm_create.reset(
        responses=[
            MockLLMResponse(
                content="Reviewed the rules",
                tool_calls=[
                    MockToolCall("load_skill", {"skill_name": "selected"}),
                    MockToolCall("read_file", {"path": ".datus/skills/selected/references/rules.md"}),
                ],
            )
        ]
    )
    node = create_interactive_node("research", config, execution_mode="workflow")
    try:
        await _run(node)
        assert node.result.success is True, node.result.error
        loaded = mock_llm_create.tool_results[1]["output"]
        assert loaded["success"] == 1, loaded
        assert "preceding fiscal quarter" in str(loaded["result"])
    finally:
        await node.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile,allowed", [("auto", True), ("normal", False)])
async def test_finalize_host_io_uses_the_same_caller_permissions_and_resource_selection(
    real_agent_config, mock_llm_create, plugin_env, profile, allowed
):
    from datus.tools.permission.profiles import get_profile

    plugin_env(
        "private",
        """
manifest_version: 1
agents:
  research:
    runtime_factory: '{pkg}.runtime:create_runtime'
""",
        files={
            "runtime.py": """
class Runtime:
    def __init__(self, host):
        self.host = host
    async def finalize(self, result, context):
        written = await self.host.call_tool("filesystem_tools.write_file", {
            "path": "public/output.txt", "content": result["response"]
        })
        if not written["success"]:
            raise ValueError(written["error"])
        return {"artifact": "public/output.txt"}
def create_runtime(config, host):
    return Runtime(host)
"""
        },
    )
    config = _configure(
        real_agent_config,
        source={"plugin": "private", "agent": "research"},
        tools=["filesystem_tools.write_file"],
        scoped_context={"project_files": ["public/**"]},
    )
    config.active_profile_name = profile
    config.permissions_config = get_profile(profile)
    mock_llm_create.reset(responses=[MockLLMResponse(content="Reviewed sales.")])
    node = create_interactive_node("research", config, execution_mode="workflow")
    try:
        await _run(node)
        artifact = Path(config.project_root) / "public" / "output.txt"
        assert node.result.success is allowed, node.result.error
        assert artifact.exists() is allowed
        if allowed:
            assert artifact.read_text() == "Reviewed sales."
            assert node.result.model_dump()["artifact"] == "public/output.txt"
        else:
            assert "permission" in node.result.error.lower()
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_invalid_json_output_fails_instead_of_claiming_a_structured_result(real_agent_config, mock_llm_create):
    node = create_interactive_node("research", _configure(real_agent_config, output={"format": "json"}))
    mock_llm_create.reset(responses=[MockLLMResponse(content="This is not JSON")])
    try:
        await _run(node)
        assert not node.result.success
        assert node.result.data is None
    finally:
        await node.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_cancelled_model_does_not_finalize_and_owner_can_release_runtime(
    real_agent_config, mock_llm_create, plugin_env, tmp_path, monkeypatch
):
    closed = tmp_path / "closed.txt"
    finalized = tmp_path / "finalized.txt"
    plugin_env(
        "private",
        """
manifest_version: 1
agents:
  research:
    runtime_factory: '{pkg}.runtime:create_runtime'
""",
        files={
            "runtime.py": """
class Runtime:
    def __init__(self, config):
        self.config = config
    def finalize(self, result, context):
        from pathlib import Path
        Path(self.config["finalized"]).write_text("finalized")
    async def close(self):
        from pathlib import Path
        Path(self.config["closed"]).write_text("closed")
def create_runtime(config, host):
    return Runtime(config)
"""
        },
    )
    config = _configure(
        real_agent_config,
        source={"plugin": "private", "agent": "research"},
        runtime_config={"closed": str(closed), "finalized": str(finalized)},
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def model_stream(**kwargs):
        entered.set()
        await release.wait()
        if False:
            yield

    monkeypatch.setattr(mock_llm_create, "generate_with_tools_stream", model_stream)
    node = create_interactive_node("research", config, execution_mode="workflow")
    execution = asyncio.create_task(_run(node))
    try:
        await entered.wait()
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
    finally:
        release.set()
        await node.aclose()
    assert not finalized.exists()
    assert closed.read_text() == "closed"


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
async def test_task_route_and_result_preserve_generic_business_fields(real_agent_config, mock_llm_create, success):
    from datus.tools.func_tool.sub_agent_task_tool import SubAgentTaskTool

    config = _configure(real_agent_config, description="Review selected concepts")
    task_tool = SubAgentTaskTool(config)
    assert "research" in task_tool._get_available_types()
    assert "Review selected concepts" in task_tool._build_task_description()
    node = task_tool._create_node("research")
    try:
        assert node.get_node_class_name() == "configurable"
        assert node._is_subagent is True
        input_data = task_tool._build_node_input(node, "Review concepts")
        assert isinstance(input_data, ConfigurableNodeInput)
        business = {"validation": {"concepts": [{"name": "sales", "valid": False}]}}
        output = {"success": success, "response": "Reviewed", **business}
        if not success:
            output["error"] = "Business validation failed"
        result = task_tool._convert_to_func_result(output, session_id=node.session_id, subagent_type="research")
        assert result.success == int(success)
        assert result.result["validation"] == business["validation"]
        assert result.result["type"] == "research"
        assert result.result["session_id"] == node.session_id
    finally:
        await node.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tools", "permissions", "execution_mode", "guidance"),
    [
        pytest.param(
            ["list_existing_osi_semantic_models"],
            {},
            "interactive",
            "No target-selection tool is available for this agent.",
            id="discovery-only",
        ),
        pytest.param(
            ["bind_osi_semantic_model_target"],
            {},
            "interactive",
            "Model discovery is unavailable for this agent.",
            id="missing-discovery",
        ),
        pytest.param(
            ["list_existing_osi_semantic_models", "bind_osi_semantic_model_target", "plan_osi_semantic_model_target"],
            {"plan_osi_semantic_model_target": "deny"},
            "interactive",
            "Call list_existing_osi_semantic_models, then select a target with bind_osi_semantic_model_target.",
            id="bind-allowed-plan-denied",
        ),
        pytest.param(
            ["list_existing_osi_semantic_models", "plan_osi_semantic_model_target"],
            {"plan_osi_semantic_model_target": "ask"},
            "interactive",
            "Call list_existing_osi_semantic_models, then select a target with plan_osi_semantic_model_target.",
            id="plan-with-confirmation",
        ),
        pytest.param(
            ["list_existing_osi_semantic_models", "plan_osi_semantic_model_target"],
            {"plan_osi_semantic_model_target": "ask"},
            "workflow",
            "No target-selection tool is available for this agent.",
            id="confirmation-unavailable",
        ),
    ],
)
async def test_unselected_semantic_model_guidance_uses_available_tools(
    real_agent_config, mock_llm_create, tools, permissions, execution_mode, guidance
):
    from datus.agent.node.stream_run_context import StreamRunContext
    from datus.schemas.action_history import ActionHistoryManager
    from datus.tools.permission.permission_config import PermissionConfig, PermissionRule

    config = _configure(
        real_agent_config,
        tools=tools,
        scoped_context={
            "datasource": real_agent_config.current_datasource,
            "project_files": ["subject/semantic_models/**"],
        },
    )
    config.permissions_config = PermissionConfig(
        default_permission="allow",
        rules=[
            PermissionRule(tool="semantic_tools", pattern=name, permission=permission)
            for name, permission in permissions.items()
        ],
    )
    config.workflow_permission_profile = "normal"
    node = create_interactive_node("research", config, execution_mode=execution_mode)
    try:
        node.input = create_node_input("Inspect the selected semantic model.", node)
        await node._before_stream(StreamRunContext(node.input, ActionHistoryManager()))
        with pytest.raises(RuntimeError) as error:
            await node._host.semantic_authoring.snapshot()
        assert str(error.value) == f"No semantic model is selected for this turn. {guidance}"
    finally:
        await node.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", [False, True])
async def test_rebuilt_plan_tool_preserves_web_review_across_turns(real_agent_config, mock_llm_create, accept):
    from types import SimpleNamespace

    from datus.agent.node.stream_run_context import StreamRunContext
    from datus.schemas.action_history import ActionHistoryManager
    from datus.tools.proxy.semantic_plan_review import apply_semantic_plan_review
    from datus.tools.proxy.tool_result_channel import ToolResultChannel

    class ReviewChannel(ToolResultChannel):
        def __init__(self):
            super().__init__()
            self.waiting = asyncio.Event()

        async def wait_for(self, call_id, timeout=None):
            self.waiting.set()
            return await super().wait_for(call_id, timeout)

    node = create_interactive_node(
        "research",
        _configure(
            real_agent_config,
            tools=["semantic_tools.write_semantic_model_plan_file"],
            scoped_context={
                "datasource": real_agent_config.current_datasource,
                "project_files": [".datus/semantic-model-plans/**"],
            },
        ),
        execution_mode="workflow",
    )
    node.tool_channel = ReviewChannel()
    # The API attaches review before prepare_turn mounts configurable tools.
    apply_semantic_plan_review(node)
    try:
        for turn in range(2):
            node.input = ConfigurableNodeInput(user_message="Review a semantic plan")
            await node._before_stream(StreamRunContext(node.input, ActionHistoryManager()))
            tool = next(tool for tool in node.tools if tool.name == "write_semantic_model_plan_file")
            call_id = f"plan-review-{turn}"
            node.tool_channel.waiting.clear()
            running = asyncio.create_task(
                tool.on_invoke_tool(
                    SimpleNamespace(tool_call_id=call_id),
                    json.dumps({"model_name": "schools", "plan_json": {"summary": f"Plan {turn}"}}),
                )
            )
            try:
                await asyncio.wait_for(node.tool_channel.waiting.wait(), timeout=2)
                preview = node.tool_review_previews[call_id]
                path = Path(real_agent_config.project_root) / preview["filePath"]
                baseline = path.read_text() if path.exists() else ""
                assert baseline == preview["oldText"]
                await node.tool_channel.publish(
                    call_id,
                    {"success": int(accept), "result": {"review_status": "accepted" if accept else "cancelled"}},
                )
                result = await asyncio.wait_for(running, timeout=2)
                assert bool(result["success"]) is accept
                if accept:
                    assert result["result"]["review_status"] == "accepted"
                    assert path.read_text() == preview["newText"]
                else:
                    assert not path.exists()
            finally:
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
    finally:
        await node.aclose()
