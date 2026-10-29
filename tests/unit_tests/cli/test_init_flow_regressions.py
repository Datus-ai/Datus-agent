"""One init run shares the chat session and restores its ordinary tool surface."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from agents.exceptions import MaxTurnsExceeded

from datus.cli.init_context import InitContext
from datus.tools.permission.profiles import get_profile
from tests.unit_tests.cli.test_chat_commands import _make_chat_commands
from tests.unit_tests.mock_llm_model import MockLLMResponse, MockToolCall


def chat_for(config, tmp_path):
    config.permissions_config = get_profile("auto")
    config.agentic_nodes["chat"]["workspace_root"] = str(tmp_path)
    chat = _make_chat_commands(config)
    return chat


def test_topics_todos_and_followup_use_the_existing_chat(real_agent_config, mock_llm_create, tmp_path):
    chat = chat_for(real_agent_config, tmp_path)
    (tmp_path / "README.md").write_text("School directory and meal program guidance")
    mock_llm_create.reset(responses=[MockLLMResponse(content="Earlier conversation")])
    chat.execute_chat_command("Remember the current project")
    node = chat.current_node
    session_id = node.session_id
    original_tools = {t.name for t in node.tools}
    servers, web = node.mcp_servers, getattr(node, "_builtin_web_tools", None)
    calls = [
        MockToolCall(
            "todo_write",
            {
                "todos_json": json.dumps(
                    [
                        {"title": "School directory", "content": "Review schools"},
                        {"title": "Meal program", "content": "Review frpm and capture unknown meaning"},
                    ]
                )
            },
        ),
        MockToolCall("describe_table", {"table_name": "schools"}),
        MockToolCall("write_file", {"path": "knowledge/schools.md", "content": "School guidance (README.md)"}),
        MockToolCall("todo_update", {"todo_id": 1, "status": "completed"}),
        MockToolCall("describe_table", {"table_name": "frpm"}),
        MockToolCall("write_file", {"path": "knowledge/meal.md", "content": "Meal program guidance (README.md)"}),
        MockToolCall(
            "write_file", {"path": "knowledge/pending-questions.md", "content": "Confirm meal-program semantics"}
        ),
        MockToolCall("todo_update", {"todo_id": 2, "status": "completed"}),
    ]
    mock_llm_create.reset(
        responses=[MockLLMResponse(content="Both topics documented; confirm meal semantics", tool_calls=calls)]
    )
    run = InitContext(tmp_path, {}, "School and meal topics")
    chat.execute_chat_command("Initialize these topics", init_context=run)
    run.close()
    assert chat.current_node is node and node.session_id == session_id
    assert run.state["status"] == "returned"
    assert node.plan_tool.todo_list().result["completed"] == 2
    assert all(item["output"]["success"] == 1 for item in mock_llm_create.tool_results)
    assert (tmp_path / "knowledge/schools.md").exists() and (tmp_path / "knowledge/meal.md").exists()
    init_tools = set(mock_llm_create.call_history[-1]["tools"])
    assert '<required_skill name="init">' in mock_llm_create.call_history[-1]["instruction"]
    assert {"todo_list", "todo_read", "todo_write", "todo_update"} <= init_tools
    assert not {"confirm_plan", "execute_sql", "bash", "task"} & init_tools
    assert not node.is_in_plan_mode()
    assert {t.name for t in node.tools} == original_tools
    assert node.mcp_servers is servers and getattr(node, "_builtin_web_tools", None) is web
    assert getattr(node, "_init_context", None) is None
    mock_llm_create.reset(
        responses=[
            MockLLMResponse(
                content="User confirmation recorded",
                tool_calls=[
                    MockToolCall("write_file", {"path": "README.md", "content": "User-requested project note"}),
                    MockToolCall("read_file", {"path": "knowledge/pending-questions.md"}),
                    MockToolCall(
                        "edit_file",
                        {
                            "path": "knowledge/meal.md",
                            "old_string": "Meal program guidance",
                            "new_string": "Meal program means subsidized meals (confirmed by user)",
                        },
                    ),
                    MockToolCall(
                        "write_file", {"path": "knowledge/pending-questions.md", "content": "Meal meaning resolved"}
                    ),
                ],
            )
        ]
    )
    chat.execute_chat_command("Meal program means subsidized meals; update the pending question")
    assert chat.current_node is node and node.session_id == session_id
    assert "execute_sql" in mock_llm_create.call_history[-1]["tools"]
    assert '<required_skill name="init">' not in mock_llm_create.call_history[-1]["instruction"]
    assert "confirmed by user" in (tmp_path / "knowledge/meal.md").read_text()
    assert (tmp_path / "README.md").read_text() == "User-requested project note"


@pytest.mark.parametrize("failure", ["budget", "provider", "interrupt"])
def test_failure_does_not_restart_or_leak_init_tools(
    real_agent_config, mock_llm_create, tmp_path, monkeypatch, failure
):
    chat = chat_for(real_agent_config, tmp_path)
    mock_llm_create.reset(responses=[MockLLMResponse(content="Existing chat")])
    chat.execute_chat_command("Hello")
    node, calls = chat.current_node, []
    original_tools = {t.name for t in node.tools}
    run = InitContext(tmp_path, {}, "Two topics")
    node.plan_tool.todo_write(json.dumps([{"title": "Meal program", "content": "Inspect frpm"}]))
    (tmp_path / "AGENTS.md").write_text("Existing school guidance")
    original = type(mock_llm_create).generate_with_tools_stream

    async def fail(self, *args, **kwargs):
        calls.append(node.session_id)
        node.pending_input_queue.push("Keep the existing human notes")
        if failure == "budget":
            raise MaxTurnsExceeded("Max turns exceeded")
        if failure == "provider":
            raise RuntimeError("Provider unavailable")
        from datus.cli.execution_state import ExecutionInterrupted

        node.interrupt_controller.interrupt()
        raise ExecutionInterrupted()
        yield  # Make this a model stream.

    monkeypatch.setattr(type(mock_llm_create), "generate_with_tools_stream", fail)
    chat.execute_chat_command("Init", init_context=run)
    run.close()
    assert len(calls) == 1 and chat.current_node is node
    assert run.state["status"] == "stopped" and run.state["stop_reason"]
    assert node.plan_tool.todo_list().result["items"][0]["status"] == "pending"
    assert (tmp_path / "AGENTS.md").read_text() == "Existing school guidance"
    assert {t.name for t in node.tools} == original_tools
    assert getattr(node, "_init_context", None) is None
    monkeypatch.setattr(type(mock_llm_create), "generate_with_tools_stream", original)
    mock_llm_create.reset(responses=[MockLLMResponse(content="Ordinary chat resumed")])
    chat.execute_chat_command("Continue normal chat")
    assert "execute_sql" in mock_llm_create.call_history[-1]["tools"]
    assert '<required_skill name="init">' not in mock_llm_create.call_history[-1]["instruction"]


@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("cancellation", ["exception", "keyboard", "cooperative", "rollback"])
def test_cancelled_init_cannot_reuse_previous_success(
    real_agent_config, mock_llm_create, tmp_path, monkeypatch, interactive, cancellation
):
    from datus.cli.execution_state import ExecutionInterrupted

    chat = chat_for(real_agent_config, tmp_path)
    mock_llm_create.reset(responses=[MockLLMResponse(content="Previous successful chat")])
    chat.execute_chat_command("Hello")
    node = chat.current_node
    previous_result = node.result
    assert previous_result.success is True
    original_tools = {tool.name for tool in node.tools}
    run = InitContext(tmp_path, {}, "Schools")
    (tmp_path / "AGENTS.md").write_text("School notes written")
    display = MagicMock()
    display.__enter__.return_value = display
    display.__exit__.return_value = False
    display.unanswered_rollback_requested = False
    display.interrupt_requested = False
    display.has_streamed_response = False

    async def cancelled_stream(*args, **kwargs):
        if cancellation == "keyboard":
            raise KeyboardInterrupt()
        if cancellation == "exception":
            # Cancellation need not leave the reusable controller interrupted.
            raise ExecutionInterrupted("cancelled")
        node.interrupt_controller.interrupt()
        display.interrupt_requested = True
        display.unanswered_rollback_requested = cancellation == "rollback"
        return
        yield

    monkeypatch.setattr(node, "execute_stream_with_interactions", cancelled_stream)
    if not interactive:
        monkeypatch.setattr(chat, "_create_new_node", lambda *args, **kwargs: node)
    with patch("datus.cli.chat_commands.ActionHistoryDisplay") as display_class:
        display_class.return_value.display_streaming_actions.return_value = display
        chat._execute_chat("Initialize schools", interactive=interactive, init_context=run)
    run.close()
    assert node.result is previous_result
    assert run.state["status"] == "stopped" and "interrupted" in run.state["stop_reason"].lower()
    assert (tmp_path / "AGENTS.md").read_text() == "School notes written"
    assert {tool.name for tool in node.tools} == original_tools
    assert getattr(node, "_init_context", None) is None
    assert node.interrupt_controller.is_interrupted == (not interactive and cancellation != "exception")


def test_setup_error_restores_tools_and_current_session(real_agent_config, mock_llm_create, tmp_path):
    chat = chat_for(real_agent_config, tmp_path)
    node = chat.get_or_create_node()
    tools, servers, web = node.tools, node.mcp_servers, getattr(node, "_builtin_web_tools", None)
    run = InitContext(tmp_path, {}, "docs")
    with patch("datus.cli.chat_commands.ActionHistoryDisplay", side_effect=RuntimeError("display setup failed")):
        chat.execute_chat_command("init", init_context=run)
    run.close()
    assert chat.current_node is node
    assert node.tools is tools and node.mcp_servers is servers and getattr(node, "_builtin_web_tools", None) is web
    assert getattr(node, "_init_context", None) is None
    assert "display setup failed" in run.state["stop_reason"]


def test_refresh_reapplies_limits_and_cleanup_restores_surface(real_agent_config, mock_llm_create, tmp_path):
    chat = chat_for(real_agent_config, tmp_path)
    node = chat.get_or_create_node()
    original_tools, servers, web = node.tools, node.mcp_servers, getattr(node, "_builtin_web_tools", None)
    run = InitContext(tmp_path, {}, "docs")
    with pytest.raises(RuntimeError, match="after refresh"):
        with run.activate(node):
            for _ in range(2):
                node.setup_tools()
                node._get_session_system_prompt()
                assert not {"execute_sql", "bash", "task", "confirm_plan"} & {t.name for t in node.tools}
                assert {"todo_list", "todo_read", "todo_write", "todo_update"} <= {t.name for t in node.tools}
                assert node.mcp_servers == {} and getattr(node, "_builtin_web_tools", None) is None
            raise RuntimeError("after refresh")
    assert (
        node.tools is original_tools
        and node.mcp_servers is servers
        and getattr(node, "_builtin_web_tools", None) is web
    )
    assert getattr(node, "_init_context", None) is None
    run.finish_execution(SimpleNamespace(success=False, error="after refresh"))
    assert run.state["status"] == "stopped"


def test_late_user_input_is_retained_without_resetting_the_run(
    real_agent_config, mock_llm_create, tmp_path, monkeypatch
):
    chat = chat_for(real_agent_config, tmp_path)
    node = chat.get_or_create_node()
    original = type(mock_llm_create).generate_with_tools_stream
    mock_llm_create.reset(responses=[MockLLMResponse(content="Docs reviewed")])
    correction = "Also preserve the finance team's manual annotations"

    async def late_input(self, *args, **kwargs):
        async for action in original(self, *args, **kwargs):
            yield action
        node.pending_input_queue.push(correction)

    monkeypatch.setattr(type(mock_llm_create), "generate_with_tools_stream", late_input)
    run = InitContext(tmp_path, {}, "Review project docs")
    chat.execute_chat_command("init", init_context=run)
    run.close()
    assert run.state["status"] == "stopped" and "unprocessed" in run.state["stop_reason"]
    assert len(mock_llm_create.call_history) == 1
    assert node.pending_input_queue.snapshot() == [correction]
    assert run.user_context()[-1]["text"] == correction
    assert chat.current_node is node and getattr(node, "_init_context", None) is None


@pytest.mark.asyncio
async def test_plugin_refresh_keeps_metadata_guard_callable(real_agent_config, mock_llm_create, tmp_path):
    import asyncio
    from dataclasses import replace

    from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool

    node = chat_for(real_agent_config, tmp_path).get_or_create_node()
    calls = []

    def describe_table(table_name: str) -> FuncToolResult:
        """Return metadata only."""
        calls.append(table_name)
        return FuncToolResult(result={"columns": [{"name": "id"}]})

    def transform():
        if getattr(node, "_tool_transformers_applied", False):
            return
        transformed = []
        for tool in node.tools:

            async def invoke(context, arguments, original=tool.on_invoke_tool):
                return await original(context, arguments)

            transformed.append(replace(tool, on_invoke_tool=invoke))
        node.tools = transformed
        node._tool_transformers_applied = True

    run = InitContext(tmp_path, {}, "orders")
    with run.activate(node), patch.object(node, "_ensure_tool_transformers", side_effect=transform):
        node.tools = [trans_to_function_tool(describe_table)]
        for table in ["orders", "customers"]:
            node._tool_transformers_applied = False
            node._apply_init_tool_scope()
            tool = next(t for t in node.tools if t.name == "describe_table")
            result = await asyncio.wait_for(tool.on_invoke_tool(None, json.dumps({"table_name": table})), timeout=1)
            assert result["success"] == 1
    assert calls == ["orders", "customers"]
