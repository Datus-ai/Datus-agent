"""Constructed long-loop checks using the real SDK loop, compactor and session store."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents import Agent, ModelSettings, Runner
from agents.exceptions import MaxTurnsExceeded

from datus.agent.node.context_rewriter import MidTurnCompactor
from datus.cli.execution_state import PendingInputQueue
from datus.cli.init_context import InitContext
from datus.configuration.agent_config import ModelConfig
from datus.models.openai_model import OpenAIModel
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from tests.unit_tests.cli.test_chat_commands import _make_chat_commands
from tests.unit_tests.models.conftest import RecordingScriptedModel


class InitScript(RecordingScriptedModel):
    def __init__(self, steps):
        super().__init__(["tool"] * len(steps) + ["final"])
        self.steps = steps
        self.instructions = []

    async def stream_response(self, system_instructions, input, *args, **kwargs):
        self.instructions.append(system_instructions)
        step = self.steps[self.turn] if self.turn < len(self.steps) else None
        if step:
            self._tool_name = step[0]
        async for event in super().stream_response(system_instructions, input, *args, **kwargs):
            if step:
                event.response.output[0].arguments = json.dumps(step[1])
            yield event


@pytest.mark.asyncio
@pytest.mark.parametrize("exhaust_budget", [False, True])
async def test_same_sdk_run_compacts_repeatedly_and_preserves_task(
    real_agent_config, mock_llm_create, tmp_path, exhaust_budget
):
    chat = _make_chat_commands(real_agent_config)
    node = chat.get_or_create_node()
    session = node._get_or_create_session()
    session_id = node.session_id
    node.model = OpenAIModel(
        ModelConfig(type="openai", model="gpt-4o", api_key="test", context_length=16384, max_tokens=1024)
    )
    node._compact_cfg.major.token_threshold = 0.65
    node._compact_cfg.minor.enabled = False
    # Only the summary model is stubbed; splitting, archival, session rewrite
    # and all SDK model-input filters execute their production implementations.
    node.model.summarize_items = AsyncMock(
        return_value={
            "content": "Earlier sources read. Continue the current goal using todos and pending questions.",
            "usage": {"output_tokens": 20},
        }
    )
    run = InitContext(tmp_path, {}, "Initialize orders and customers; preserve existing manual notes")
    queue = PendingInputQueue()
    read_paths = []
    for i in range(11):
        (tmp_path / f"topic{i}.md").write_text(
            f"SOURCE_{i} " + "A business rule is explicitly documented here with a source reference. " * 130
        )
    (tmp_path / "knowledge").mkdir()
    pending = tmp_path / "knowledge/pending-questions.md"
    pending.write_text("Unconfirmed: what is a customer account?")

    def read_file(path: str) -> FuncToolResult:
        """Read one scoped source document."""
        read_paths.append(path)
        if path == "topic2.md":
            correction = "Use booked status only; do not overwrite manual notes"
            queue.push(correction)
            run.record_user_updates([correction])
        return FuncToolResult(result={"content": (tmp_path / path).read_text()})

    def ask_user(questions: list[str]) -> FuncToolResult:
        """Clarify the current task with the user."""
        return FuncToolResult(result=[{"question": "Customer scope?", "answer": "Keep customer guidance separate"}])

    steps = [
        (
            "todo_write",
            {
                "todos_json": json.dumps(
                    [
                        {"title": "Orders", "content": "Document order guidance"},
                        {
                            "title": "Customers",
                            "content": "Read customer sources and preserve unknown business meaning",
                        },
                    ]
                )
            },
        ),
        ("read_file", {"path": "topic0.md"}),
        ("todo_update", {"todo_id": 1, "status": "completed"}),
        ("ask_user", {"questions": ["Customer scope?"]}),
    ]
    for i in range(1, 11):
        steps.append(("read_file", {"path": f"topic{i}.md"}))
        if i % 3 == 0:
            steps.extend([("todo_list", {}), ("todo_read", {"todo_id": 2})])
    steps.extend(
        [
            ("read_file", {"path": "knowledge/pending-questions.md"}),
            ("todo_update", {"todo_id": 2, "status": "completed"}),
        ]
    )
    scripted = InitScript(steps)
    max_turns = 12 if exhaust_budget else 30
    with run.activate(node):
        tools = run.wrap_tools(
            [trans_to_function_tool(read_file), trans_to_function_tool(ask_user), *node.plan_tool.available_tools()]
        )
        rewriter = MidTurnCompactor(node, refresh_instruction=lambda: node._inject_required_skills("Ordinary chat"))
        config = node.model._build_run_config(pending_input_queue=queue, session=session, context_rewriter=rewriter)
        config.tracing_disabled = True
        agent = Agent(
            name="init",
            model=scripted,
            instructions=node._inject_required_skills("Follow the current init scope"),
            tools=tools,
            model_settings=ModelSettings(max_tokens=1024),
            hooks=rewriter,
        )
        result = Runner.run_streamed(
            agent, input=run.state["scope"], session=session, max_turns=max_turns, run_config=config
        )
        try:
            async for _ in result.stream_events():
                pass
        except MaxTurnsExceeded as exc:
            assert exhaust_budget
            run.finish_execution(SimpleNamespace(success=False, error=str(exc)))
        else:
            assert not exhaust_budget
            run.finish_execution(SimpleNamespace(success=True))
        assert rewriter.compactions >= 2
        assert node.model.summarize_items.await_count >= 2
        assert all("Initialize orders and customers" in text for text in scripted.instructions)
        assert "Use booked status only" in scripted.instructions[-1]
        assert '<required_skill name="init">' in scripted.instructions[-1]
        assert "Keep customer guidance separate" in scripted.instructions[-1]
        todo_views = [json.dumps(items) for items in scripted.inputs if '"name": "todo_list"' in json.dumps(items)]
        assert any("completed" in text and "pending" in text for text in todo_views)
        assert any("Read customer sources" in json.dumps(items) for items in scripted.inputs)
        capacity = node.model.request_capacity()
        for instruction, items in zip(scripted.instructions, scripted.inputs):
            capacity.check(
                {
                    "instructions": instruction,
                    "input": items,
                    "tools": [{"name": t.name, "parameters": t.params_json_schema} for t in tools],
                }
            )
        assert node._session is session and node.session_id == session_id
        assert list(node._get_archive().dir.glob("**/*.jsonl")), "Original histories must be archived"
    run.close()
    if exhaust_budget:
        assert scripted.turn == max_turns
        assert run.state["status"] == "stopped"
        assert node.plan_tool.todo_list().result["completed"] == 1
        assert "turn" in run.state["stop_reason"].lower()
    else:
        assert run.state["status"] == "returned"
        assert node.plan_tool.todo_list().result["completed"] == 2
        assert "knowledge/pending-questions.md" in read_paths
        assert any("what is a customer account?" in json.dumps(items) for items in scripted.inputs[-2:])
        persisted = json.dumps(await session.get_items(), default=str)
        assert "done" in persisted and "Initialize orders and customers" in persisted
