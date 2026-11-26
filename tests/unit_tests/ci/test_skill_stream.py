"""The Skill live-test driver must consume permission broker requests."""

import asyncio
from types import SimpleNamespace

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.cli.execution_state import InteractionBroker, InterruptController
from datus.schemas.action_bus import ActionBus
from datus.schemas.action_history import ActionHistory, ActionRole
from datus.schemas.interaction_event import InteractionEvent
from tests.helpers.skill_stream import run_skill_stream


class SkillNode:
    def __init__(self, root, event=None, *, hangs=False):
        self.agent_config = SimpleNamespace(home=root)
        self.interaction_broker = InteractionBroker()
        self.interrupt_controller = InterruptController()
        self.action_bus = ActionBus()
        self.event = event
        self.hangs = hangs
        self.closed = False
        self.answers = None

    def _get_or_create_broker(self):
        self.interaction_broker.reset_queue()
        return self.interaction_broker

    async def execute_stream(self, manager):
        if self.hangs:
            await asyncio.Event().wait()
        else:
            self.answers = await self.interaction_broker.request([self.event])
        yield ActionHistory.create_action(ActionRole.ASSISTANT, "finished", "done", {}, status="success")

    async def execute_stream_with_interactions(self, manager):
        # Exercise the actual production fan-in, not a replacement stream.
        from contextlib import aclosing

        async with aclosing(AgenticNode.execute_stream_with_interactions(self, manager)) as stream:
            async for action in stream:
                yield action

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_skill_driver_answers_fixture_permission_through_real_broker(tmp_path):
    node = SkillNode(
        tmp_path,
        InteractionEvent(
            title="Bash Permission",
            content="python scripts/generate_report.py",
            choices={"y": "Allow once", "n": "Deny"},
            default_choice="n",
        ),
    )
    actions = await run_skill_stream(node, None, timeout=2)
    assert node.answers == [["y"]]
    assert any(action.role == "interaction" and action.status == "success" for action in actions)
    assert actions[-1].action_type == "finished"
    assert node.closed


@pytest.mark.asyncio
async def test_skill_driver_rejects_unrelated_permission_instead_of_waiting(tmp_path):
    node = SkillNode(
        tmp_path,
        InteractionEvent(title="Bash Permission", content="curl https://example.org", choices={"y": "Allow once"}),
    )
    with pytest.raises(AssertionError, match="Unexpected Skill interaction"):
        await run_skill_stream(node, None, timeout=2)
    assert node.answers is None
    assert node.closed
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("action_bus:")]


@pytest.mark.asyncio
async def test_skill_driver_bounds_a_stalled_model_without_killing_pytest(tmp_path):
    node = SkillNode(tmp_path, hangs=True)
    with pytest.raises(AssertionError, match="Skill stream exceeded.*observed actions"):
        await run_skill_stream(node, None, timeout=0.01)
    assert node.closed
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("action_bus:")]
