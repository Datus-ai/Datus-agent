"""The Skill live-test driver must consume permission broker requests."""

import asyncio
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.cli.execution_state import InteractionBroker, InterruptController
from datus.schemas.action_bus import ActionBus
from datus.schemas.action_history import ActionHistory, ActionRole
from datus.schemas.interaction_event import InteractionEvent
from datus.tools.permission.permission_hooks import PermissionHooks
from tests.helpers.skill_stream import run_skill_stream


class SkillNode:
    def __init__(self, root, event=None, *, hangs=False):
        self.agent_config = SimpleNamespace(home=root, project_root=root / "workspace")
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


@pytest.fixture
def fixture_home(tmp_path):
    home = tmp_path / "fixture home"
    (home / "workspace").mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / "data" / "skills" / "report-generator"
    shutil.copytree(source, home / "skills" / "report-generator")
    return home


def bash_event(command):
    # Format the request using the production permission hook.
    content = PermissionHooks._format_bash_prompt_content(
        None, command, SimpleNamespace(ask_segments=(), reason="Fixture command")
    )
    return InteractionEvent(title="Bash Permission", content=content, choices={"y": "Allow once", "n": "Deny"})


def file_event(path, operation="read_file"):
    return InteractionEvent(
        title="Permission",
        content=(
            "### External Filesystem Access\n\n"
            f"**Tool:** `filesystem_tools.{operation}`\n"
            f"**Path:** `{path}`  _(outside project root)_\n"
        ),
        choices={"y": "Allow once", "n": "Deny"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("options", ["", " --format json", " --format csv", " --format markdown"])
async def test_skill_driver_answers_fixture_permission_through_real_broker(fixture_home, relative, options):
    script = fixture_home / "skills/report-generator/scripts/generate_report.py"
    script_arg = "../skills/report-generator/scripts/generate_report.py" if relative else str(script)
    node = SkillNode(
        fixture_home,
        bash_event(f"python3 {shlex.quote(script_arg)}{options}"),
    )
    actions = await run_skill_stream(node, None, timeout=2)
    assert node.answers == [["y"]]
    assert any(action.role == "interaction" and action.status == "success" for action in actions)
    assert actions[-1].action_type == "finished"
    assert node.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "echo report-generator",
        "echo generate_report.py",
        "echo {home}",
        "python {script} && echo unrelated",
        "python {script}; echo unrelated",
        "python {script} | cat",
        "python {script} > report-generator.txt",
        "python {script} $(echo unrelated)",
        "python {script} `echo unrelated`",
        "python {script}\necho unrelated",
        "python -c 'print(42)' {script}",
        "python /elsewhere/report-generator/scripts/generate_report.py",
        "python scripts/generate_report.py",
        "python {script} --input /elsewhere/data.json",
        "python {script} --format json --output other",
        "bash -c 'python {script}'",
    ],
)
async def test_skill_driver_rejects_fixture_names_in_unapproved_commands(fixture_home, command):
    command = command.format(
        home=shlex.quote(str(fixture_home)),
        script=shlex.quote(str(fixture_home / "skills/report-generator/scripts/generate_report.py")),
    )
    node = SkillNode(fixture_home, bash_event(command))
    with pytest.raises(AssertionError, match="Unexpected Skill interaction"):
        await run_skill_stream(node, None, timeout=2)
    assert node.answers is None
    assert node.closed
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("action_bus:")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,path,allowed",
    [
        ("read_file", "skills/report-generator/SKILL.md", True),
        ("read_file", "skills/report-generator/scripts/generate_report.py", True),
        ("list_directory", "skills/report-generator", True),
        ("list_directory", "skills/report-generator/scripts", True),
        ("write_file", "skills/report-generator/SKILL.md", False),
        ("read_file", "skills/report-generator/../../unrelated.txt", False),
        ("list_directory", ".", False),
    ],
)
async def test_skill_driver_validates_fixture_file_operations(fixture_home, operation, path, allowed):
    (fixture_home / "unrelated.txt").write_text("unrelated")
    node = SkillNode(fixture_home, file_event(fixture_home / path, operation))
    if allowed:
        await run_skill_stream(node, None, timeout=2)
        assert node.answers == [["y"]]
    else:
        with pytest.raises(AssertionError, match="Unexpected Skill interaction"):
            await run_skill_stream(node, None, timeout=2)
        assert node.answers is None
    assert node.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("bash", [False, True])
async def test_skill_driver_rejects_fixture_symlink_escape(fixture_home, bash):
    external = fixture_home.parent / "external.py"
    external.write_text("print('unrelated')")
    script = fixture_home / "skills/report-generator/scripts/generate_report.py"
    script.unlink()
    script.symlink_to(external)
    event = bash_event(f"python {shlex.quote(str(script))}") if bash else file_event(script)
    node = SkillNode(fixture_home, event)
    with pytest.raises(AssertionError, match="Unexpected Skill interaction"):
        await run_skill_stream(node, None, timeout=2)
    assert node.answers is None
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
