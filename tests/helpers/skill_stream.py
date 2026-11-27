"""Drive the same interactive Skill stream as a client, with bounded waits."""

import asyncio
import re
import shlex
from contextlib import aclosing
from pathlib import Path


def _fixture_permission(event, home, workspace):
    """Allow only reading the installed fixture or invoking its report script."""
    if "y" not in event.get("choices", {}):
        return False
    # Keep the allowlist anchored under home without resolving symlinks in the
    # fixture itself: a fixture symlink must not authorize an external target.
    skill = Path(home).resolve() / "skills" / "report-generator"
    script = skill / "scripts" / "generate_report.py"
    content = event.get("content", "")
    if event.get("title") == "Permission":
        match = re.fullmatch(
            r"### External Filesystem Access\n\n"
            r"\*\*Tool:\*\* `filesystem_tools\.(read_file|list_directory)`\n"
            r"\*\*Path:\*\* `([^`\n]+)`  _\(outside project root\)_\n",
            content,
        )
        if not match:
            return False
        operation, raw_path = match.groups()
        path = Path(raw_path)
        if not path.is_absolute():
            return False
        if operation == "read_file":
            return path.resolve() in {skill / "SKILL.md", script} and path.is_file()
        return path.resolve() in {skill, skill / "scripts"} and path.is_dir()
    if event.get("title") != "Bash Permission":
        return False
    match = re.fullmatch(
        r"### Bash Command Permission\n\n```bash\n([^\n]+)\n```\n\n"
        r"\*\*Reason:\*\* [^\n]*\n(?:\*\*AI review:\*\* [^\n]*\n)?",
        content,
    )
    if not match:
        return False
    command = match.group(1)
    # No shell expansion, redirection, pipelines or chained commands. shlex
    # handles quoted paths, but must never hide shell execution operators.
    if re.search(r"[$`\r\n<>|;&]", command):
        return False
    try:
        args = shlex.split(command)
    except ValueError:
        return False
    if len(args) < 2 or args[0] not in {"python", "python3"}:
        return False
    path = Path(args[1])
    if not path.is_absolute():
        path = Path(workspace) / path
    return (
        path.resolve() == script
        and path.is_file()
        and (not args[2:] or args[2:] in [["--format", value] for value in ("json", "csv", "markdown")])
    )


async def run_skill_stream(node, action_manager, *, timeout=240):
    """Answer only fixture Skill permissions; fail promptly on other questions.

    A raw execute_stream consumer never sees broker requests. In particular,
    running a report Skill's script may require bash or external-path approval,
    even after skills.* was approved. Use the merged client stream so those
    requests cannot silently wait for a human in nightly.
    """
    actions = []
    try:
        async with asyncio.timeout(timeout):
            async with aclosing(node.execute_stream_with_interactions(action_manager)) as stream:
                async for action in stream:
                    actions.append(action)
                    if action.role != "interaction" or action.status != "processing":
                        continue
                    events = (action.input or {}).get("events", [])
                    answers = []
                    for event in events:
                        allowed = _fixture_permission(event, node.agent_config.home, node.agent_config.project_root)
                        if not allowed:
                            raise AssertionError(f"Unexpected Skill interaction: {event}")
                        # Approve this isolated fixture operation once; never
                        # mutate the application's permission configuration.
                        answers.append(["y"])
                    assert answers, f"Empty Skill interaction: {action.input}"
                    accepted = await node.interaction_broker.submit(action.action_id, answers)
                    assert accepted, f"Skill interaction was no longer pending: {action.action_id}"
    except TimeoutError as exc:
        progress = [(action.action_type, str(action.status)) for action in actions]
        raise AssertionError(f"Skill stream exceeded {timeout}s; observed actions: {progress}") from exc
    finally:
        node.interaction_broker.close()
        await node.aclose()
    return actions
