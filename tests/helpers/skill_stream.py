"""Drive the same interactive Skill stream as a client, with bounded waits."""

import asyncio
from contextlib import aclosing


async def run_skill_stream(node, action_manager, *, timeout=240):
    """Answer only fixture Skill permissions; fail promptly on other questions.

    A raw execute_stream consumer never sees broker requests. In particular,
    running a report Skill's script may require bash or external-path approval,
    even after skills.* was approved. Use the merged client stream so those
    requests cannot silently wait for a human in nightly.
    """
    actions = []
    fixture_home = str(node.agent_config.home)
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
                        content = event.get("content", "")
                        allowed = (
                            event.get("title") in {"Permission", "Bash Permission"}
                            and "y" in event.get("choices", {})
                            and any(
                                marker in content for marker in (fixture_home, "report-generator", "generate_report.py")
                            )
                        )
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
