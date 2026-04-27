# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Task-tool-shaped subagent emission for ``/bootstrap`` flows.

Every bootstrap stream that calls into an :class:`AgenticNode`'s
``execute_stream()`` should wrap the call with :func:`as_task_subagent`.
The wrapper produces the same :class:`ActionHistory` topology as
:class:`datus.tools.func_tool.sub_agent_task_tool.SubAgentTaskTool` —
outer ``task`` action (depth=0), forwarded inner stream (depth=1),
``subagent_complete`` terminator, and a paired ``complete_<call_id>``
SUCCESS / FAILED action — so the existing
:class:`InlineStreamingContext` daemon collapses each invocation as a
``⏺ {subagent_type}({description})`` group identical to chat's
``task(gen_sql, …)`` rendering.

Helpers in this module are pure async generators — no console writes,
no globals.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Dict, Optional

from datus.schemas.action_history import (
    SUBAGENT_COMPLETE_ACTION_TYPE,
    ActionHistory,
    ActionHistoryManager,
    ActionRole,
    ActionStatus,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


InnerStreamFactory = Callable[
    [ActionHistoryManager],
    AsyncGenerator[ActionHistory, None] | AsyncIterator[ActionHistory] | Awaitable[AsyncIterator[ActionHistory]],
]


def _to_inline_message(text: Optional[str]) -> str:
    """Collapse newlines/tabs/whitespace runs to a single space.

    The TUI streaming pinned-region renders each forwarded depth=1 action
    as a single line (``text.plain`` is added directly to a
    ``LiveDisplayLine`` segment); embedded ``\\n`` breaks line wrapping.
    """
    if not text:
        return ""
    return " ".join(text.split())


def _outer_action(
    call_id: str,
    subagent_type: str,
    description: str,
    start_time: datetime,
) -> ActionHistory:
    return ActionHistory(
        action_id=call_id,
        role=ActionRole.TOOL,
        action_type="task",
        messages=f"task({subagent_type}, {description})",
        input={
            "type": subagent_type,
            "description": description,
            "prompt": description,
            "function_name": "task",
            "_task_description": description,
        },
        output=None,
        status=ActionStatus.PROCESSING,
        start_time=start_time,
        end_time=None,
        depth=0,
        parent_action_id=None,
    )


def _terminal_action(
    call_id: str,
    outer: ActionHistory,
    status: ActionStatus,
    start_time: datetime,
    end_time: datetime,
    output: Optional[Dict[str, Any]] = None,
) -> ActionHistory:
    return ActionHistory(
        action_id=f"complete_{call_id}",
        role=ActionRole.TOOL,
        action_type="task",
        messages=outer.messages,
        input=outer.input,
        output=output,
        status=status,
        start_time=start_time,
        end_time=end_time,
        depth=0,
        parent_action_id=None,
    )


def _format_final_output(subagent_type: str, description: str, output: Dict[str, Any]) -> str:
    """Render a subagent's terminal ``output`` for the main REPL output.

    LLM-backed bootstrap subagents (``gen_sql_summary``, ``gen_semantic_model``,
    ``gen_metrics``) all return a ``BaseResult``-shaped dict whose ``response``
    field is the human-facing Markdown summary; surface it as-is. For
    subagents that don't follow that contract, fall back to a fenced JSON
    dump so the operator still sees the raw payload.
    """
    label = description or "<no description>"
    if isinstance(output, dict):
        response = output.get("response")
        if isinstance(response, str) and response.strip():
            return f"**{subagent_type}** ({label}):\n\n{response.strip()}"
    try:
        body = json.dumps(output, indent=2, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - defensive
        body = str(output)
    return f"**{subagent_type}** final output ({label}):\n```json\n{body}\n```"


def _subagent_complete(
    call_id: str,
    subagent_type: str,
    tool_count: int,
    status: ActionStatus,
    start_time: datetime,
    end_time: datetime,
) -> ActionHistory:
    return ActionHistory(
        action_id=str(uuid.uuid4()),
        role=ActionRole.SYSTEM,
        action_type=SUBAGENT_COMPLETE_ACTION_TYPE,
        messages="",
        input=None,
        output={"subagent_type": subagent_type, "tool_count": tool_count},
        status=status,
        start_time=start_time,
        end_time=end_time,
        depth=1,
        parent_action_id=call_id,
    )


async def as_task_subagent(
    subagent_type: str,
    description: str,
    inner_factory: InnerStreamFactory,
) -> AsyncGenerator[ActionHistory, None]:
    """Wrap an inner async generator as a chat-style ``task`` subagent group.

    Args:
        subagent_type: Identifier shown as the group header
            (e.g. ``"gen_sql_summary"``, ``"gen_metrics"``).
        description: Goal label rendered next to the header
            (e.g. ``"orders.sql"``).
        inner_factory: Callable that, given a fresh
            :class:`ActionHistoryManager`, returns the inner action stream
            from an :class:`AgenticNode.execute_stream` call. Returning a
            coroutine that resolves to an async iterator is also allowed,
            so callers can pass either ``node.execute_stream`` directly or
            a coroutine helper.

    Yields:
        :class:`ActionHistory` entries in the order
        ``outer_processing → forwarded depth=1 actions → subagent_complete → complete_<id>``.
    """
    call_id = str(uuid.uuid4())
    start_time = datetime.now()
    outer = _outer_action(call_id, subagent_type, description, start_time)
    yield outer

    inner_mgr = ActionHistoryManager()
    tool_count = 0
    final_status = ActionStatus.SUCCESS
    last_output: Optional[Dict[str, Any]] = None

    try:
        produced = inner_factory(inner_mgr)
        if hasattr(produced, "__await__"):
            inner_stream = await produced  # type: ignore[assignment]
        else:
            inner_stream = produced

        async for action in inner_stream:  # type: ignore[union-attr]
            action.depth = 1
            action.parent_action_id = call_id
            action.messages = _to_inline_message(action.messages)
            if action.role == ActionRole.TOOL.value or action.role == ActionRole.TOOL:
                tool_count += 1
            if action.status == ActionStatus.FAILED.value or action.status == ActionStatus.FAILED:
                final_status = ActionStatus.FAILED
            if action.output and isinstance(action.output, dict):
                last_output = action.output
            yield action
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("as_task_subagent caught: %s", exc, exc_info=True)
        final_status = ActionStatus.FAILED
        error_action = ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="error",
            messages=_to_inline_message(str(exc)),
            input_data={"function_name": "error"},
            status=ActionStatus.FAILED,
        )
        error_action.depth = 1
        error_action.parent_action_id = call_id
        yield error_action

    end_time = datetime.now()
    yield _subagent_complete(call_id, subagent_type, tool_count, final_status, start_time, end_time)
    yield _terminal_action(call_id, outer, final_status, start_time, end_time, output=last_output)

    if last_output is not None:
        yield message_action(
            _format_final_output(subagent_type, description, last_output),
            action_type="bootstrap_subagent_final_output",
        )


def message_action(
    text: str,
    *,
    role: ActionRole = ActionRole.ASSISTANT,
    status: ActionStatus = ActionStatus.SUCCESS,
    action_type: str = "bootstrap_message",
) -> ActionHistory:
    """Build a depth=0 ActionHistory entry for a human-readable status line.

    Designed to be used inside ``async def stream_*`` generators so callers
    can simply ``yield message_action("Loading...")`` without nesting
    another async-for.
    """
    now = datetime.now()
    return ActionHistory(
        action_id=str(uuid.uuid4()),
        role=role,
        action_type=action_type,
        messages=text,
        input=None,
        output=None,
        status=status,
        start_time=now,
        end_time=now,
        depth=0,
        parent_action_id=None,
    )


__all__ = ["as_task_subagent", "message_action", "InnerStreamFactory"]
