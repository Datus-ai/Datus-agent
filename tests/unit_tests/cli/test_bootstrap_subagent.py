# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.bootstrap_subagent`."""

from __future__ import annotations

from typing import AsyncGenerator, List

import pytest

from datus.cli.bootstrap_subagent import as_task_subagent, message_action
from datus.schemas.action_history import (
    SUBAGENT_COMPLETE_ACTION_TYPE,
    ActionHistory,
    ActionHistoryManager,
    ActionRole,
    ActionStatus,
)


@pytest.mark.asyncio
async def test_as_task_subagent_emits_full_topology() -> None:
    """outer PROCESSING → inner forwarded depth=1 → subagent_complete → complete_<id>."""

    async def inner_factory(_mgr: ActionHistoryManager) -> AsyncGenerator[ActionHistory, None]:
        yield ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="read_query",
            messages="step 1",
            input_data={"function_name": "read_query"},
            status=ActionStatus.SUCCESS,
        )
        yield ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="read_query",
            messages="step 2",
            input_data={"function_name": "read_query"},
            status=ActionStatus.SUCCESS,
        )

    actions: List[ActionHistory] = []
    async for action in as_task_subagent("gen_sql_summary", "orders.sql", inner_factory):
        actions.append(action)

    assert len(actions) == 5  # outer + 2 inner + subagent_complete + terminal
    outer, inner1, inner2, complete, terminal = actions

    assert outer.action_type == "task"
    assert outer.input["type"] == "gen_sql_summary"
    assert outer.input["_task_description"] == "orders.sql"
    assert outer.status == ActionStatus.PROCESSING.value

    for inner in (inner1, inner2):
        assert inner.depth == 1
        assert inner.parent_action_id == outer.action_id

    assert complete.action_type == SUBAGENT_COMPLETE_ACTION_TYPE
    assert complete.depth == 1
    assert complete.parent_action_id == outer.action_id
    assert complete.output["tool_count"] == 2

    assert terminal.action_id == f"complete_{outer.action_id}"
    assert terminal.status == ActionStatus.SUCCESS.value


@pytest.mark.asyncio
async def test_as_task_subagent_marks_failed_when_inner_yields_failure() -> None:
    async def inner_factory(_mgr):
        yield ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="read_query",
            messages="boom",
            input_data={"function_name": "read_query"},
            status=ActionStatus.FAILED,
        )

    actions = [a async for a in as_task_subagent("gen_sql_summary", "x", inner_factory)]
    assert actions[-1].status == ActionStatus.FAILED.value
    assert actions[-2].action_type == SUBAGENT_COMPLETE_ACTION_TYPE
    assert actions[-2].status == ActionStatus.FAILED.value


@pytest.mark.asyncio
async def test_as_task_subagent_handles_inner_exception() -> None:
    async def inner_factory(_mgr):
        raise RuntimeError("crash")
        yield  # pragma: no cover - unreachable

    actions = [a async for a in as_task_subagent("gen_sql_summary", "x", inner_factory)]

    # We expect: outer PROCESSING + synthesized error sub-action + subagent_complete + terminal
    assert actions[0].status == ActionStatus.PROCESSING.value
    error_action = next(a for a in actions if a.action_type == "error")
    assert error_action.depth == 1
    assert "crash" in error_action.messages
    assert actions[-1].status == ActionStatus.FAILED.value


@pytest.mark.asyncio
async def test_as_task_subagent_renders_response_field_as_markdown_when_present() -> None:
    """LLM subagents (sql_summary/semantic_model/metrics) carry ``response`` markdown — surface it raw."""

    final_payload = {
        "response": "## Summary\n- generated `orders.yml`\n- 3 columns indexed",
        "sql_summary_file": "subject/sql_summaries/orders.yml",
        "success": True,
    }

    async def inner_factory(_mgr: ActionHistoryManager) -> AsyncGenerator[ActionHistory, None]:
        yield ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="sql_summary_response",
            messages="done",
            input_data={"function_name": "sql_summary"},
            output_data=final_payload,
            status=ActionStatus.SUCCESS,
        )

    actions = [a async for a in as_task_subagent("gen_sql_summary", "orders.sql", inner_factory)]

    assert len(actions) == 5
    final = actions[-1]
    assert final.action_type == "bootstrap_subagent_final_output"
    assert final.depth == 0
    assert final.parent_action_id is None
    assert final.role == ActionRole.ASSISTANT.value
    assert "**gen_sql_summary** (orders.sql)" in final.messages
    assert "## Summary" in final.messages
    assert "generated `orders.yml`" in final.messages
    assert "```json" not in final.messages

    terminal = actions[-2]
    assert terminal.action_id.startswith("complete_")
    assert terminal.output == final_payload


@pytest.mark.asyncio
async def test_as_task_subagent_falls_back_to_json_dump_when_no_response_field() -> None:
    """Non-LLM subagents (no ``response`` field) still get a JSON dump so payload is not lost."""

    final_payload = {"sub_agent_yaml": "/tmp/sub.yml", "success": True}

    async def inner_factory(_mgr: ActionHistoryManager) -> AsyncGenerator[ActionHistory, None]:
        yield ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="persist_yaml",
            messages="done",
            input_data={"function_name": "persist_yaml"},
            output_data=final_payload,
            status=ActionStatus.SUCCESS,
        )

    actions = [a async for a in as_task_subagent("save_subagents", "dashboard", inner_factory)]
    final = actions[-1]
    assert final.action_type == "bootstrap_subagent_final_output"
    assert "**save_subagents** final output (dashboard)" in final.messages
    assert "```json" in final.messages
    assert "sub_agent_yaml" in final.messages


@pytest.mark.asyncio
async def test_as_task_subagent_falls_back_to_json_when_response_blank() -> None:
    """Empty/whitespace-only ``response`` falls back to JSON dump rather than printing an empty block."""

    final_payload = {"response": "   ", "semantic_models": ["a.yml"], "success": True}

    async def inner_factory(_mgr: ActionHistoryManager) -> AsyncGenerator[ActionHistory, None]:
        yield ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="gen_metrics_response",
            messages="done",
            input_data={"function_name": "gen_metrics"},
            output_data=final_payload,
            status=ActionStatus.SUCCESS,
        )

    actions = [a async for a in as_task_subagent("gen_metrics", "dashboard", inner_factory)]
    final = actions[-1]
    assert "```json" in final.messages
    assert "semantic_models" in final.messages


@pytest.mark.asyncio
async def test_as_task_subagent_skips_final_output_when_inner_yields_no_dict_output() -> None:
    """No final-output message is appended when no inner action carries a dict ``output``."""

    async def inner_factory(_mgr: ActionHistoryManager) -> AsyncGenerator[ActionHistory, None]:
        yield ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="read_query",
            messages="step",
            input_data={"function_name": "read_query"},
            status=ActionStatus.SUCCESS,
        )

    actions = [a async for a in as_task_subagent("gen_sql_summary", "x", inner_factory)]
    assert all(a.action_type != "bootstrap_subagent_final_output" for a in actions)
    assert actions[-1].action_id.startswith("complete_")


@pytest.mark.asyncio
async def test_as_task_subagent_emits_final_output_even_on_failed_inner() -> None:
    """A FAILED inner action with a dict output still surfaces its final output to main output."""

    failure_payload = {
        "response": "## Failed\nValidation error: missing primary key",
        "success": False,
    }

    async def inner_factory(_mgr: ActionHistoryManager) -> AsyncGenerator[ActionHistory, None]:
        yield ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="gen_metrics_response",
            messages="boom",
            input_data={"function_name": "gen_metrics"},
            output_data=failure_payload,
            status=ActionStatus.FAILED,
        )

    actions = [a async for a in as_task_subagent("gen_metrics", "dashboard", inner_factory)]
    assert actions[-1].action_type == "bootstrap_subagent_final_output"
    assert actions[-1].status == ActionStatus.SUCCESS.value
    assert "missing primary key" in actions[-1].messages
    terminal = actions[-2]
    assert terminal.status == ActionStatus.FAILED.value


@pytest.mark.asyncio
async def test_as_task_subagent_strips_newlines_from_inner_messages() -> None:
    """Depth=1 forwarded actions must be single-line (TUI pinned region renders one line per action)."""

    async def inner_factory(_mgr: ActionHistoryManager) -> AsyncGenerator[ActionHistory, None]:
        yield ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="read_query",
            messages="line1\nline2\tline3  line4\r\ntail",
            input_data={"function_name": "read_query"},
            status=ActionStatus.SUCCESS,
        )

    actions = [a async for a in as_task_subagent("gen_sql_summary", "x", inner_factory)]
    inner = actions[1]
    assert inner.depth == 1
    assert inner.messages == "line1 line2 line3 line4 tail"
    assert "\n" not in inner.messages
    assert "\r" not in inner.messages
    assert "\t" not in inner.messages


@pytest.mark.asyncio
async def test_as_task_subagent_strips_newlines_from_exception_message() -> None:
    """Exception strings (which often carry tracebacks with ``\\n``) must be inlined too."""

    async def inner_factory(_mgr):
        raise RuntimeError("first line\nsecond line")
        yield  # pragma: no cover - unreachable

    actions = [a async for a in as_task_subagent("gen_sql_summary", "x", inner_factory)]
    error_action = next(a for a in actions if a.action_type == "error")
    assert error_action.depth == 1
    assert error_action.messages == "first line second line"
    assert "\n" not in error_action.messages


@pytest.mark.asyncio
async def test_as_task_subagent_preserves_newlines_in_final_output_message() -> None:
    """``bootstrap_subagent_final_output`` is depth=0 markdown — its ``\\n`` must be preserved."""

    final_payload = {"response": "## Title\n- bullet", "success": True}

    async def inner_factory(_mgr: ActionHistoryManager) -> AsyncGenerator[ActionHistory, None]:
        yield ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="sql_summary_response",
            messages="done",
            input_data={"function_name": "sql_summary"},
            output_data=final_payload,
            status=ActionStatus.SUCCESS,
        )

    actions = [a async for a in as_task_subagent("gen_sql_summary", "orders.sql", inner_factory)]
    final = actions[-1]
    assert final.action_type == "bootstrap_subagent_final_output"
    assert final.depth == 0
    assert "## Title\n" in final.messages
    assert "- bullet" in final.messages


def test_message_action_builds_assistant_entry() -> None:
    action = message_action("hello", role=ActionRole.ASSISTANT)
    assert action.role == ActionRole.ASSISTANT.value
    assert action.depth == 0
    assert action.messages == "hello"
    assert action.status == ActionStatus.SUCCESS.value


def test_message_action_failed_keeps_role_but_marks_status() -> None:
    action = message_action("bad", status=ActionStatus.FAILED)
    assert action.status == ActionStatus.FAILED.value
