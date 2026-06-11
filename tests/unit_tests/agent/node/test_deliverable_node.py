# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for :class:`DeliverableAgenticNode.execute_stream` retry loop.

Focus: the loop's response to ``ValidationHook.final_report`` recording a
blocking failure. The real hook fires Layer A + Layer B at ``on_end``; here we
substitute a lightweight fake hook that returns a controlled sequence of
reports so we can assert exactly how many times the stream is driven and what
state ends up on the final :class:`NodeResult`.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from datus.schemas.action_history import ActionHistoryManager
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.validation import CheckResult, TableTarget, ValidationReport
from tests.unit_tests.mock_llm_model import build_simple_response


class _FakeValidationHook:
    """Per-attempt report sequencer used to drive the retry loop.

    ``reports[i]`` is the value returned by :attr:`final_report` during
    attempt ``i + 1``. ``reset_session`` advances the index by one; the
    pre-loop reset in ``execute_stream`` is what seeds the very first
    attempt's value.
    """

    def __init__(self, reports: List[Optional[ValidationReport]]):
        self._reports = reports
        self._index = -1
        self._session_targets: list = []

    @property
    def final_report(self) -> Optional[ValidationReport]:
        if 0 <= self._index < len(self._reports):
            return self._reports[self._index]
        return None

    @property
    def session_targets(self) -> list:
        return list(self._session_targets)

    def reset_session(self) -> None:
        self._index += 1
        self._session_targets = []

    def set_parent_session(self, session) -> None:  # noqa: D401
        """Stub — tests don't exercise the parent-session fork path."""
        self._parent_session = session

    # Minimal AgentHooks surface — the mock LLM does not invoke these, but
    # CompositeHooks may iterate hook lists during construction.
    async def on_run_start(self, context, agent) -> None:  # pragma: no cover
        return None

    async def on_run_end(self, context, agent, output) -> None:  # pragma: no cover
        return None

    async def on_tool_start(self, context, agent, tool) -> None:  # pragma: no cover
        return None

    async def on_tool_end(self, context, agent, tool, result) -> None:  # pragma: no cover
        return None

    async def on_end(self, context, agent, output) -> None:  # pragma: no cover
        return None


def _make_blocking_report() -> ValidationReport:
    target = TableTarget(database="d", table="t")
    return ValidationReport(
        target=target,
        checks=[
            CheckResult(
                name="on_end_blocker",
                passed=False,
                severity="blocking",
                source="builtin",
                error="synthetic blocking failure for retry-loop test",
            )
        ],
    )


def _count_stream_calls(mock_llm_create) -> int:
    return sum(1 for c in mock_llm_create.call_history if c.get("method") == "generate_with_tools_stream")


class TestRetryLoopDrivenByOnEnd:
    """execute_stream must drive retries off ValidationHook.final_report."""

    @pytest.mark.asyncio
    async def test_retry_driven_by_on_end_final_report(self, real_agent_config, mock_llm_create):
        """Attempt 1 blocks at on_end, attempt 2 clears → NodeResult.success=True."""
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Attempt 1 done."),
                build_simple_response("Attempt 2 done."),
            ]
        )

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node._validation_hook = _FakeValidationHook([_make_blocking_report(), None])
        node.input = SemanticNodeInput(user_message="Create a table")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Stream was invoked exactly twice — once per attempt.
        assert _count_stream_calls(mock_llm_create) == 2
        # Terminal action reports success.
        last = actions[-1]
        output = last.output or {}
        assert output.get("success") is True

    @pytest.mark.asyncio
    async def test_retries_exhausted_on_end_blocking(self, real_agent_config, mock_llm_create):
        """3 blocking attempts exhaust the retry budget → success=False +
        validation_report surfaces on the NodeResult."""
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Attempt 1."),
                build_simple_response("Attempt 2."),
                build_simple_response("Attempt 3."),
            ]
        )

        node = GenTableAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node._validation_hook = _FakeValidationHook([_make_blocking_report()] * 3)
        node.input = SemanticNodeInput(user_message="Create a table")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Stream invoked max_retries (default 3) times.
        assert _count_stream_calls(mock_llm_create) == 3
        last = actions[-1]
        output = last.output or {}
        assert output.get("success") is False
        assert output["validation_report"]["checks"][0]["error"] == "synthetic blocking failure for retry-loop test"


class TestDeliverableDatasourceContext:
    """The frozen system prompt is datasource-free; the live selection rides per turn.

    Mirrors the chat-node design: ``_prepare_template_context`` (rendered into
    the per-session snapshot) carries only session-stable values, while the
    current datasource/dialect arrives in each user message via
    ``_build_datasource_reminder``.
    """

    def _make_node(self, *, db_func_tool=None, current_datasource=None, services=None):
        from types import SimpleNamespace

        from datus.agent.node.deliverable_node import DeliverableAgenticNode

        node = DeliverableAgenticNode.__new__(DeliverableAgenticNode)
        node.agent_config = SimpleNamespace(current_datasource=current_datasource, services=services)
        node.tools = []
        node.mcp_servers = {}
        node.ask_user_tool = None
        node.db_func_tool = db_func_tool
        return node

    def test_template_context_has_no_datasource_keys(self):
        from types import SimpleNamespace

        node = self._make_node(current_datasource="main")
        ctx = node._prepare_template_context(SimpleNamespace(user_message="create table t"))

        assert set(ctx) == {"native_tools", "mcp_tools", "has_ask_user_tool"}

    def test_enhanced_message_uses_datasource_reminder_with_db_tool(self):
        from types import SimpleNamespace

        services = SimpleNamespace(datasources={"main": SimpleNamespace(type="duckdb")})
        db_tool = SimpleNamespace(connector=SimpleNamespace(database_name="proddb"))
        node = self._make_node(db_func_tool=db_tool, current_datasource="main", services=services)

        user_input = SimpleNamespace(user_message="create table t", catalog="", database="", db_schema="")
        message = node._build_enhanced_message(user_input)

        # The merged reminder supersedes the legacy Context line entirely.
        assert "Current datasource: main (dialect: duckdb, database: proddb)" in message
        assert "Context:" not in message
        assert "create table t" in message

    def test_enhanced_message_falls_back_to_legacy_context_without_db_tool(self):
        from types import SimpleNamespace

        node = self._make_node(db_func_tool=None, current_datasource="main")

        user_input = SimpleNamespace(user_message="make dashboard", catalog="c1", database="db1", db_schema="s1")
        message = node._build_enhanced_message(user_input)

        assert "Context: catalog: c1, database: db1, schema: s1" in message
        assert "Current datasource:" not in message
