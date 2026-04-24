# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for :class:`TableDeliverableAgenticNode.execute_stream` retry loop.

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
        assert output.get("validation_report") is not None
