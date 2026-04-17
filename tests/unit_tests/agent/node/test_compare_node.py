# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for CompareNode (non-agentic synchronous wrapper).

CI-level: zero external deps, zero network, zero API keys.
CompareAgenticNode and LLM calls are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node.compare_node import CompareNode
from datus.schemas.action_history import ActionRole, ActionStatus
from datus.schemas.compare_node_models import CompareInput, CompareResult
from datus.schemas.node_models import SQLContext, SqlTask
from datus.utils.exceptions import DatusException

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_config():
    cfg = MagicMock()
    cfg.agentic_nodes = {}
    cfg.permissions_config = None
    cfg.skills_config = None
    cfg.prompt_version = None
    cfg.project_root = "."
    return cfg


def _make_sql_task():
    return SqlTask(
        database_type="sqlite",
        database_name="test_db",
        task="Find total sales",
    )


def _make_sql_context():
    return SQLContext(
        sql_query="SELECT SUM(sales) FROM orders",
        explanation="Sum of all sales",
        sql_return="100000",
        sql_error="",
    )


def _make_compare_input():
    return CompareInput(
        sql_task=_make_sql_task(),
        sql_context=_make_sql_context(),
        expectation="Expected total sales to be 100000",
    )


def _make_model():
    m = MagicMock()
    m.generate_with_json_output.return_value = {
        "explanation": "The SQL matches expectations.",
        "suggest": "No changes needed.",
    }
    return m


def _make_node(agent_config=None):
    cfg = agent_config or _make_agent_config()
    node = CompareNode(
        node_id="compare_1",
        description="Compare node",
        node_type="compare",
        agent_config=cfg,
    )
    node.model = _make_model()
    node.input = _make_compare_input()
    return node


def _make_workflow():
    wf = MagicMock()
    wf.task = _make_sql_task()
    wf.get_last_sqlcontext.return_value = _make_sql_context()
    wf.context.sql_contexts = []
    return wf


# ---------------------------------------------------------------------------
# TestCompareNodeInit
# ---------------------------------------------------------------------------


class TestCompareNodeInit:
    def test_node_creates(self):
        node = _make_node()
        assert node.id == "compare_1"
        assert node.description == "Compare node"


# ---------------------------------------------------------------------------
# TestSetupInput
# ---------------------------------------------------------------------------


class TestSetupInputCompareNode:
    def test_setup_input_creates_compare_input(self):
        node = _make_node()
        node.input = None  # no pre-set expectation
        wf = _make_workflow()
        result = node.setup_input(wf)

        assert result["success"] is True
        assert isinstance(node.input, CompareInput)
        assert node.input.sql_task.database_name == "test_db"

    def test_setup_input_uses_string_expectation(self):
        """If node.input is a non-empty string, it becomes the expectation."""
        node = _make_node()
        node.input = "Expected: total sales = 100000"
        wf = _make_workflow()
        node.setup_input(wf)

        assert node.input.expectation == "Expected: total sales = 100000"

    def test_setup_input_empty_string_gives_empty_expectation(self):
        node = _make_node()
        node.input = "  "  # whitespace only
        wf = _make_workflow()
        node.setup_input(wf)

        assert node.input.expectation == ""


# ---------------------------------------------------------------------------
# TestUpdateContext
# ---------------------------------------------------------------------------


class TestUpdateContextCompareNode:
    def test_update_context_appends_sql_context(self):
        node = _make_node()
        node.result = CompareResult(
            success=True,
            explanation="SQL matches expectations",
            suggest="No changes needed",
        )

        wf = _make_workflow()
        result = node.update_context(wf)

        assert result["success"] is True
        assert len(wf.context.sql_contexts) == 1
        appended = wf.context.sql_contexts[0]
        assert "SQL matches expectations" in appended.explanation

    def test_update_context_handles_exception(self):
        node = _make_node()
        node.result = CompareResult(
            success=True,
            explanation="explanation",
            suggest="suggest",
        )

        wf = _make_workflow()
        # Make node.input raise when sql_query is accessed
        bad_input = MagicMock()
        bad_input.sql_context.sql_query = "SELECT 1"
        node.input = bad_input
        wf.context.sql_contexts = MagicMock()
        wf.context.sql_contexts.append = MagicMock(side_effect=RuntimeError("append error"))
        result = node.update_context(wf)

        assert result["success"] is False
        assert "append error" in result["message"]


# ---------------------------------------------------------------------------
# TestExecuteCompare
# ---------------------------------------------------------------------------


class TestExecuteCompare:
    def test_execute_compare_returns_result(self):
        node = _make_node()
        with patch("datus.agent.node.compare_node.CompareAgenticNode") as mock_cls:
            mock_cls._prepare_prompt_components.return_value = (
                "sys",
                "user",
                [{"role": "system", "content": "sys"}],
            )
            mock_cls._parse_comparison_output.return_value = {
                "explanation": "Looks correct",
                "suggest": "None",
            }
            result = node._execute_compare()

        assert result.success is True
        assert result.explanation == "Looks correct"
        assert result.suggest == "None"

    def test_execute_compare_raises_when_not_compare_input(self):
        node = _make_node()
        node.input = "not a CompareInput"

        with pytest.raises(DatusException):
            node._execute_compare()

    def test_execute_compare_raises_when_no_model(self):
        node = _make_node()
        node.model = None

        with pytest.raises(DatusException):
            node._execute_compare()

    def test_execute_compare_returns_failure_on_exception(self):
        node = _make_node()
        with patch("datus.agent.node.compare_node.CompareAgenticNode") as mock_cls:
            mock_cls._prepare_prompt_components.side_effect = RuntimeError("prompt error")
            result = node._execute_compare()

        assert result.success is False
        assert "prompt error" in result.error

    def test_execute_sets_result(self):
        node = _make_node()
        with patch("datus.agent.node.compare_node.CompareAgenticNode") as mock_cls:
            mock_cls._prepare_prompt_components.return_value = (
                "sys",
                "user",
                [{"role": "system", "content": "sys"}],
            )
            mock_cls._parse_comparison_output.return_value = {
                "explanation": "OK",
                "suggest": "None",
            }
            node.execute()

        assert node.result is not None
        assert node.result.success is True


# ---------------------------------------------------------------------------
# TestExecuteStream
# ---------------------------------------------------------------------------


class TestExecuteStreamCompareNode:
    @pytest.mark.asyncio
    async def test_execute_stream_yields_actions(self):
        """execute_stream yields setup + delegated stream actions."""
        node = _make_node()

        mock_compare_result = CompareResult(
            success=True,
            explanation="Matches",
            suggest="None",
        )
        success_action = MagicMock()
        success_action.status = ActionStatus.SUCCESS
        success_action.output = mock_compare_result.model_dump()
        success_action.role = ActionRole.ASSISTANT
        success_action.action_type = "compare_sql_response"

        async def _mock_stream(*args, **kwargs):
            yield success_action

        with patch("datus.agent.node.compare_node.CompareAgenticNode") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.execute_stream = _mock_stream
            mock_instance.input = None

            actions = []
            async for action in node.execute_stream():
                actions.append(action)

        # At minimum: setup_action + delegated action
        assert len(actions) >= 2

    @pytest.mark.asyncio
    async def test_execute_stream_no_model_returns_nothing(self):
        """When model is None, execute_stream yields nothing."""
        node = _make_node()
        node.model = None
        actions = []
        async for action in node.execute_stream():
            actions.append(action)

        assert actions == []

    @pytest.mark.asyncio
    async def test_execute_stream_propagates_exception(self):
        """execute_stream re-raises unexpected errors."""
        node = _make_node()

        async def _raise(*args, **kwargs):
            raise RuntimeError("stream error")
            yield  # noqa

        with patch("datus.agent.node.compare_node.CompareAgenticNode") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.execute_stream = _raise
            mock_instance.input = None

            with pytest.raises(RuntimeError, match="stream error"):
                async for _ in node.execute_stream():
                    pass
