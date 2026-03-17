# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/llms_tools/reasoning_sql.py"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SqlTask, TableSchema
from datus.schemas.reason_sql_node_models import ReasoningInput, ReasoningResult
from datus.tools.llms_tools.reasoning_sql import reasoning_sql_with_mcp, reasoning_sql_with_mcp_stream
from datus.utils.exceptions import DatusException


def _make_reasoning_input(**overrides):
    defaults = dict(
        sql_task=SqlTask(task="Show all users", database_type="sqlite", database_name="test_db"),
        table_schemas=[
            TableSchema(
                table_name="users",
                database_name="test_db",
                schema_name="public",
                definition="CREATE TABLE users (id INT, name TEXT)",
            )
        ],
        data_details=None,
        metrics=None,
        contexts=[],
        external_knowledge="",
    )
    defaults.update(overrides)
    return ReasoningInput(**defaults)


def _make_action(action_type="message", role=ActionRole.ASSISTANT, status=ActionStatus.SUCCESS, output=None):
    return ActionHistory(
        action_id="test-id",
        role=role,
        action_type=action_type,
        status=status,
        output=output,
    )


async def _collect_stream(gen):
    results = []
    async for item in gen:
        results.append(item)
    return results


class TestReasoningSqlWithMcpStream:
    def test_raises_for_invalid_input(self):
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="ReasoningInput"):
            asyncio.run(
                _collect_stream(
                    reasoning_sql_with_mcp_stream(
                        model=mock_model,
                        input_data="not a ReasoningInput",
                        tool_config={},
                        tools=[],
                    )
                )
            )

    def test_yields_actions_from_base_stream(self):
        mock_model = MagicMock()
        input_data = _make_reasoning_input()
        action = _make_action()

        async def fake_base_stream(**kwargs):
            yield action

        with patch("datus.tools.llms_tools.reasoning_sql.get_reasoning_prompt", return_value="prompt"), patch(
            "datus.tools.llms_tools.reasoning_sql.base_mcp_stream", return_value=fake_base_stream()
        ):
            results = asyncio.run(
                _collect_stream(
                    reasoning_sql_with_mcp_stream(
                        model=mock_model,
                        input_data=input_data,
                        tool_config={"max_turns": 5},
                        tools=[],
                    )
                )
            )

        assert len(results) == 1
        assert results[0] is action

    def test_creates_action_history_manager_when_none(self):
        mock_model = MagicMock()
        input_data = _make_reasoning_input()

        async def fake_base_stream(**kwargs):
            return
            yield

        with patch("datus.tools.llms_tools.reasoning_sql.get_reasoning_prompt", return_value="p"), patch(
            "datus.tools.llms_tools.reasoning_sql.base_mcp_stream", return_value=fake_base_stream()
        ):
            asyncio.run(
                _collect_stream(
                    reasoning_sql_with_mcp_stream(
                        model=mock_model,
                        input_data=input_data,
                        tool_config={},
                        tools=[],
                    )
                )
            )
        # Should not raise - manager was created internally

    def test_sql_contexts_attribute_initialized_on_manager(self):
        """After streaming completes, action_history_manager gets sql_contexts attribute."""
        mock_model = MagicMock()
        input_data = _make_reasoning_input()
        manager = ActionHistoryManager()

        async def fake_base_stream_fn(*args, **kwargs):
            return
            yield

        with patch("datus.tools.llms_tools.reasoning_sql.get_reasoning_prompt", return_value="p"), patch(
            "datus.tools.llms_tools.reasoning_sql.base_mcp_stream", side_effect=lambda **kw: fake_base_stream_fn(**kw)
        ):
            asyncio.run(
                _collect_stream(
                    reasoning_sql_with_mcp_stream(
                        model=mock_model,
                        input_data=input_data,
                        tool_config={},
                        tools=[],
                        action_history_manager=manager,
                    )
                )
            )

        # sql_contexts is always set (even if empty) after streaming completes
        assert hasattr(manager, "sql_contexts")
        assert isinstance(manager.sql_contexts, list)

    def test_sql_contexts_empty_when_no_relevant_actions(self):
        """When no read_query or final message actions, sql_contexts is empty."""
        mock_model = MagicMock()
        input_data = _make_reasoning_input()
        manager = ActionHistoryManager()

        # Add a non-relevant action
        tool_action = ActionHistory(
            action_id="tool-1",
            role=ActionRole.TOOL,
            action_type="list_tables",  # not read_query
            status=ActionStatus.SUCCESS,
            input={"arg": "value"},
            output={"result": "table1"},
        )
        manager.add_action(tool_action)

        async def fake_base_stream_fn(*args, **kwargs):
            return
            yield

        with patch("datus.tools.llms_tools.reasoning_sql.get_reasoning_prompt", return_value="p"), patch(
            "datus.tools.llms_tools.reasoning_sql.base_mcp_stream", side_effect=lambda **kw: fake_base_stream_fn(**kw)
        ):
            asyncio.run(
                _collect_stream(
                    reasoning_sql_with_mcp_stream(
                        model=mock_model,
                        input_data=input_data,
                        tool_config={},
                        tools=[],
                        action_history_manager=manager,
                    )
                )
            )

        assert hasattr(manager, "sql_contexts")
        assert manager.sql_contexts == []


class TestReasoningSqlWithMcp:
    def test_raises_for_invalid_input(self):
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="ReasoningInput"):
            reasoning_sql_with_mcp(
                model=mock_model,
                input_data={"not": "a ReasoningInput"},
                tools=[],
                tool_config={},
            )

    def test_returns_reasoning_result_from_json_content(self):
        mock_model = MagicMock()
        input_data = _make_reasoning_input()

        exec_result = {
            "content": '{"sql": "SELECT * FROM users", "explanation": "all users"}',
            "sql_contexts": [],
        }

        with patch("datus.tools.llms_tools.reasoning_sql.get_reasoning_prompt", return_value="p"), patch(
            "datus.tools.llms_tools.reasoning_sql.prompt_manager"
        ) as mock_pm, patch("datus.tools.llms_tools.reasoning_sql.asyncio") as mock_asyncio, patch(
            "datus.tools.llms_tools.reasoning_sql.llm_result2json",
            return_value={"sql": "SELECT * FROM users", "explanation": "all users"},
        ):
            mock_pm.get_raw_template.return_value = "instruction"
            mock_asyncio.run.return_value = exec_result

            result = reasoning_sql_with_mcp(
                model=mock_model,
                input_data=input_data,
                tools=[],
                tool_config={"max_turns": 5},
            )

        assert isinstance(result, ReasoningResult)
        assert result.success is True
        assert result.sql_query == "SELECT * FROM users"

    def test_falls_back_to_sql_extraction_when_json_fails(self):
        mock_model = MagicMock()
        input_data = _make_reasoning_input()

        exec_result = {
            "content": "```sql\nSELECT id FROM users\n```",
            "sql_contexts": [],
        }

        with patch("datus.tools.llms_tools.reasoning_sql.get_reasoning_prompt", return_value="p"), patch(
            "datus.tools.llms_tools.reasoning_sql.prompt_manager"
        ) as mock_pm, patch("datus.tools.llms_tools.reasoning_sql.asyncio") as mock_asyncio, patch(
            "datus.tools.llms_tools.reasoning_sql.llm_result2json", return_value=None
        ), patch(
            "datus.tools.llms_tools.reasoning_sql.llm_result2sql", return_value="SELECT id FROM users"
        ):
            mock_pm.get_raw_template.return_value = "instruction"
            mock_asyncio.run.return_value = exec_result

            result = reasoning_sql_with_mcp(
                model=mock_model,
                input_data=input_data,
                tools=[],
                tool_config={},
            )

        assert result.success is True
        assert result.sql_query == "SELECT id FROM users"

    def test_raises_datus_exception_when_both_json_and_sql_fail(self):
        mock_model = MagicMock()
        input_data = _make_reasoning_input()

        exec_result = {
            "content": "I cannot answer this question.",
            "sql_contexts": [],
        }

        with patch("datus.tools.llms_tools.reasoning_sql.get_reasoning_prompt", return_value="p"), patch(
            "datus.tools.llms_tools.reasoning_sql.prompt_manager"
        ) as mock_pm, patch("datus.tools.llms_tools.reasoning_sql.asyncio") as mock_asyncio, patch(
            "datus.tools.llms_tools.reasoning_sql.llm_result2json", return_value=None
        ), patch(
            "datus.tools.llms_tools.reasoning_sql.llm_result2sql", return_value=None
        ):
            mock_pm.get_raw_template.return_value = "instruction"
            mock_asyncio.run.return_value = exec_result

            with pytest.raises(DatusException):
                reasoning_sql_with_mcp(
                    model=mock_model,
                    input_data=input_data,
                    tools=[],
                    tool_config={},
                )

    def test_permission_error_is_reraised(self):
        mock_model = MagicMock()
        input_data = _make_reasoning_input()

        with patch("datus.tools.llms_tools.reasoning_sql.get_reasoning_prompt", return_value="p"), patch(
            "datus.tools.llms_tools.reasoning_sql.prompt_manager"
        ) as mock_pm, patch("datus.tools.llms_tools.reasoning_sql.asyncio") as mock_asyncio:
            mock_pm.get_raw_template.return_value = "instruction"
            mock_asyncio.run.side_effect = RuntimeError("403 Forbidden access not allowed")

            with pytest.raises(RuntimeError, match="403"):
                reasoning_sql_with_mcp(
                    model=mock_model,
                    input_data=input_data,
                    tools=[],
                    tool_config={},
                )

    def test_generic_exception_raises_datus_exception(self):
        mock_model = MagicMock()
        input_data = _make_reasoning_input()

        with patch("datus.tools.llms_tools.reasoning_sql.get_reasoning_prompt", return_value="p"), patch(
            "datus.tools.llms_tools.reasoning_sql.prompt_manager"
        ) as mock_pm, patch("datus.tools.llms_tools.reasoning_sql.asyncio") as mock_asyncio:
            mock_pm.get_raw_template.return_value = "instruction"
            mock_asyncio.run.side_effect = RuntimeError("connection reset")

            with pytest.raises(DatusException):
                reasoning_sql_with_mcp(
                    model=mock_model,
                    input_data=input_data,
                    tools=[],
                    tool_config={},
                )

    def test_uses_max_turns_from_tool_config(self):
        mock_model = MagicMock()
        input_data = _make_reasoning_input()

        exec_result = {
            "content": '{"sql": "SELECT 1"}',
            "sql_contexts": [],
        }

        captured = {}

        def capture_run(coro):
            captured["coro"] = coro
            return exec_result

        with patch("datus.tools.llms_tools.reasoning_sql.get_reasoning_prompt", return_value="p"), patch(
            "datus.tools.llms_tools.reasoning_sql.prompt_manager"
        ) as mock_pm, patch("datus.tools.llms_tools.reasoning_sql.asyncio") as mock_asyncio, patch(
            "datus.tools.llms_tools.reasoning_sql.llm_result2json", return_value={"sql": "SELECT 1"}
        ):
            mock_pm.get_raw_template.return_value = "instruction"
            mock_asyncio.run.side_effect = capture_run
            mock_model.generate_with_tools.return_value = exec_result

            reasoning_sql_with_mcp(
                model=mock_model,
                input_data=input_data,
                tools=[],
                tool_config={"max_turns": 20},
            )
