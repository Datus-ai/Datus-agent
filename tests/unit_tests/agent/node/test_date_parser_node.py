# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for DateParserNode - zero external dependencies."""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node import Node
from datus.configuration.node_type import NodeType
from datus.schemas.date_parser_node_models import DateParserInput, DateParserResult, ExtractedDate
from datus.schemas.node_models import SqlTask


def make_agent_config(date_parser_language=None):
    cfg = MagicMock()
    cfg.datasource_configs = {}
    cfg.current_datasource = "test"
    if date_parser_language:
        node_cfg = MagicMock()
        node_cfg.input.language = date_parser_language
        cfg.nodes = {"date_parser": node_cfg}
    else:
        cfg.nodes = {}
    return cfg


def make_sql_task(task="Show sales in January 2024"):
    return SqlTask(task=task, database_type="sqlite", database_name="db")


def make_date_parser_input():
    return DateParserInput(sql_task=make_sql_task())


def make_node(input_data=None, agent_config=None):
    cfg = agent_config or make_agent_config()
    return Node.new_instance(
        "date_parser_1",
        "Date Parser",
        NodeType.TYPE_DATE_PARSER,
        input_data=input_data,
        agent_config=cfg,
    )


class TestDateParserNodeGetLanguage:
    """Test _get_language_setting()."""

    def test_default_language_is_en(self):
        node = make_node()
        assert node._get_language_setting() == "en"

    def test_language_from_config(self):
        cfg = make_agent_config(date_parser_language="cn")
        node = make_node(agent_config=cfg)
        assert node._get_language_setting() == "cn"

    def test_no_agent_config(self):
        node = make_node()
        node.agent_config = None
        assert node._get_language_setting() == "en"


class TestDateParserNodeExecute:
    """Test execute() / _execute_date_parsing()."""

    def test_execute_success(self):
        node = make_node(make_date_parser_input())
        node.model = MagicMock()

        extracted = [
            ExtractedDate(
                original_text="January 2024",
                parsed_date="2024-01-01",
                start_date="2024-01-01",
                end_date="2024-01-31",
                date_type="range",
                confidence=0.95,
            )
        ]

        # get_default_current_date is imported inline inside _execute_date_parsing
        # so patch it at its source module
        with patch("datus.utils.time_utils.get_default_current_date", return_value="2024-01-15"):
            with patch("datus.agent.node.date_parser_node.DateParserTool") as mock_tool_cls:
                mock_tool = MagicMock()
                mock_tool.execute.return_value = extracted
                mock_tool.generate_date_context.return_value = "January 2024: 2024-01-01 to 2024-01-31"
                mock_tool_cls.return_value = mock_tool
                node.execute()

        assert node.result.success is True
        assert len(node.result.extracted_dates) == 1

    def test_execute_with_no_dates_extracted(self):
        node = make_node(make_date_parser_input())
        node.model = MagicMock()

        with patch("datus.utils.time_utils.get_default_current_date", return_value="2024-01-15"):
            with patch("datus.agent.node.date_parser_node.DateParserTool") as mock_tool_cls:
                mock_tool = MagicMock()
                mock_tool.execute.return_value = []
                mock_tool.generate_date_context.return_value = ""
                mock_tool_cls.return_value = mock_tool
                node.execute()

        assert node.result.success is True
        assert node.result.extracted_dates == []

    def test_execute_with_exception(self):
        node = make_node(make_date_parser_input())
        node.model = MagicMock()

        with patch("datus.agent.node.date_parser_node.DateParserTool", side_effect=RuntimeError("parse error")):
            node.execute()

        assert node.result.success is False
        assert "parse error" in node.result.error

    def test_execute_updates_external_knowledge(self):
        """When date_context is non-empty, external_knowledge should be updated."""
        task = SqlTask(
            task="Q1 sales",
            database_type="sqlite",
            database_name="db",
            external_knowledge="prior knowledge",
        )
        node = make_node(DateParserInput(sql_task=task))
        node.model = MagicMock()

        with patch("datus.utils.time_utils.get_default_current_date", return_value="2024-02-01"):
            with patch("datus.agent.node.date_parser_node.DateParserTool") as mock_tool_cls:
                mock_tool = MagicMock()
                mock_tool.execute.return_value = []
                mock_tool.generate_date_context.return_value = "Q1: 2024-01-01 to 2024-03-31"
                mock_tool_cls.return_value = mock_tool
                node.execute()

        assert node.result.success is True
        assert "Q1" in node.result.enriched_task.date_ranges


class TestDateParserNodeSetupInput:
    """Test setup_input."""

    def test_setup_input_builds_date_parser_input(self):
        node = make_node()
        workflow = MagicMock()
        workflow.task = make_sql_task()

        result = node.setup_input(workflow)
        assert result["success"] is True
        assert isinstance(node.input, DateParserInput)
        assert node.input.sql_task.task == "Show sales in January 2024"


class TestDateParserNodeUpdateContext:
    """Test update_context."""

    def test_update_context_success(self):
        node = make_node(make_date_parser_input())
        extracted = [
            ExtractedDate(
                original_text="January",
                parsed_date="2024-01-01",
                start_date="2024-01-01",
                end_date="2024-01-31",
                date_type="specific",
            )
        ]
        enriched = make_sql_task("enriched task")
        node.result = DateParserResult(
            success=True,
            extracted_dates=extracted,
            enriched_task=enriched,
            date_context="date_ctx_value",
        )

        class SimpleWorkflow:
            task = make_sql_task()
            date_context = None

        workflow = SimpleWorkflow()
        result = node.update_context(workflow)
        assert result["success"] is True
        assert workflow.task.task == "enriched task"
        assert workflow.date_context == "date_ctx_value"

    def test_update_context_appends_to_existing(self):
        node = make_node(make_date_parser_input())
        node.result = DateParserResult(
            success=True,
            extracted_dates=[],
            enriched_task=make_sql_task("enriched"),
            date_context="new context",
        )
        workflow = MagicMock()
        workflow.date_context = "existing context"

        node.update_context(workflow)
        assert "existing context" in workflow.date_context
        assert "new context" in workflow.date_context

    def test_update_context_failure_result(self):
        node = make_node(make_date_parser_input())
        node.result = DateParserResult(
            success=False,
            error="parse failed",
            extracted_dates=[],
            enriched_task=make_sql_task(),
            date_context="",
        )
        workflow = MagicMock()

        result = node.update_context(workflow)
        # Should still succeed (graceful degradation)
        assert result["success"] is True

    def test_update_context_exception(self):
        node = make_node(make_date_parser_input())
        node.result = DateParserResult(
            success=True,
            extracted_dates=[
                ExtractedDate(
                    original_text="Jan",
                    parsed_date="2024-01-01",
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                    date_type="specific",
                )
            ],
            enriched_task=make_sql_task(),
            date_context="ctx",
        )

        # Pass an object whose .task assignment raises
        class BrokenWorkflow:
            @property
            def task(self):
                return make_sql_task()

            @task.setter
            def task(self, value):
                raise RuntimeError("cannot set task")

        result = node.update_context(BrokenWorkflow())
        assert result["success"] is False


class TestDateParserNodeStream:
    """Test execute_stream (stub generator that yields None then returns)."""

    @pytest.mark.asyncio
    async def test_execute_stream_yields_none_value(self):
        node = make_node(make_date_parser_input())
        actions = []
        async for action in node.execute_stream():
            actions.append(action)
        # DateParserNode.execute_stream does `yield; return` so yields exactly one None
        assert actions == [None]
