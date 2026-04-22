# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
from unittest.mock import MagicMock, patch

import pytest

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.reference_template_tools import ReferenceTemplateTools


@pytest.fixture
def mock_agent_config():
    config = MagicMock()
    config.current_datasource = "test_datasource"
    return config


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.get_reference_template_size.return_value = 3
    return rag


@pytest.fixture
def tools(mock_agent_config, mock_rag):
    with patch(
        "datus.tools.func_tool.reference_template_tools.ReferenceTemplateRAG",
        return_value=mock_rag,
    ):
        return ReferenceTemplateTools(mock_agent_config)


class TestReferenceTemplateToolsInit:
    def test_has_reference_templates_true(self, tools):
        assert tools.has_reference_templates is True

    def test_has_reference_templates_false(self, mock_agent_config):
        mock_rag = MagicMock()
        mock_rag.get_reference_template_size.return_value = 0
        with patch(
            "datus.tools.func_tool.reference_template_tools.ReferenceTemplateRAG",
            return_value=mock_rag,
        ):
            t = ReferenceTemplateTools(mock_agent_config)
        assert t.has_reference_templates is False

    def test_available_tools_when_has_templates(self, tools):
        available = tools.available_tools()
        assert len(available) == 3

    def test_available_tools_when_no_templates(self, mock_agent_config):
        mock_rag = MagicMock()
        mock_rag.get_reference_template_size.return_value = 0
        with patch(
            "datus.tools.func_tool.reference_template_tools.ReferenceTemplateRAG",
            return_value=mock_rag,
        ):
            t = ReferenceTemplateTools(mock_agent_config)
        assert t.available_tools() == []


class TestSearchReferenceTemplate:
    def test_search_success(self, tools, mock_rag):
        mock_rag.search_reference_templates.return_value = [
            {"name": "test_template", "template": "SELECT {{col}}", "parameters": '[{"name": "col"}]'}
        ]
        result = tools.search_reference_template("find template")
        assert result.success == 1
        assert len(result.result) == 1

    def test_search_with_subject_path(self, tools, mock_rag):
        mock_rag.search_reference_templates.return_value = []
        tools.search_reference_template("query", subject_path=["Sales", "Revenue"])
        mock_rag.search_reference_templates.assert_called_once_with(
            query_text="query",
            subject_path=["Sales", "Revenue"],
            top_n=5,
            selected_fields=["name", "template", "parameters", "summary", "tags"],
        )

    def test_search_handles_exception(self, tools, mock_rag):
        mock_rag.search_reference_templates.side_effect = Exception("DB error")
        result = tools.search_reference_template("query")
        assert result.success == 0
        assert "DB error" in result.error


class TestGetReferenceTemplate:
    def test_get_success(self, tools, mock_rag):
        mock_rag.get_reference_template_detail.return_value = [
            {"name": "tpl", "template": "SELECT {{x}}", "parameters": '[{"name": "x"}]'}
        ]
        result = tools.get_reference_template(["Sales"], "tpl")
        assert result.success == 1
        assert result.result["name"] == "tpl"

    def test_get_not_found(self, tools, mock_rag):
        mock_rag.get_reference_template_detail.return_value = []
        result = tools.get_reference_template(["Sales"], "nonexistent")
        assert result.success == 0
        assert "No matched result" in result.error

    def test_get_handles_exception(self, tools, mock_rag):
        mock_rag.get_reference_template_detail.side_effect = Exception("error")
        result = tools.get_reference_template(["Sales"], "tpl")
        assert result.success == 0


class TestRenderReferenceTemplate:
    def test_render_success(self, tools, mock_rag):
        mock_rag.get_reference_template_detail.return_value = [
            {
                "name": "daily_sales",
                "template": "SELECT * FROM orders WHERE dt > '{{start_date}}' AND region = '{{region}}'",
                "parameters": json.dumps([{"name": "start_date"}, {"name": "region"}]),
            }
        ]
        result = tools.render_reference_template(
            ["Sales"], "daily_sales", json.dumps({"start_date": "2024-01-01", "region": "US"})
        )
        assert result.success == 1
        assert "2024-01-01" in result.result["rendered_sql"]
        assert "US" in result.result["rendered_sql"]
        assert result.result["template_name"] == "daily_sales"

    def test_render_missing_parameter(self, tools, mock_rag):
        mock_rag.get_reference_template_detail.return_value = [
            {
                "name": "tpl",
                "template": "SELECT * FROM t WHERE dt > '{{start_date}}' AND region = '{{region}}'",
                "parameters": json.dumps([{"name": "start_date"}, {"name": "region"}]),
            }
        ]
        result = tools.render_reference_template(["Sales"], "tpl", json.dumps({"start_date": "2024-01-01"}))
        assert result.success == 0
        assert "region" in result.error
        assert "Missing parameters" in result.error or "missing" in result.error.lower()
        # Error should include expected and provided params to help model retry
        assert "start_date" in result.error
        assert "Expected parameters" in result.error or "requires parameters" in result.error

    def test_render_template_not_found(self, tools, mock_rag):
        mock_rag.get_reference_template_detail.return_value = []
        result = tools.render_reference_template(["Sales"], "nonexistent", json.dumps({"x": "1"}))
        assert result.success == 0
        assert "not found" in result.error.lower()

    def test_render_empty_template(self, tools, mock_rag):
        mock_rag.get_reference_template_detail.return_value = [{"name": "empty", "template": "", "parameters": "[]"}]
        result = tools.render_reference_template(["Sales"], "empty", "{}")
        assert result.success == 0
        assert "empty" in result.error.lower()

    def test_render_no_params_needed(self, tools, mock_rag):
        mock_rag.get_reference_template_detail.return_value = [
            {
                "name": "simple",
                "template": "SELECT count(*) FROM users",
                "parameters": "[]",
            }
        ]
        result = tools.render_reference_template(["Sales"], "simple", "{}")
        assert result.success == 1
        assert result.result["rendered_sql"] == "SELECT count(*) FROM users"

    def test_render_invalid_json_params(self, tools, mock_rag):
        result = tools.render_reference_template(["Sales"], "tpl", "not valid json")
        assert result.success == 0
        assert "Invalid params format" in result.error

    def test_render_exception_in_storage(self, tools, mock_rag):
        mock_rag.get_reference_template_detail.side_effect = Exception("storage error")
        result = tools.render_reference_template(["Sales"], "tpl", '{"x": "1"}')
        assert result.success == 0
        assert "storage error" in result.error


class TestExecuteReferenceTemplate:
    def test_execute_success(self, tools, mock_rag):
        """Render + execute succeeds, returns both SQL and query result."""
        mock_rag.get_reference_template_detail.return_value = [
            {
                "name": "daily_sales",
                "template": "SELECT * FROM orders WHERE region = '{{region}}'",
                "parameters": json.dumps([{"name": "region"}]),
            }
        ]
        mock_db = MagicMock()
        mock_db.read_query.return_value = FuncToolResult(success=1, result=[{"id": 1, "region": "US"}])
        tools.db_func_tool = mock_db

        result = tools.execute_reference_template(["Sales"], "daily_sales", json.dumps({"region": "US"}))
        assert result.success == 1
        assert "US" in result.result["rendered_sql"]
        assert result.result["query_result"] == [{"id": 1, "region": "US"}]
        assert result.result["template_name"] == "daily_sales"
        mock_db.read_query.assert_called_once()

    def test_execute_render_fails(self, tools, mock_rag):
        """When render fails, execute returns the render error without calling DB."""
        mock_rag.get_reference_template_detail.return_value = []
        mock_db = MagicMock()
        tools.db_func_tool = mock_db

        result = tools.execute_reference_template(["Sales"], "nonexistent", json.dumps({"x": "1"}))
        assert result.success == 0
        assert "not found" in result.error.lower()
        mock_db.read_query.assert_not_called()

    def test_execute_query_fails(self, tools, mock_rag):
        """Render succeeds but query execution fails, returns error with rendered SQL."""
        mock_rag.get_reference_template_detail.return_value = [
            {
                "name": "tpl",
                "template": "SELECT {{col}} FROM t",
                "parameters": json.dumps([{"name": "col"}]),
            }
        ]
        mock_db = MagicMock()
        mock_db.read_query.return_value = FuncToolResult(success=0, error="syntax error near 'col'")
        tools.db_func_tool = mock_db

        result = tools.execute_reference_template(["Sales"], "tpl", json.dumps({"col": "id"}))
        assert result.success == 0
        assert "query execution failed" in result.error
        assert result.result["rendered_sql"] == "SELECT id FROM t"

    def test_execute_no_db_tool(self, tools, mock_rag):
        """Without db_func_tool, returns error suggesting manual render + read_query."""
        mock_rag.get_reference_template_detail.return_value = [
            {
                "name": "tpl",
                "template": "SELECT 1",
                "parameters": "[]",
            }
        ]
        tools.db_func_tool = None

        result = tools.execute_reference_template(["Sales"], "tpl", "{}")
        assert result.success == 0
        assert "not available" in result.error.lower()
        assert "read_query" in result.error

    def test_execute_db_exception(self, tools, mock_rag):
        """DB tool raises exception, returns error with rendered SQL preserved."""
        mock_rag.get_reference_template_detail.return_value = [
            {
                "name": "tpl",
                "template": "SELECT {{x}}",
                "parameters": json.dumps([{"name": "x"}]),
            }
        ]
        mock_db = MagicMock()
        mock_db.read_query.side_effect = RuntimeError("connection lost")
        tools.db_func_tool = mock_db

        result = tools.execute_reference_template(["Sales"], "tpl", json.dumps({"x": "1"}))
        assert result.success == 0
        assert "connection lost" in result.error
        assert result.result["rendered_sql"] == "SELECT 1"

    def test_execute_with_database_param(self, tools, mock_rag):
        """Database parameter is forwarded to read_query."""
        mock_rag.get_reference_template_detail.return_value = [
            {
                "name": "tpl",
                "template": "SELECT 1",
                "parameters": "[]",
            }
        ]
        mock_db = MagicMock()
        mock_db.read_query.return_value = FuncToolResult(success=1, result=[{"1": 1}])
        tools.db_func_tool = mock_db

        tools.execute_reference_template(["Sales"], "tpl", "{}", datasource="analytics_db")
        mock_db.read_query.assert_called_once_with("SELECT 1", datasource="analytics_db")

    def test_available_tools_includes_execute_when_db_present(self, mock_agent_config, mock_rag):
        """available_tools includes execute_reference_template when db_func_tool is set."""
        mock_db = MagicMock()
        with patch(
            "datus.tools.func_tool.reference_template_tools.ReferenceTemplateRAG",
            return_value=mock_rag,
        ):
            t = ReferenceTemplateTools(mock_agent_config, db_func_tool=mock_db)
        tool_names = [tool.name for tool in t.available_tools()]
        assert "execute_reference_template" in tool_names

    def test_available_tools_excludes_execute_when_no_db(self, mock_agent_config, mock_rag):
        """available_tools excludes execute_reference_template when db_func_tool is None."""
        with patch(
            "datus.tools.func_tool.reference_template_tools.ReferenceTemplateRAG",
            return_value=mock_rag,
        ):
            t = ReferenceTemplateTools(mock_agent_config)
        tool_names = [tool.name for tool in t.available_tools()]
        assert "execute_reference_template" not in tool_names
        assert "render_reference_template" in tool_names


class TestReferenceTemplateToolsFactory:
    def test_create_dynamic(self, mock_agent_config, mock_rag):
        with patch(
            "datus.tools.func_tool.reference_template_tools.ReferenceTemplateRAG",
            return_value=mock_rag,
        ):
            tools = ReferenceTemplateTools.create_dynamic(mock_agent_config, sub_agent_name="test")
        assert tools.has_reference_templates is True

    def test_create_static(self, mock_agent_config, mock_rag):
        with patch(
            "datus.tools.func_tool.reference_template_tools.ReferenceTemplateRAG",
            return_value=mock_rag,
        ):
            tools = ReferenceTemplateTools.create_static(mock_agent_config, sub_agent_name="test", database_name="db")
        assert tools.has_reference_templates is True
