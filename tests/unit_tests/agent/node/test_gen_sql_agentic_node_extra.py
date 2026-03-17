# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Additional unit tests for GenSQLAgenticNode covering uncovered methods.

CI-level: zero external deps, zero network, zero API keys.
Focuses on helper methods not covered by test_gen_sql_agentic_node.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(real_agent_config, mock_llm_create, node_name="gensql"):
    from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

    return GenSQLAgenticNode(
        node_id="gensql_extra",
        description="Extra gensql test",
        node_type=NodeType.TYPE_GENSQL,
        agent_config=real_agent_config,
        node_name=node_name,
    )


# ---------------------------------------------------------------------------
# TestGetSqlPreview (static method)
# ---------------------------------------------------------------------------


class TestGetSqlPreview:
    def test_short_sql_no_truncation(self):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        sql = "SELECT id, name\nFROM users"
        result = GenSQLAgenticNode._get_sql_preview(sql, max_lines=5)
        assert result == sql

    def test_long_sql_truncated(self):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        sql = "\n".join([f"line{i}" for i in range(10)])
        result = GenSQLAgenticNode._get_sql_preview(sql, max_lines=3)
        assert "line0" in result
        assert "line1" in result
        assert "line2" in result
        assert "7 more lines" in result

    def test_exact_lines_no_truncation(self):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        sql = "a\nb\nc"
        result = GenSQLAgenticNode._get_sql_preview(sql, max_lines=3)
        assert result == sql
        assert "more lines" not in result

    def test_single_line(self):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        sql = "SELECT 1"
        result = GenSQLAgenticNode._get_sql_preview(sql, max_lines=5)
        assert result == "SELECT 1"


# ---------------------------------------------------------------------------
# TestGetSqlPreviewLines
# ---------------------------------------------------------------------------


class TestGetSqlPreviewLines:
    def test_default_preview_lines(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        result = node._get_sql_preview_lines()
        assert result == 5  # default

    def test_custom_preview_lines_from_config(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.node_config["sql_preview_lines"] = "10"
        result = node._get_sql_preview_lines()
        assert result == 10


# ---------------------------------------------------------------------------
# TestReadExistingSqlFile
# ---------------------------------------------------------------------------


class TestReadExistingSqlFile:
    def test_returns_none_when_no_filesystem_tool(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.filesystem_func_tool = None
        result = node._read_existing_sql_file("some/file.sql")
        assert result is None

    def test_returns_none_when_empty_path(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        result = node._read_existing_sql_file("")
        assert result is None

    def test_returns_content_on_success(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_fs_tool = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = "SELECT 1 FROM table"
        mock_fs_tool.read_file.return_value = mock_result
        node.filesystem_func_tool = mock_fs_tool

        result = node._read_existing_sql_file("query.sql")
        assert result == "SELECT 1 FROM table"

    def test_returns_none_when_read_fails(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_fs_tool = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.result = None
        mock_fs_tool.read_file.return_value = mock_result
        node.filesystem_func_tool = mock_fs_tool

        result = node._read_existing_sql_file("query.sql")
        assert result is None

    def test_returns_none_when_result_not_string(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_fs_tool = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = {"key": "value"}  # not a string
        mock_fs_tool.read_file.return_value = mock_result
        node.filesystem_func_tool = mock_fs_tool

        result = node._read_existing_sql_file("query.sql")
        assert result is None


# ---------------------------------------------------------------------------
# TestSetupMcpServers
# ---------------------------------------------------------------------------


class TestSetupMcpServers:
    def test_empty_mcp_config_returns_empty(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.node_config = {"mcp": ""}
        result = node._setup_mcp_servers()
        assert result == {}

    def test_no_mcp_config_returns_empty(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.node_config = {}
        result = node._setup_mcp_servers()
        assert result == {}

    def test_metricflow_mcp_setup(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.node_config = {"mcp": "metricflow_mcp"}

        mock_server = MagicMock()
        with patch.object(node, "_setup_metricflow_mcp", return_value=mock_server):
            with patch.object(node, "_setup_mcp_server_from_config", return_value=None):
                result = node._setup_mcp_servers()

        assert "metricflow_mcp" in result


# ---------------------------------------------------------------------------
# TestSetupInput
# ---------------------------------------------------------------------------


class TestSetupInputGenSQL:
    def test_setup_input_creates_gen_sql_node_input(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.input = None  # force creation

        wf = MagicMock()
        wf.task.task = "Find total sales"
        wf.task.external_knowledge = "revenue = total sales"
        wf.task.catalog_name = "cat"
        wf.task.database_name = "california_schools"
        wf.task.schema_name = "main"
        wf.task.current_date = None
        wf.context.table_schemas = []
        wf.context.metrics = []
        wf.metadata.get.return_value = False

        result = node.setup_input(wf)

        assert result["success"] is True
        assert isinstance(node.input, GenSQLNodeInput)
        assert node.input.user_message == "Find total sales"

    def test_setup_input_updates_existing_input(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.input = GenSQLNodeInput(user_message="old message")

        wf = MagicMock()
        wf.task.task = "new message"
        wf.task.external_knowledge = ""
        wf.task.catalog_name = ""
        wf.task.database_name = "california_schools"
        wf.task.schema_name = ""
        wf.task.current_date = None
        wf.context.table_schemas = []
        wf.context.metrics = []
        wf.metadata.get.return_value = False

        node.setup_input(wf)

        assert node.input.user_message == "new message"

    def test_setup_input_sets_date_parsing_reference(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_date_tools = MagicMock()
        node.date_parsing_tools = mock_date_tools

        wf = MagicMock()
        wf.task.task = "query"
        wf.task.external_knowledge = ""
        wf.task.catalog_name = ""
        wf.task.database_name = "california_schools"
        wf.task.schema_name = ""
        wf.task.current_date = "2024-01-15"
        wf.context.table_schemas = []
        wf.context.metrics = []
        wf.metadata.get.return_value = False

        node.setup_input(wf)

        mock_date_tools.set_reference_date.assert_called_once_with("2024-01-15")


# ---------------------------------------------------------------------------
# TestRebuildTools
# ---------------------------------------------------------------------------


class TestRebuildTools:
    def test_rebuild_tools_with_all_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)

        mock_db = MagicMock()
        mock_db.available_tools.return_value = [MagicMock(name="list_tables")]
        mock_ctx = MagicMock()
        mock_ctx.available_tools.return_value = [MagicMock(name="search_schema")]
        mock_date = MagicMock()
        mock_date.available_tools.return_value = [MagicMock(name="parse_date")]

        node.db_func_tool = mock_db
        node.context_search_tools = mock_ctx
        node.date_parsing_tools = mock_date
        node.filesystem_func_tool = None
        node._platform_doc_tool = None

        node._rebuild_tools()

        assert len(node.tools) == 3

    def test_rebuild_tools_empty_when_no_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.db_func_tool = None
        node.context_search_tools = None
        node.date_parsing_tools = None
        node.filesystem_func_tool = None
        node._platform_doc_tool = None

        node._rebuild_tools()

        assert node.tools == []


# ---------------------------------------------------------------------------
# TestGetNodeName
# ---------------------------------------------------------------------------


class TestGetNodeNameGenSQL:
    def test_get_node_name_returns_configured_name(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create, node_name="gensql")
        assert node.get_node_name() == "gensql"


# ---------------------------------------------------------------------------
# TestExecuteStreamGenSQL (focused on input_not_set)
# ---------------------------------------------------------------------------


class TestExecuteStreamGenSQLExtra:
    @pytest.mark.asyncio
    async def test_execute_stream_raises_when_no_input(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.input = None

        with pytest.raises(ValueError, match="GenSQL input not set"):
            async for _ in node.execute_stream():
                pass
