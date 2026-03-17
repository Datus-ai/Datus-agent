# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for PlatformDocSearchTool - CI level, zero external dependencies."""

from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.platform_doc_search import PlatformDocSearchTool

# Patch targets for locally-imported symbols inside platform_doc_search.py methods
_SEARCH_TOOL_PATH = "datus.tools.search_tools.search_tool.SearchTool"
_TAVILY_PATH = "datus.tools.search_tools.search_tool.search_by_tavily"


@pytest.fixture
def mock_agent_config():
    config = Mock()
    config.tavily_api_key = None
    return config


@pytest.fixture
def doc_search_tool(mock_agent_config):
    return PlatformDocSearchTool(agent_config=mock_agent_config)


class TestAllToolsName:
    def test_returns_expected_tool_names(self):
        names = PlatformDocSearchTool.all_tools_name()
        assert "list_document_nav" in names
        assert "get_document" in names
        assert "search_document" in names
        assert "web_search_document" in names
        assert len(names) == 4


class TestAvailableTools:
    def test_returns_four_tools(self, doc_search_tool):
        with patch("datus.tools.func_tool.platform_doc_search.trans_to_function_tool") as mock_trans:
            mock_trans.side_effect = lambda f: Mock(name=f.__name__)
            tools = doc_search_tool.available_tools()
        assert len(tools) == 4


class TestListDocumentNav:
    def _make_inner_tool(self, mock_result):
        mock_inner = Mock()
        mock_inner.list_document_nav.return_value = mock_result
        return mock_inner

    def test_success(self, doc_search_tool):
        mock_result = Mock()
        mock_result.success = True
        mock_result.platform = "duckdb"
        mock_result.version = "0.9"
        mock_result.nav_tree = [{"name": "DDL", "children": []}]
        mock_result.total_docs = 10

        with patch(_SEARCH_TOOL_PATH, return_value=self._make_inner_tool(mock_result)):
            result = doc_search_tool.list_document_nav(platform="duckdb")

        assert result.success == 1
        assert result.result["platform"] == "duckdb"
        assert result.result["version"] == "0.9"
        assert result.result["nav_tree"] == [{"name": "DDL", "children": []}]
        assert result.result["total_docs"] == 10

    def test_success_with_version(self, doc_search_tool):
        mock_result = Mock()
        mock_result.success = True
        mock_result.platform = "snowflake"
        mock_result.version = "7.0"
        mock_result.nav_tree = []
        mock_result.total_docs = 0

        mock_inner = self._make_inner_tool(mock_result)
        with patch(_SEARCH_TOOL_PATH, return_value=mock_inner):
            result = doc_search_tool.list_document_nav(platform="snowflake", version="7.0")

        assert result.success == 1
        mock_inner.list_document_nav.assert_called_once_with(platform="snowflake", version="7.0")

    def test_search_tool_returns_failure(self, doc_search_tool):
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Platform not found"

        with patch(_SEARCH_TOOL_PATH, return_value=self._make_inner_tool(mock_result)):
            result = doc_search_tool.list_document_nav(platform="unknown")

        assert result.success == 0
        assert result.error == "Platform not found"

    def test_exception_returns_failure(self, doc_search_tool):
        with patch(_SEARCH_TOOL_PATH, side_effect=Exception("import error")):
            result = doc_search_tool.list_document_nav(platform="duckdb")

        assert result.success == 0
        assert "import error" in result.error


class TestGetDocument:
    def _make_inner_tool(self, mock_result):
        mock_inner = Mock()
        mock_inner.get_document.return_value = mock_result
        return mock_inner

    def test_success(self, doc_search_tool):
        mock_result = Mock()
        mock_result.success = True
        mock_result.platform = "duckdb"
        mock_result.version = "0.9"
        mock_result.title = "CREATE TABLE"
        mock_result.hierarchy = "DDL > CREATE TABLE"
        mock_result.chunk_count = 3
        mock_result.chunks = [{"chunk_text": "content", "title": "CREATE TABLE"}]

        with patch(_SEARCH_TOOL_PATH, return_value=self._make_inner_tool(mock_result)):
            result = doc_search_tool.get_document(platform="duckdb", titles=["DDL", "CREATE TABLE"])

        assert result.success == 1
        assert result.result["title"] == "CREATE TABLE"
        assert result.result["chunk_count"] == 3
        assert len(result.result["chunks"]) == 1

    def test_search_tool_returns_failure(self, doc_search_tool):
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Document not found"

        with patch(_SEARCH_TOOL_PATH, return_value=self._make_inner_tool(mock_result)):
            result = doc_search_tool.get_document(platform="duckdb", titles=["Missing"])

        assert result.success == 0
        assert result.error == "Document not found"

    def test_exception_returns_failure(self, doc_search_tool):
        with patch(_SEARCH_TOOL_PATH, side_effect=Exception("conn error")):
            result = doc_search_tool.get_document(platform="duckdb", titles=["DDL"])

        assert result.success == 0
        assert "conn error" in result.error


class TestSearchDocument:
    def _make_inner_tool(self, mock_result):
        mock_inner = Mock()
        mock_inner.search_document.return_value = mock_result
        return mock_inner

    def test_success(self, doc_search_tool):
        mock_result = Mock()
        mock_result.success = True
        mock_result.docs = [{"chunk_text": "CREATE TABLE syntax", "title": "DDL"}]
        mock_result.doc_count = 1

        mock_inner = self._make_inner_tool(mock_result)
        with patch(_SEARCH_TOOL_PATH, return_value=mock_inner):
            result = doc_search_tool.search_document(
                platform="duckdb",
                keywords=["CREATE TABLE syntax"],
                top_n=3,
            )

        assert result.success == 1
        assert result.result["doc_count"] == 1
        mock_inner.search_document.assert_called_once_with(
            platform="duckdb",
            keywords=["CREATE TABLE syntax"],
            version=None,
            top_n=3,
        )

    def test_search_tool_returns_failure(self, doc_search_tool):
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Index unavailable"

        with patch(_SEARCH_TOOL_PATH, return_value=self._make_inner_tool(mock_result)):
            result = doc_search_tool.search_document(platform="duckdb", keywords=["test"])

        assert result.success == 0
        assert result.error == "Index unavailable"

    def test_exception_returns_failure(self, doc_search_tool):
        with patch(_SEARCH_TOOL_PATH, side_effect=Exception("timeout")):
            result = doc_search_tool.search_document(platform="duckdb", keywords=["ddl"])

        assert result.success == 0
        assert "timeout" in result.error


class TestWebSearchDocument:
    def test_success(self, doc_search_tool):
        mock_result = Mock()
        mock_result.success = True
        mock_result.docs = ["doc1 content", "doc2 content"]
        mock_result.doc_count = 2

        with patch(_TAVILY_PATH, return_value=mock_result):
            result = doc_search_tool.web_search_document(keywords=["snowflake COPY INTO"], max_results=5)

        assert result.success == 1
        assert result.result["doc_count"] == 2

    def test_success_with_include_domains(self, doc_search_tool):
        mock_result = Mock()
        mock_result.success = True
        mock_result.docs = ["content"]
        mock_result.doc_count = 1

        with patch(_TAVILY_PATH, return_value=mock_result) as mock_fn:
            result = doc_search_tool.web_search_document(
                keywords=["query"],
                max_results=3,
                include_domains=["docs.snowflake.com"],
            )

        assert result.success == 1
        mock_fn.assert_called_once_with(
            keywords=["query"],
            max_results=3,
            search_depth="advanced",
            include_answer="basic",
            include_raw_content="markdown",
            include_domains=["docs.snowflake.com"],
            api_key=None,
        )

    def test_uses_tavily_key_from_config(self, mock_agent_config):
        mock_agent_config.tavily_api_key = "my-tavily-key"
        tool = PlatformDocSearchTool(agent_config=mock_agent_config)

        mock_result = Mock()
        mock_result.success = True
        mock_result.docs = []
        mock_result.doc_count = 0

        with patch(_TAVILY_PATH, return_value=mock_result) as mock_fn:
            tool.web_search_document(keywords=["test"])

        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs["api_key"] == "my-tavily-key"

    def test_search_fails(self, doc_search_tool):
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Tavily API error"

        with patch(_TAVILY_PATH, return_value=mock_result):
            result = doc_search_tool.web_search_document(keywords=["test"])

        assert result.success == 0
        assert result.error == "Tavily API error"

    def test_exception_returns_failure(self, doc_search_tool):
        with patch(_TAVILY_PATH, side_effect=Exception("network error")):
            result = doc_search_tool.web_search_document(keywords=["test"])

        assert result.success == 0
        assert "network error" in result.error
