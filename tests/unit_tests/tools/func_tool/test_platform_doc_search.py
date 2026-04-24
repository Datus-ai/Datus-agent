# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for PlatformDocSearchTool - CI level, zero external dependencies."""

import logging
from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.platform_doc_search import PlatformDocSearchTool

# Patch targets for locally-imported symbols inside platform_doc_search.py methods
_SEARCH_TOOL_PATH = "datus.tools.search_tools.search_tool.SearchTool"
_TAVILY_PATH = "datus.tools.search_tools.search_tool.search_by_tavily"
_LIST_PLATFORMS_PATH = "datus.storage.document.store.list_indexed_platforms"
_TRANS_PATH = "datus.tools.func_tool.platform_doc_search.trans_to_function_tool"


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
    """Availability filter matrix: trio gated by indexed platforms,
    web_search_document gated by tavily key (config OR env)."""

    @staticmethod
    def _tool_names(tools):
        return [t._name for t in tools]

    @pytest.fixture
    def _patch_trans(self):
        with patch(_TRANS_PATH) as mock_trans:

            def _fake(func):
                mock = Mock()
                mock._name = func.__name__
                return mock

            mock_trans.side_effect = _fake
            yield mock_trans

    def test_all_available(self, mock_agent_config, _patch_trans, monkeypatch):
        mock_agent_config.tavily_api_key = "k"
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tool = PlatformDocSearchTool(agent_config=mock_agent_config)

        with patch(_LIST_PLATFORMS_PATH, return_value=["duckdb"]):
            tools = tool.available_tools()

        names = self._tool_names(tools)
        assert names == [
            "list_document_nav",
            "get_document",
            "search_document",
            "web_search_document",
        ]

    def test_only_web_when_no_platforms(self, mock_agent_config, _patch_trans, monkeypatch, caplog):
        mock_agent_config.tavily_api_key = "k"
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tool = PlatformDocSearchTool(agent_config=mock_agent_config)

        with caplog.at_level(logging.INFO, logger="datus.tools.func_tool.platform_doc_search"):
            with patch(_LIST_PLATFORMS_PATH, return_value=[]):
                tools = tool.available_tools()

        assert self._tool_names(tools) == ["web_search_document"]
        assert any("no indexed docstore" in rec.message for rec in caplog.records)
        assert not any("web_search_document" in rec.message and "Skipping" in rec.message for rec in caplog.records)

    def test_only_trio_when_no_tavily(self, mock_agent_config, _patch_trans, monkeypatch, caplog):
        mock_agent_config.tavily_api_key = None
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tool = PlatformDocSearchTool(agent_config=mock_agent_config)

        with caplog.at_level(logging.INFO, logger="datus.tools.func_tool.platform_doc_search"):
            with patch(_LIST_PLATFORMS_PATH, return_value=["duckdb"]):
                tools = tool.available_tools()

        assert self._tool_names(tools) == ["list_document_nav", "get_document", "search_document"]
        assert any("Skipping web_search_document" in rec.message for rec in caplog.records)

    def test_empty_when_nothing_available(self, mock_agent_config, _patch_trans, monkeypatch, caplog):
        mock_agent_config.tavily_api_key = None
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tool = PlatformDocSearchTool(agent_config=mock_agent_config)

        with caplog.at_level(logging.INFO, logger="datus.tools.func_tool.platform_doc_search"):
            with patch(_LIST_PLATFORMS_PATH, return_value=[]):
                tools = tool.available_tools()

        assert tools == []
        messages = " | ".join(rec.message for rec in caplog.records)
        assert "no indexed docstore" in messages
        assert "Skipping web_search_document" in messages

    def test_tavily_env_fallback(self, mock_agent_config, _patch_trans, monkeypatch):
        mock_agent_config.tavily_api_key = None
        monkeypatch.setenv("TAVILY_API_KEY", "envkey")
        tool = PlatformDocSearchTool(agent_config=mock_agent_config)

        with patch(_LIST_PLATFORMS_PATH, return_value=[]):
            tools = tool.available_tools()

        assert self._tool_names(tools) == ["web_search_document"]


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
    @pytest.fixture
    def tavily_tool(self):
        """Tool with tavily_api_key set so web_search_document reaches the Tavily call."""
        config = Mock()
        config.tavily_api_key = "test-tavily-key"
        return PlatformDocSearchTool(agent_config=config)

    def test_no_tavily_key_returns_empty(self, doc_search_tool):
        """When tavily_api_key is None, should return early with empty result."""
        result = doc_search_tool.web_search_document(keywords=["test"])
        assert result.success == 1
        assert result.result == []

    def test_success(self, tavily_tool):
        mock_result = Mock()
        mock_result.success = True
        mock_result.docs = ["doc1 content", "doc2 content"]
        mock_result.doc_count = 2

        with patch(_TAVILY_PATH, return_value=mock_result):
            result = tavily_tool.web_search_document(keywords=["snowflake COPY INTO"], max_results=5)

        assert result.success == 1
        assert result.result["doc_count"] == 2

    def test_success_with_include_domains(self, tavily_tool):
        mock_result = Mock()
        mock_result.success = True
        mock_result.docs = ["content"]
        mock_result.doc_count = 1

        with patch(_TAVILY_PATH, return_value=mock_result) as mock_fn:
            result = tavily_tool.web_search_document(
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
            api_key="test-tavily-key",
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

    def test_search_fails(self, tavily_tool):
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Tavily API error"

        with patch(_TAVILY_PATH, return_value=mock_result):
            result = tavily_tool.web_search_document(keywords=["test"])

        assert result.success == 0
        assert result.error == "Tavily API error"

    def test_exception_returns_failure(self, tavily_tool):
        with patch(_TAVILY_PATH, side_effect=Exception("network error")):
            result = tavily_tool.web_search_document(keywords=["test"])

        assert result.success == 0
        assert "network error" in result.error
