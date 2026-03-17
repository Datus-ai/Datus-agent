# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.tools.search_tools.search_tool — _get_document_store."""

from unittest.mock import MagicMock, patch

from datus.schemas.doc_search_node_models import DocSearchInput, DocSearchResult
from datus.tools.search_tools.search_tool import SearchTool, search_by_tavily


class TestGetDocumentStore:
    """Tests for SearchTool._get_document_store."""

    def test_returns_store_when_has_data(self):
        """Should return the store when it has data."""
        mock_store = MagicMock()
        mock_store.has_data.return_value = True

        with patch("datus.tools.search_tools.search_tool.document_store", return_value=mock_store):
            tool = SearchTool.__new__(SearchTool)
            result = tool._get_document_store("test_platform")

        assert result is mock_store

    def test_returns_none_when_no_data(self):
        """Should return None when the store has no data."""
        mock_store = MagicMock()
        mock_store.has_data.return_value = False

        with patch("datus.tools.search_tools.search_tool.document_store", return_value=mock_store):
            tool = SearchTool.__new__(SearchTool)
            result = tool._get_document_store("test_platform")

        assert result is None


def _make_tool():
    mock_config = MagicMock()
    tool = SearchTool.__new__(SearchTool)
    tool.agent_config = mock_config
    return tool


def _make_store_with_data(rows=None, versions=None):
    store = MagicMock()
    store.has_data.return_value = True
    store.get_all_rows.return_value = rows or []
    store.list_versions.return_value = versions or []
    store.search_docs.return_value = []
    return store


class TestVersionSortKey:
    def test_simple_versions(self):
        versions = ["v1.0", "v2.0", "v10.0", "v3.0"]
        sorted_v = sorted(versions, key=SearchTool._version_sort_key)
        assert sorted_v == ["v1.0", "v2.0", "v3.0", "v10.0"]

    def test_handles_non_numeric_prefix(self):
        key_a = SearchTool._version_sort_key("v3.9")
        key_b = SearchTool._version_sort_key("v3.10")
        assert key_a < key_b  # 3.9 < 3.10 semantically

    def test_empty_string(self):
        key = SearchTool._version_sort_key("")
        assert key == [""]


class TestResolveLatestVersion:
    def test_returns_none_for_empty_store(self):
        store = MagicMock()
        store.list_versions.return_value = []
        result = SearchTool._resolve_latest_version(store)
        assert result is None

    def test_returns_latest_version(self):
        store = MagicMock()
        store.list_versions.return_value = [
            {"version": "v1.0"},
            {"version": "v3.0"},
            {"version": "v2.0"},
        ]
        result = SearchTool._resolve_latest_version(store)
        assert result == "v3.0"

    def test_semantic_version_ordering(self):
        store = MagicMock()
        store.list_versions.return_value = [{"version": "v3.9"}, {"version": "v3.10"}]
        result = SearchTool._resolve_latest_version(store)
        assert result == "v3.10"


class TestNormalizeListField:
    def test_list_input_returned_as_is(self):
        result = SearchTool._normalize_list_field(["a", "b"])
        assert result == ["a", "b"]

    def test_string_split_by_gt(self):
        result = SearchTool._normalize_list_field("DDL > CREATE TABLE")
        assert result == ["DDL", "CREATE TABLE"]

    def test_empty_string_returns_empty_list(self):
        result = SearchTool._normalize_list_field("")
        assert result == []

    def test_none_returns_empty_list(self):
        result = SearchTool._normalize_list_field(None)
        assert result == []


class TestBuildNavTree:
    def test_empty_doc_map_returns_empty_list(self):
        tool = _make_tool()
        result = tool._build_nav_tree({})
        assert result == []

    def test_simple_nav_path(self):
        tool = _make_tool()
        doc_map = {
            "doc1.md": {"nav_path": ["SQL Reference", "DDL"], "title": "CREATE TABLE"},
        }
        result = tool._build_nav_tree(doc_map)
        assert len(result) > 0
        # Top level should be "SQL Reference"
        names = [n["name"] for n in result]
        assert "SQL Reference" in names

    def test_multiple_docs_in_same_path(self):
        tool = _make_tool()
        doc_map = {
            "doc1.md": {"nav_path": ["SQL"], "title": "SELECT"},
            "doc2.md": {"nav_path": ["SQL"], "title": "INSERT"},
        }
        result = tool._build_nav_tree(doc_map)
        sql_node = next(n for n in result if n["name"] == "SQL")
        children_names = [c["name"] for c in sql_node["children"]]
        assert "SELECT" in children_names
        assert "INSERT" in children_names

    def test_leaf_title_same_as_last_nav_segment_not_duplicated(self):
        tool = _make_tool()
        doc_map = {
            "doc.md": {"nav_path": ["Admin", "Backup"], "title": "Backup"},
        }
        result = tool._build_nav_tree(doc_map)
        admin_node = next(n for n in result if n["name"] == "Admin")
        # "Backup" should appear once as a child, not duplicated
        backup_nodes = [c for c in admin_node["children"] if c["name"] == "Backup"]
        assert len(backup_nodes) == 1


class TestListDocumentNav:
    def test_returns_empty_when_no_store(self):
        tool = _make_tool()
        with patch.object(tool, "_get_document_store", return_value=None):
            result = tool.list_document_nav("test_platform")
        assert result.success is True
        assert result.nav_tree == []
        assert result.total_docs == 0

    def test_returns_empty_when_no_rows(self):
        tool = _make_tool()
        store = _make_store_with_data(rows=[])
        with patch.object(tool, "_get_document_store", return_value=store), patch.object(
            tool, "_resolve_latest_version", return_value="v1.0"
        ):
            result = tool.list_document_nav("test_platform", version="v1.0")
        assert result.success is True
        assert result.total_docs == 0

    def test_returns_nav_tree_with_data(self):
        tool = _make_tool()
        rows = [
            {"doc_path": "doc1.md", "version": "v1.0", "nav_path": ["SQL"], "title": "SELECT"},
        ]
        store = _make_store_with_data(rows=rows, versions=[{"version": "v1.0"}])
        with patch.object(tool, "_get_document_store", return_value=store):
            result = tool.list_document_nav("test_platform", version="v1.0")
        assert result.success is True
        assert result.total_docs == 1

    def test_handles_exception_gracefully(self):
        tool = _make_tool()
        with patch.object(tool, "_get_document_store", side_effect=Exception("store error")):
            result = tool.list_document_nav("test_platform")
        assert result.success is False
        assert "store error" in result.error

    def test_defaults_to_latest_version_when_not_specified(self):
        tool = _make_tool()
        store = _make_store_with_data(rows=[], versions=[{"version": "v2.0"}])
        with patch.object(tool, "_get_document_store", return_value=store), patch.object(
            tool, "_resolve_latest_version", return_value="v2.0"
        ) as mock_latest:
            tool.list_document_nav("platform")
        mock_latest.assert_called_once()


class TestGetDocument:
    def test_returns_error_for_empty_platform(self):
        tool = _make_tool()
        result = tool.get_document("", ["DDL"])
        assert result.success is False
        assert "platform" in result.error

    def test_returns_error_for_empty_titles(self):
        tool = _make_tool()
        result = tool.get_document("snowflake", [])
        assert result.success is False
        assert "titles" in result.error

    def test_returns_empty_when_no_store(self):
        tool = _make_tool()
        with patch.object(tool, "_get_document_store", return_value=None):
            result = tool.get_document("snowflake", ["DDL"])
        assert result.success is True
        assert result.chunks == []

    def test_returns_empty_when_no_rows(self):
        tool = _make_tool()
        store = _make_store_with_data(rows=[])
        with patch.object(tool, "_get_document_store", return_value=store), patch.object(
            tool, "_resolve_latest_version", return_value="v1.0"
        ):
            result = tool.get_document("snowflake", ["DDL"], version="v1.0")
        assert result.success is True
        assert result.chunk_count == 0

    def test_returns_matching_chunks(self):
        tool = _make_tool()
        rows = [
            {
                "doc_path": "ddl/create.md",
                "chunk_id": "c1",
                "chunk_index": 0,
                "chunk_text": "CREATE TABLE ...",
                "title": "CREATE TABLE",
                "titles": ["DDL", "CREATE TABLE"],
                "hierarchy": "SQL > DDL > CREATE TABLE",
                "nav_path": ["SQL", "DDL"],
                "version": "v1.0",
                "keywords": [],
            }
        ]
        store = _make_store_with_data(rows=rows)
        with patch.object(tool, "_get_document_store", return_value=store), patch.object(
            tool, "_resolve_latest_version", return_value="v1.0"
        ):
            result = tool.get_document("snowflake", ["DDL", "CREATE TABLE"], version="v1.0")
        assert result.success is True
        assert result.chunk_count == 1

    def test_handles_exception_gracefully(self):
        tool = _make_tool()
        with patch.object(tool, "_get_document_store", side_effect=RuntimeError("db error")):
            result = tool.get_document("snowflake", ["DDL"])
        assert result.success is False
        assert "db error" in result.error


class TestSearchDocument:
    def test_returns_error_for_empty_platform(self):
        tool = _make_tool()
        result = tool.search_document("", ["keyword"])
        assert result.success is False
        assert "platform" in result.error

    def test_returns_error_for_empty_keywords(self):
        tool = _make_tool()
        result = tool.search_document("snowflake", [])
        assert result.success is False
        assert "keywords" in result.error

    def test_returns_empty_when_no_store(self):
        tool = _make_tool()
        with patch.object(tool, "_get_document_store", return_value=None):
            result = tool.search_document("snowflake", ["select"])
        assert result.success is True
        assert result.doc_count == 0

    def test_searches_each_keyword(self):
        tool = _make_tool()
        mock_doc = {"chunk_id": "c1", "chunk_text": "SELECT ..."}
        store = _make_store_with_data()
        store.search_docs.return_value = [mock_doc]
        with patch.object(tool, "_get_document_store", return_value=store), patch.object(
            tool, "_resolve_latest_version", return_value="v1.0"
        ):
            result = tool.search_document("snowflake", ["select", "insert"], top_n=3)
        assert result.success is True
        assert result.doc_count == 2  # 1 doc per keyword
        assert "select" in result.docs
        assert "insert" in result.docs

    def test_handles_keyword_search_exception(self):
        tool = _make_tool()
        store = _make_store_with_data()
        store.search_docs.side_effect = Exception("search failed")
        with patch.object(tool, "_get_document_store", return_value=store), patch.object(
            tool, "_resolve_latest_version", return_value="v1.0"
        ):
            result = tool.search_document("snowflake", ["badkeyword"])
        assert result.success is True
        assert result.docs["badkeyword"] == []

    def test_handles_outer_exception(self):
        tool = _make_tool()
        with patch.object(tool, "_get_document_store", side_effect=RuntimeError("outer error")):
            result = tool.search_document("snowflake", ["kw"])
        assert result.success is False

    def test_execute_delegates_to_search_document(self):
        tool = _make_tool()
        input_data = DocSearchInput(platform="snowflake", keywords=["select"], version="v1.0", top_n=5)
        with patch.object(tool, "search_document", return_value=DocSearchResult(success=True)) as mock_search:
            tool.execute(input_data)
        mock_search.assert_called_once_with(
            platform="snowflake",
            keywords=["select"],
            version="v1.0",
            top_n=5,
        )


class TestSearchByTavily:
    def test_returns_error_when_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            result = search_by_tavily(["select"], api_key=None)
        assert result.success is False
        assert "TAVILY_API_KEY" in result.error

    def test_returns_empty_for_empty_keywords(self):
        result = search_by_tavily([], api_key="fake_key")
        assert result.success is True
        assert result.doc_count == 0

    def test_successful_search(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"content": "SELECT docs here", "raw_content": None},
            ],
            "answer": None,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            result = search_by_tavily(["SELECT"], api_key="test_key")

        assert result.success is True
        assert result.doc_count == 1

    def test_prepends_answer_when_available(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"content": "result 1"}],
            "answer": "This is the answer",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            result = search_by_tavily(["q"], api_key="key", include_answer="basic")

        assert result.doc_count == 2  # answer + 1 result

    def test_http_error_returns_failure(self):
        import requests as req

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        http_error = req.HTTPError(response=mock_response)

        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = http_error
            result = search_by_tavily(["q"], api_key="bad_key")

        assert result.success is False

    def test_connection_error_returns_failure(self):
        with patch("requests.post", side_effect=Exception("connection refused")):
            result = search_by_tavily(["q"], api_key="key")
        assert result.success is False
        assert "External search failed" in result.error

    def test_uses_env_api_key(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [], "answer": None}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response), patch.dict(
            "os.environ", {"TAVILY_API_KEY": "env_key"}
        ):
            result = search_by_tavily(["q"])
        assert result.success is True

    def test_include_domains_added_to_payload(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [], "answer": None}
        mock_response.raise_for_status = MagicMock()
        captured = {}

        def capture_post(url, json, headers, timeout):
            captured["payload"] = json
            return mock_response

        with patch("requests.post", side_effect=capture_post):
            search_by_tavily(["q"], api_key="key", include_domains=["docs.snowflake.com"])

        assert "include_domains" in captured["payload"]
        assert captured["payload"]["include_domains"] == ["docs.snowflake.com"]
