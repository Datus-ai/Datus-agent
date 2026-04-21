# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for Document SearchTool.

Tests list_document_nav, get_document, search_document, and delete_docs
against a real LanceDB-backed store with auto-constructed test data.
"""

from typing import List

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.document.schemas import PlatformDocChunk
from datus.storage.document.store import document_store
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_PLATFORM = "test_search_tool"


def _make_chunk(
    doc_path: str = "docs/sql-reference/functions/date-functions.md",
    chunk_index: int = 0,
    version: str = "v1",
    chunk_text: str = "The DATE_TRUNC function truncates a date or datetime value to a specified precision.",
    title: str = "DATE_TRUNC",
    titles: List[str] = None,
    nav_path: List[str] = None,
    group_name: str = "SQL Reference",
    hierarchy: str = "SQL Reference > Functions > Date Functions > DATE_TRUNC",
) -> PlatformDocChunk:
    """Build a PlatformDocChunk for testing."""
    return PlatformDocChunk(
        chunk_id=PlatformDocChunk.generate_chunk_id(doc_path, chunk_index, version),
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        title=title,
        titles=titles or [title],
        nav_path=nav_path or ["SQL Reference", "Functions", "Date Functions"],
        group_name=group_name,
        hierarchy=hierarchy,
        version=version,
        source_type="local",
        source_url="",
        doc_path=doc_path,
        keywords=["sql", "date"],
        language="en",
    )


def _build_test_chunks() -> List[PlatformDocChunk]:
    """Build a set of realistic test chunks covering multiple docs and hierarchy paths."""
    chunks = []

    # Doc 1: DATE_TRUNC (2 chunks)
    chunks.append(
        _make_chunk(
            doc_path="docs/sql-reference/functions/date-functions.md",
            chunk_index=0,
            chunk_text="The DATE_TRUNC function truncates a date or datetime value to a specified precision.",
            title="DATE_TRUNC",
            hierarchy="SQL Reference > Functions > Date Functions > DATE_TRUNC",
            nav_path=["SQL Reference", "Functions", "Date Functions"],
        )
    )
    chunks.append(
        _make_chunk(
            doc_path="docs/sql-reference/functions/date-functions.md",
            chunk_index=1,
            chunk_text="Syntax: DATE_TRUNC('unit', datetime_expr). Supported units: year, month, day, hour, minute.",
            title="DATE_TRUNC",
            hierarchy="SQL Reference > Functions > Date Functions > DATE_TRUNC",
            nav_path=["SQL Reference", "Functions", "Date Functions"],
        )
    )

    # Doc 2: CURRENT_DATE (1 chunk)
    chunks.append(
        _make_chunk(
            doc_path="docs/sql-reference/functions/current-date.md",
            chunk_index=0,
            chunk_text="CURRENT_DATE returns the current date as a DATE type value. It takes no arguments.",
            title="CURRENT_DATE",
            hierarchy="SQL Reference > Functions > Date Functions > CURRENT_DATE",
            nav_path=["SQL Reference", "Functions", "Date Functions"],
        )
    )

    # Doc 3: CREATE TABLE (2 chunks, different nav path)
    chunks.append(
        _make_chunk(
            doc_path="docs/sql-reference/ddl/create-table.md",
            chunk_index=0,
            chunk_text="CREATE TABLE creates a new table in the database. Supports column definitions and constraints.",
            title="CREATE TABLE",
            hierarchy="SQL Reference > DDL > CREATE TABLE",
            nav_path=["SQL Reference", "DDL"],
        )
    )
    chunks.append(
        _make_chunk(
            doc_path="docs/sql-reference/ddl/create-table.md",
            chunk_index=1,
            chunk_text=(
                "Example: CREATE TABLE users (id INT, name VARCHAR(100),"
                " created_at DATETIME DEFAULT CURRENT_TIMESTAMP)."
            ),
            title="CREATE TABLE",
            hierarchy="SQL Reference > DDL > CREATE TABLE",
            nav_path=["SQL Reference", "DDL"],
        )
    )

    # Doc 4: Loading data (different top-level nav)
    chunks.append(
        _make_chunk(
            doc_path="docs/loading/stream-load.md",
            chunk_index=0,
            chunk_text="Stream Load allows you to load data from local files into tables via HTTP PUT requests.",
            title="Stream Load",
            hierarchy="Data Loading > Stream Load",
            nav_path=["Data Loading"],
            group_name="Data Loading",
        )
    )

    return chunks


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_agent_config()


@pytest.mark.acceptance
class TestSearchTool:
    """Test SearchTool methods: list_document_nav, get_document, search_document.

    Auto-constructs test data in a dedicated platform store, then exercises
    all three SearchTool methods against it.
    """

    @pytest.fixture(autouse=True)
    def setup_store(self, agent_config: AgentConfig):
        """Populate a test store with constructed data and initialize SearchTool."""
        self.agent_config = agent_config
        self.store = document_store(TEST_PLATFORM)

        # Ensure clean state, then populate with test chunks
        self.store.delete_docs(version=None)
        self.store.store_chunks(_build_test_chunks())

        from datus.tools.search_tools.search_tool import SearchTool

        self.tool = SearchTool(agent_config=agent_config)

        yield

        # Cleanup after each test
        self.store.delete_docs(version=None)

    # ----- list_document_nav + get_document (dependent) -----

    def test_list_document_nav_returns_tree(self):
        """list_document_nav should return a hierarchical navigation tree."""
        result = self.tool.list_document_nav(platform=TEST_PLATFORM, version="v1")
        assert result.success, f"list_document_nav failed: {result.error}"
        assert result.platform == TEST_PLATFORM
        assert result.total_docs > 0, "Should have at least 1 unique doc"
        assert len(result.nav_tree) > 0, "Nav tree should not be empty"

        # Each top-level item should be a pure tree node with name and children
        for item in result.nav_tree:
            assert "name" in item or "version" in item, "Tree node should have 'name' or 'version'"
            if "name" in item:
                assert "children" in item, "Tree node should have 'children'"
                assert isinstance(item["children"], list), "children should be a list"
                assert item["name"], "name should not be empty"
            else:
                assert "tree" in item, "Multi versions should have 'tree'"

    def test_list_document_nav_empty_platform(self):
        """list_document_nav for a non-existent platform returns empty tree."""
        result = self.tool.list_document_nav(platform="nonexistent_platform_xyz")

        assert result.success
        assert result.total_docs == 0
        assert result.nav_tree == []

    def test_get_document_by_title_from_nav(self):
        """get_document should retrieve chunks when given a title from the nav tree."""
        # First get the nav tree to find a real title
        nav_result = self.tool.list_document_nav(platform=TEST_PLATFORM, version="v1")
        assert nav_result.success and len(nav_result.nav_tree) > 0

        # Find a leaf node (node with empty children) from the pure tree
        def _find_first_doc(nodes):
            for node in nodes:
                if "tree" in node:
                    found = _find_first_doc(node["tree"])
                    if found:
                        return found
                    continue
                if not node.get("children"):
                    return node.get("name")
                found = _find_first_doc(node["children"])
                if found:
                    return found
            return None

        title = _find_first_doc(nav_result.nav_tree)
        assert title, "Should find at least one document title in the nav tree"

        # Use the title to get document chunks
        result = self.tool.get_document(platform=TEST_PLATFORM, titles=[title], version="v1")

        assert result.success, f"get_document failed: {result.error}"
        assert result.platform == TEST_PLATFORM
        assert result.chunk_count > 0, f"Should find chunks for title '{title}'"
        assert len(result.chunks) == result.chunk_count

        # Chunks should have expected fields
        for chunk in result.chunks:
            assert "chunk_text" in chunk
            assert chunk["chunk_text"], "chunk_text should not be empty"

    def test_get_document_no_match(self):
        """get_document with a non-existent title returns empty results."""
        result = self.tool.get_document(
            platform="local",
            titles=["ZZZZZ_NONEXISTENT_TITLE_ZZZZZ"],
        )

        assert result.success
        assert result.chunk_count == 0
        assert result.chunks == []

    # ----- search_document -----

    def test_search_document_returns_results(self):
        """search_document should find results for a relevant keyword."""
        result = self.tool.search_document(
            platform=TEST_PLATFORM,
            keywords=["DATE_TRUNC"],
            version="v1",
            top_n=3,
        )

        assert result.success, f"search_document failed: {result.error}"
        assert result.doc_count > 0, "Should find at least one result for keyword DATE_TRUNC"

        assert "DATE_TRUNC" in result.docs, "Keyword 'DATE_TRUNC' missing from result.docs"
        assert len(result.docs["DATE_TRUNC"]) > 0, "No results for keyword 'DATE_TRUNC'"

        # Verify chunk fields from the first result
        first_chunk = result.docs["DATE_TRUNC"][0]
        assert "chunk_text" in first_chunk
        assert "title" in first_chunk
        assert "hierarchy" in first_chunk
        assert "doc_path" in first_chunk

    def test_search_document_multiple_keywords(self):
        """search_document with multiple keywords returns results for each."""
        result = self.tool.search_document(
            platform=TEST_PLATFORM,
            keywords=["DATE_TRUNC", "CURRENT_DATE"],
            version="v1",
            top_n=3,
        )

        assert result.success, f"search_document failed: {result.error}"
        # Each keyword should have its own entry in the docs dict
        for keyword in ["DATE_TRUNC", "CURRENT_DATE"]:
            assert keyword in result.docs, f"Keyword '{keyword}' missing from result.docs"

    def test_search_document_wrong_platform(self):
        """search_document on a non-existent platform returns empty results."""
        result = self.tool.search_document(
            platform="nonexistent_platform_xyz",
            keywords=["anything"],
            top_n=3,
        )

        assert result.success
        assert result.doc_count == 0

    # ----- delete_docs -----

    def test_delete_docs_by_version(self):
        """delete_docs(version=...) should remove only chunks of that version."""
        tmp_store = document_store("test_version_filter")
        # Clear any leftover data from previous runs
        tmp_store.delete_docs(version=None)

        # Insert chunks for two versions
        chunks_v1 = [
            PlatformDocChunk(
                chunk_id=PlatformDocChunk.generate_chunk_id("a.md", i, "v1"),
                chunk_text=f"v1 chunk {i}",
                chunk_index=i,
                title="Doc A",
                titles=["Doc A"],
                nav_path=[],
                group_name="",
                hierarchy="Doc A",
                version="v1",
                source_type="local",
                source_url="",
                doc_path="a.md",
            )
            for i in range(3)
        ]
        chunks_v2 = [
            PlatformDocChunk(
                chunk_id=PlatformDocChunk.generate_chunk_id("b.md", i, "v2"),
                chunk_text=f"v2 chunk {i}",
                chunk_index=i,
                title="Doc B",
                titles=["Doc B"],
                nav_path=[],
                group_name="",
                hierarchy="Doc B",
                version="v2",
                source_type="local",
                source_url="",
                doc_path="b.md",
            )
            for i in range(2)
        ]
        tmp_store.store_chunks(chunks_v1 + chunks_v2)

        # Verify both versions exist
        assert tmp_store.table.count_rows() == 5

        # Delete v1 only
        deleted = tmp_store.delete_docs(version="v1")
        assert deleted == 3

        # v2 should remain
        remaining = tmp_store.table.count_rows()
        assert remaining == 2

    def test_delete_docs_all(self):
        """delete_docs(version=None) should remove all chunks (drop + recreate)."""
        tmp_store = document_store("test_get_stats")
        # Clear any leftover data from previous runs
        tmp_store.delete_docs(version=None)

        chunks = [
            PlatformDocChunk(
                chunk_id=PlatformDocChunk.generate_chunk_id("c.md", i, "v1"),
                chunk_text=f"chunk {i}",
                chunk_index=i,
                title="Doc C",
                titles=["Doc C"],
                nav_path=[],
                group_name="",
                hierarchy="Doc C",
                version="v1",
                source_type="local",
                source_url="",
                doc_path="c.md",
            )
            for i in range(4)
        ]
        tmp_store.store_chunks(chunks)
        assert tmp_store.table.count_rows() == 4

        # Delete all
        deleted = tmp_store.delete_docs(version=None)
        assert deleted == 4
        assert tmp_store.table.count_rows() == 0

    def test_delete_docs_empty_store(self):
        """delete_docs on an empty store should return 0."""
        tmp_store = document_store("test_delete_empty")
        # Clear any leftover data from previous runs
        tmp_store.delete_docs(version=None)
        deleted = tmp_store.delete_docs(version="v1")
        assert deleted == 0
