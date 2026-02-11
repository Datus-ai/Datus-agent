# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Tests for Document Storage Module

Includes tests for:
- DocumentStore and search functionality
- Document schemas and data models
- Parsers (Markdown, HTML)
- Chunker (SemanticChunker)
- Cleaner (DocumentCleaner)
- Metadata extraction
- Rate limiter
- Integration tests (Fetch → Parse → Clean → Chunk)
"""

from typing import List

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.document.schemas import (
    CONTENT_TYPE_MARKDOWN,
    SOURCE_TYPE_GITHUB,
    FetchedDocument,
    ParsedDocument,
    ParsedSection,
    PlatformDocChunk,
)
from datus.storage.document.store import document_store, get_platform_doc_schema
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@pytest.fixture
def agent_config() -> AgentConfig:
    return load_agent_config()


# =============================================================================
# Schema Tests
# =============================================================================


class TestSchemas:
    """Test data model schemas."""

    def test_fetched_document_creation(self):
        """Test FetchedDocument creation."""
        doc = FetchedDocument(
            platform="snowflake",
            version="v1.2.3",
            source_url="https://github.com/test/repo",
            source_type=SOURCE_TYPE_GITHUB,
            doc_path="docs/intro.md",
            raw_content="# Introduction\n\nTest content.",
            content_type=CONTENT_TYPE_MARKDOWN,
        )

        assert doc.platform == "snowflake"
        assert doc.version == "v1.2.3"
        assert doc.content_type == CONTENT_TYPE_MARKDOWN

    def test_parsed_section_get_all_content(self):
        """Test ParsedSection content aggregation."""
        child = ParsedSection(level=2, title="Child", content="Child content")
        parent = ParsedSection(
            level=1,
            title="Parent",
            content="Parent content",
            children=[child],
        )

        all_content = parent.get_all_content()
        assert "Parent content" in all_content
        assert "Child content" in all_content

    def test_platform_doc_chunk_generate_id(self):
        """Test chunk ID generation is deterministic."""
        id1 = PlatformDocChunk.generate_chunk_id(
            doc_path="docs/intro.md",
            chunk_index=0,
            version="v1.0.0",
        )
        id2 = PlatformDocChunk.generate_chunk_id(
            doc_path="docs/intro.md",
            chunk_index=0,
            version="v1.0.0",
        )

        assert id1 == id2
        assert len(id1) == 32  # MD5 hash length

    def test_platform_doc_chunk_different_inputs(self):
        """Test that different inputs produce different IDs."""
        id1 = PlatformDocChunk.generate_chunk_id("docs/a.md", 0, "v1")
        id2 = PlatformDocChunk.generate_chunk_id("docs/b.md", 0, "v1")
        id3 = PlatformDocChunk.generate_chunk_id("docs/a.md", 1, "v1")

        assert id1 != id2
        assert id1 != id3

    def test_get_platform_doc_schema(self):
        """Test schema generation."""
        schema = get_platform_doc_schema(embedding_dim=384)

        field_names = [field.name for field in schema]
        assert "chunk_id" in field_names
        assert "chunk_text" in field_names
        assert "vector" in field_names
        assert "version" in field_names
        assert "titles" in field_names
        assert "platform" not in field_names


# =============================================================================
# Parser Tests
# =============================================================================


class TestMarkdownParser:
    """Test Markdown parser."""

    def test_parse_simple_document(self):
        """Test parsing a simple Markdown document."""
        from datus.storage.document.parser.markdown_parser import MarkdownParser

        parser = MarkdownParser()

        doc = FetchedDocument(
            platform="test",
            version="v1",
            source_url="http://test",
            source_type=SOURCE_TYPE_GITHUB,
            doc_path="test.md",
            raw_content="# Title\n\nSome content.\n\n## Section\n\nMore content.",
            content_type=CONTENT_TYPE_MARKDOWN,
        )

        parsed = parser.parse(doc)

        assert parsed.title == "Title"
        assert len(parsed.sections) > 0

    def test_parse_with_code_blocks(self):
        """Test parsing preserves code blocks."""
        from datus.storage.document.parser.markdown_parser import MarkdownParser

        parser = MarkdownParser()

        content = """# Guide

Here is code:

```python
def hello():
    print("Hello")
```

More text.
"""
        doc = FetchedDocument(
            platform="test",
            version="v1",
            source_url="http://test",
            source_type=SOURCE_TYPE_GITHUB,
            doc_path="test.md",
            raw_content=content,
            content_type=CONTENT_TYPE_MARKDOWN,
        )

        parsed = parser.parse(doc)

        # Check code block is preserved somewhere in the content
        all_content = ""
        for section in parsed.sections:
            all_content += section.get_all_content()

        assert "```python" in all_content or "def hello" in all_content


# =============================================================================
# Chunker Tests
# =============================================================================


class TestSemanticChunker:
    """Test semantic chunker."""

    def test_chunk_simple_document(self):
        """Test chunking a simple document."""
        from datus.storage.document.chunker.semantic_chunker import SemanticChunker

        chunker = SemanticChunker()

        parsed = ParsedDocument(
            title="Test Document",
            sections=[
                ParsedSection(
                    level=1,
                    title="Introduction",
                    content="This is the introduction.",
                    children=[],
                ),
                ParsedSection(
                    level=2,
                    title="Details",
                    content="These are the details.",
                    children=[],
                ),
            ],
            metadata={
                "nav_path": ["Guides", "User Guide"],
                "group_name": "Guides",
            },
        )

        metadata = {
            "platform": "test",
            "version": "v1",
            "source_type": "github",
            "source_url": "http://test",
            "doc_path": "test.md",
        }

        chunks = chunker.chunk(parsed, metadata)

        assert len(chunks) > 0
        assert all(isinstance(c, PlatformDocChunk) for c in chunks)
        # Verify new fields
        for chunk in chunks:
            assert chunk.nav_path == ["Guides", "User Guide"]
            assert chunk.group_name == "Guides"
            assert "Guides" in chunk.hierarchy

    def test_chunk_preserves_code_blocks(self):
        """Test that chunking preserves code blocks."""
        from datus.storage.document.chunker.semantic_chunker import ChunkingConfig, SemanticChunker

        config = ChunkingConfig(chunk_size=100, preserve_code_blocks=True)
        chunker = SemanticChunker(config=config)

        code_content = """Here is code:

```python
def very_long_function():
    # This is a long function
    x = 1
    y = 2
    z = 3
    return x + y + z
```

End of section."""

        parsed = ParsedDocument(
            title="Code Test",
            sections=[
                ParsedSection(level=1, title="Code", content=code_content, children=[]),
            ],
        )

        metadata = {
            "platform": "test",
            "version": "v1",
            "source_type": "github",
            "source_url": "http://test",
            "doc_path": "test.md",
        }

        chunks = chunker.chunk(parsed, metadata)

        # Find the chunk with the code block
        code_chunks = [c for c in chunks if "```python" in c.chunk_text]
        assert len(code_chunks) > 0

        # Verify code block is complete (has both opening and closing)
        for chunk in code_chunks:
            if "```python" in chunk.chunk_text:
                assert chunk.chunk_text.count("```") >= 2


# =============================================================================
# Cleaner Tests
# =============================================================================


class TestDocumentCleaner:
    """Test document cleaner."""

    def test_clean_text(self):
        """Test basic text cleaning."""
        from datus.storage.document.cleaner.doc_cleaner import DocumentCleaner

        cleaner = DocumentCleaner()

        # Test control character removal
        text = "Hello\x00World\x07!"
        cleaned = cleaner.clean_text(text)
        assert "\x00" not in cleaned
        assert "\x07" not in cleaned

    def test_clean_preserves_code_blocks(self):
        """Test that cleaning preserves code blocks."""
        from datus.storage.document.cleaner.doc_cleaner import DocumentCleaner

        cleaner = DocumentCleaner(preserve_code_blocks=True)

        text = """Some text.

```python
def   hello():
    pass
```

More text."""

        cleaned = cleaner.clean_text(text)
        assert "```python" in cleaned
        assert "def   hello" in cleaned  # Spaces in code preserved


# =============================================================================
# Metadata Extractor Tests
# =============================================================================


class TestMetadataExtractor:
    """Test metadata extractor."""

    def test_extract_keywords(self):
        """Test keyword extraction."""
        from datus.storage.document.parser.metadata_extractor import MetadataExtractor

        extractor = MetadataExtractor()

        text = """
        This document describes how to CREATE TABLE in Snowflake.
        You can use SELECT to query data and JOIN tables together.
        The PRIMARY KEY constraint ensures uniqueness.
        Use CREATE TABLE to create a new table. SELECT is also common.
        """

        keywords = extractor.extract_keywords(text, platform="snowflake")

        assert len(keywords) > 0
        # Should find SQL keywords or compound terms
        keyword_str = " ".join(keywords).lower()
        # Check for either single keywords or compound terms
        assert any(term in keyword_str for term in ["create", "select", "join", "table", "primary_key", "create_table"])

    def test_detect_language_english(self):
        """Test English language detection."""
        from datus.storage.document.parser.metadata_extractor import MetadataExtractor

        extractor = MetadataExtractor()

        text = "This is a test document in English."
        lang = extractor._detect_language(text)
        assert lang == "en"

    def test_detect_language_chinese(self):
        """Test Chinese language detection."""
        from datus.storage.document.parser.metadata_extractor import MetadataExtractor

        extractor = MetadataExtractor()

        text = "这是一个中文测试文档，包含很多中文字符。"
        lang = extractor._detect_language(text)
        assert lang == "zh"


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestRateLimiter:
    """Test rate limiter."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes correctly."""
        from datus.storage.document.fetcher.rate_limiter import RateLimiter

        limiter = RateLimiter()

        # Should have default configs
        assert "api.github.com" in limiter._configs
        assert "default" in limiter._configs

    def test_configure_github_authenticated(self):
        """Test GitHub authenticated configuration."""
        from datus.storage.document.fetcher.rate_limiter import RateLimiter

        limiter = RateLimiter()
        limiter.configure_github_authenticated(requests_per_hour=5000)

        config = limiter._configs["api.github.com"]
        assert config.requests_per_hour == 5000


# =============================================================================
# SearchTool Tests
# =============================================================================


class TestSearchTool:
    """Test SearchTool methods: list_document_nav, get_document, search_document.

    Uses the benchmark documents to populate a real store, then exercises
    all three SearchTool methods against it.
    """

    # Default platform for tests — must match the pre-populated store data.
    TEST_PLATFORM = "starrocks"

    @pytest.fixture(autouse=True)
    def setup_store(self, agent_config: AgentConfig):
        """Import documents into the store once for all tests in this class."""
        self.agent_config = agent_config
        self.store = document_store(agent_config.document_storage_path(self.TEST_PLATFORM))

        from datus.tools.search_tools.search_tool import SearchTool

        self.tool = SearchTool(agent_config=agent_config)
        # Point the tool to the already-populated store
        self.tool._document_store = self.store

    # ----- list_document_nav + get_document (dependent) -----

    @pytest.mark.parametrize("platform,version", [("starrocks", "")])
    def test_list_document_nav_returns_tree(self, platform: str, version: str):
        result = self.tool.list_document_nav(platform=platform, version=version)
        assert result.success, f"list_document_nav failed: {result.error}"
        assert result.platform == platform
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

    @pytest.mark.parametrize("platform,version", [("starrocks", "")])
    def test_get_document_by_title_from_nav(self, platform: str, version: str):
        """get_document should retrieve chunks when given a title from the nav tree."""
        # First get the nav tree to find a real title
        nav_result = self.tool.list_document_nav(platform=platform, version=version)
        assert nav_result.success and len(nav_result.nav_tree) > 0

        # Find a leaf node (node with empty children) from the pure tree
        # Handles both single-version (name/children) and multi-version (version/tree) formats
        def _find_first_doc(nodes):
            for node in nodes:
                # Multi-version format: {"version": ..., "tree": [...]}
                if "tree" in node:
                    found = _find_first_doc(node["tree"])
                    if found:
                        return found
                    continue
                # Leaf node: has name but empty children
                if not node.get("children"):
                    return node.get("name")
                found = _find_first_doc(node["children"])
                if found:
                    return found
            return None

        title = _find_first_doc(nav_result.nav_tree)
        assert title, "Should find at least one document title in the nav tree"

        # Use the title to get document chunks
        result = self.tool.get_document(platform=platform, titles=[title], version=version)

        assert result.success, f"get_document failed: {result.error}"
        assert result.platform == platform
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

    @pytest.mark.parametrize("platform,version,keywords", [("starrocks", "", ["DATE_TRUNC"])])
    def test_search_document_returns_results(self, platform: str, version: str, keywords: List[str]):
        """search_document should find results for a relevant keyword."""
        result = self.tool.search_document(
            platform=platform,
            keywords=keywords,
            version=version if version else None,
            top_n=3,
        )

        assert result.success, f"search_document failed: {result.error}"
        assert result.doc_count > 0, f"Should find at least one result for keywords {keywords}"

        # Verify each keyword has results in the docs dict
        for keyword in keywords:
            assert keyword in result.docs, f"Keyword '{keyword}' missing from result.docs"
            assert len(result.docs[keyword]) > 0, f"No results for keyword '{keyword}'"

        # Verify chunk fields from the first keyword's results
        first_chunk = result.docs[keywords[0]][0]
        assert "chunk_text" in first_chunk
        assert "title" in first_chunk
        assert "hierarchy" in first_chunk
        assert "doc_path" in first_chunk

    # ----- delete_docs -----

    def test_delete_docs_by_version(self):
        """delete_docs(version=...) should remove only chunks of that version."""
        # Create a temporary store to avoid polluting the shared fixture
        import shutil
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_store = document_store(tmp_dir)

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
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_delete_docs_all(self):
        """delete_docs(version=None) should remove all chunks (drop + recreate)."""
        import shutil
        import tempfile

        from datus.storage.document.schemas import PlatformDocChunk

        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_store = document_store(tmp_dir)

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
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_delete_docs_empty_store(self):
        """delete_docs on an empty store should return 0."""
        import shutil
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_store = document_store(tmp_dir)
            deleted = tmp_store.delete_docs(version="v1")
            assert deleted == 0
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_search_document_wrong_platform(self):
        """search_document on a non-existent platform returns empty results."""
        result = self.tool.search_document(
            platform="nonexistent_platform_xyz",
            keywords=["anything"],
            top_n=3,
        )

        assert result.success
        assert result.doc_count == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests (require network access)."""

    @pytest.mark.skip(reason="Requires GitHub token")
    def test_github_fetcher_integration(self):
        """Test GitHub fetcher with real API."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetcher

        fetcher = GitHubFetcher(platform="test")
        info = fetcher.get_repo_info("duckdb/duckdb")

        assert info["full_name"] == "duckdb/duckdb"
        assert "description" in info

    @pytest.mark.skip(reason="Requires network access to docs.snowflake.com")
    def test_web_fetcher_integration(self):
        """Full integration test: Fetch → Parse → Clean → Chunk.

        Tests the complete documentation processing pipeline:
        1. Fetch HTML documents from Snowflake docs
        2. Parse HTML to extract structured content
        3. Clean the parsed content
        4. Chunk into embedding-ready PlatformDocChunk objects

        Note: Using max_depth=0 to limit crawling scope.
        """
        from datus.storage.document.chunker.semantic_chunker import ChunkingConfig, SemanticChunker
        from datus.storage.document.cleaner.doc_cleaner import DocumentCleaner
        from datus.storage.document.fetcher.web_fetcher import WebFetcher
        from datus.storage.document.parser.html_parser import HTMLParser

        # ========== Step 1: Fetch ==========
        print("\n" + "=" * 60)
        print("Step 1: FETCH - Fetching documents from Snowflake docs...")
        print("=" * 60)

        fetcher = WebFetcher(platform="snowflake", version="latest")
        fetched_docs = fetcher.fetch(
            "https://docs.snowflake.com/en/user-guide/data-load-snowpipe-intro",
            max_depth=0,
        )

        print(f"✓ Fetched {len(fetched_docs)} document(s)")
        for i, doc in enumerate(fetched_docs[:3]):
            print(f"  [{i+1}] {doc.doc_path} ({len(doc.raw_content)} bytes)")

        assert len(fetched_docs) >= 1
        assert fetched_docs[0].platform == "snowflake"

        # ========== Step 2: Parse ==========
        print("\n" + "=" * 60)
        print("Step 2: PARSE - Parsing HTML documents...")
        print("=" * 60)

        parser = HTMLParser()
        parsed_docs = []

        for doc in fetched_docs:
            try:
                parsed = parser.parse(doc)
                parsed_docs.append((doc, parsed))
                print(f"✓ Parsed: {parsed.title or doc.doc_path}")
                print(f"    Sections: {len(parsed.sections)}")
            except Exception as e:
                print(f"✗ Failed to parse {doc.doc_path}: {e}")

        assert len(parsed_docs) >= 1
        print(f"✓ Successfully parsed {len(parsed_docs)} document(s)")

        # ========== Step 3: Clean ==========
        print("\n" + "=" * 60)
        print("Step 3: CLEAN - Cleaning parsed content...")
        print("=" * 60)

        cleaner = DocumentCleaner(preserve_code_blocks=True)

        for _doc, parsed in parsed_docs:
            for section in parsed.sections:
                section.content = cleaner.clean_text(section.content)
                for child in section.children:
                    child.content = cleaner.clean_text(child.content)

        print(f"✓ Cleaned {len(parsed_docs)} document(s)")

        # ========== Step 4: Chunk ==========
        print("\n" + "=" * 60)
        print("Step 4: CHUNK - Creating embedding-ready chunks...")
        print("=" * 60)

        config = ChunkingConfig(
            chunk_size=512,
            chunk_overlap=50,
            preserve_code_blocks=True,
        )
        chunker = SemanticChunker(config=config)

        all_chunks: list[PlatformDocChunk] = []

        for doc, parsed in parsed_docs:
            metadata = {
                "platform": doc.platform,
                "version": doc.version,
                "source_type": doc.source_type,
                "source_url": doc.source_url,
                "doc_path": doc.doc_path,
            }
            chunks = chunker.chunk(parsed, metadata)
            all_chunks.extend(chunks)
            print(f"✓ {doc.doc_path}: {len(chunks)} chunks")

        print(f"\n✓ Total chunks created: {len(all_chunks)}")

        # ========== Validate Results ==========
        assert len(all_chunks) > 0
        assert all(isinstance(c, PlatformDocChunk) for c in all_chunks)

        # Print sample chunks
        print("\n" + "=" * 60)
        print("RESULTS: Sample PlatformDocChunk objects")
        print("=" * 60)

        for i, chunk in enumerate(all_chunks[:3]):
            print(f"\n--- Chunk {i+1}/{len(all_chunks)} ---")
            print(f"  chunk_id:    {chunk.chunk_id}")
            print(f"  version:     {chunk.version}")
            print(f"  title:       {chunk.title}")
            print(f"  titles:      {chunk.titles}")
            print(f"  nav_path:    {chunk.nav_path}")
            print(f"  group_name:  {chunk.group_name}")
            print(f"  hierarchy:   {chunk.hierarchy}")
            print(f"  keywords:    {chunk.keywords}")
            print(f"  language:    {chunk.language}")
            print(f"  chunk_text:  ({len(chunk.chunk_text)} chars)")

        # ========== Summary ==========
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Documents fetched:  {len(fetched_docs)}")
        print(f"  Documents parsed:   {len(parsed_docs)}")
        print(f"  Total chunks:       {len(all_chunks)}")
        print(f"  Avg chunk length:   {sum(len(c.chunk_text) for c in all_chunks) // len(all_chunks)} chars")
        print("=" * 60)

        # Final assertions
        for chunk in all_chunks:
            assert chunk.chunk_id, "chunk_id should not be empty"
            assert chunk.chunk_text, "chunk_text should not be empty"
            assert chunk.version, "version should not be empty"
            assert chunk.titles is not None, "titles should not be None"
            assert chunk.nav_path is not None, "nav_path should not be None"
            assert chunk.group_name is not None, "group_name should not be None"
            assert chunk.hierarchy is not None, "hierarchy should not be None"
