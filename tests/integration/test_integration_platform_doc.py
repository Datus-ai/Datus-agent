# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration Tests for Platform Document Pipeline

Covers the complete document processing pipeline for all source types:
- Local: Markdown, HTML, RST files
- GitHub: Repository documentation (requires GITHUB_TOKEN)
- Website: Web crawling (requires network access)

Pipeline stages tested:
  Fetch → Clean → Parse → Chunk → Store → Search
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from datus.configuration.agent_config import AgentConfig, DocumentConfig
from datus.storage.document.doc_init import init_platform_docs
from datus.storage.document.schemas import (
    CONTENT_TYPE_HTML,
    CONTENT_TYPE_MARKDOWN,
    CONTENT_TYPE_RST,
    SOURCE_TYPE_WEBSITE,
    PlatformDocChunk,
)
from datus.storage.document.store import document_store
from datus.storage.document.streaming_processor import StreamingDocProcessor
from datus.tools.search_tools.search_tool import SearchTool
from datus.utils.loggings import get_logger
from tests.conftest import load_acceptance_config

logger = get_logger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def agent_config() -> AgentConfig:
    """Load agent configuration."""
    return load_acceptance_config()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    tmp = tempfile.mkdtemp(prefix="datus_doc_test")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def local_docs_dir(temp_dir):
    """Create a temporary directory with sample documentation files."""
    docs_dir = Path(temp_dir) / "docs"
    docs_dir.mkdir(parents=True)

    # Create Markdown file
    md_file = docs_dir / "guide.md"
    md_file.write_text(
        """# User Guide

Welcome to the user guide.

## Getting Started

This section explains how to get started.

### Installation

Run the following command:

```bash
pip install datus
```

### Configuration

Create a configuration file:

```yaml
agent:
  target: openai
```

## Advanced Usage

For advanced users, see the following sections.

### Custom Pipelines

You can create custom pipelines using the API.

### Performance Tuning

Optimize performance by adjusting batch sizes.
"""
    )

    # Create HTML file
    html_file = docs_dir / "api.html"
    html_file.write_text(
        """<!DOCTYPE html>
<html>
<head>
    <title>API Reference</title>
</head>
<body>
    <h1>API Reference</h1>
    <p>This document describes the REST API.</p>

    <h2>Authentication</h2>
    <p>All API requests require authentication using Bearer tokens.</p>

    <h3>Getting a Token</h3>
    <p>Use the /auth/token endpoint to get a token.</p>

    <h2>Endpoints</h2>

    <h3>GET /api/users</h3>
    <p>Returns a list of users.</p>

    <h3>POST /api/users</h3>
    <p>Creates a new user.</p>

    <h2>Error Handling</h2>
    <p>All errors return a JSON object with an error message.</p>
</body>
</html>
"""
    )

    # Create RST file
    rst_file = docs_dir / "changelog.rst"
    rst_file.write_text(
        """=========
Changelog
=========

Version 2.0.0
=============

Major release with breaking changes.

New Features
------------

* Added support for multiple databases
* Improved performance by 50%

Bug Fixes
---------

* Fixed memory leak in connection pool
* Fixed SQL injection vulnerability

Version 1.0.0
=============

Initial release.

Features
--------

* Basic SQL generation
* Schema linking
* Chat interface
"""
    )

    # Create subdirectory with nested docs
    sub_dir = docs_dir / "advanced"
    sub_dir.mkdir()

    nested_md = sub_dir / "plugins.md"
    nested_md.write_text(
        """# Plugin System

The plugin system allows you to extend functionality.

## Creating a Plugin

Plugins are Python modules that implement the `Plugin` interface.

```python
from datus.plugins import Plugin

class MyPlugin(Plugin):
    def execute(self, context):
        return context.data
```

## Registering Plugins

Register plugins in the configuration file.
"""
    )

    return docs_dir


# =============================================================================
# Version Detection Tests
# =============================================================================


class TestVersionDetection:
    """Test version detection from paths."""

    def test_detect_versions_from_paths_basic(self):
        """Test version detection from simple version directories."""
        from datus.storage.document.doc_init import _detect_versions_from_paths

        # Basic version paths
        paths = ["1.3.0", "1.2.0"]
        versions = _detect_versions_from_paths(paths)
        assert versions == {"1.3.0", "1.2.0"}

    def test_detect_versions_from_paths_with_v_prefix(self):
        """Test version detection with 'v' prefix."""
        from datus.storage.document.doc_init import _detect_versions_from_paths

        paths = ["v1.0.0", "v2.0.0"]
        versions = _detect_versions_from_paths(paths)
        assert versions == {"1.0.0", "2.0.0"}

    def test_detect_versions_from_paths_semver(self):
        """Test version detection with pre-release tags."""
        from datus.storage.document.doc_init import _detect_versions_from_paths

        paths = ["1.0.0-beta", "1.0.0-rc1", "1.0.0"]
        versions = _detect_versions_from_paths(paths)
        assert versions == {"1.0.0-beta", "1.0.0-rc1", "1.0.0"}

    def test_detect_versions_from_paths_not_versions(self):
        """Test that non-version paths return empty set."""
        from datus.storage.document.doc_init import _detect_versions_from_paths

        # Regular paths
        paths = ["docs", "README.md", "CHANGELOG.md"]
        versions = _detect_versions_from_paths(paths)
        assert versions == set()

    def test_detect_versions_from_paths_mixed(self):
        """Test that mixed paths (version + non-version) return empty set."""
        from datus.storage.document.doc_init import _detect_versions_from_paths

        # Mixed - only some paths are versions
        paths = ["1.3.0", "docs", "README.md"]
        versions = _detect_versions_from_paths(paths)
        assert versions == set()

    def test_detect_versions_from_paths_empty(self):
        """Test empty paths returns empty set."""
        from datus.storage.document.doc_init import _detect_versions_from_paths

        versions = _detect_versions_from_paths([])
        assert versions == set()


# =============================================================================
# Local Fetcher Integration Tests
# =============================================================================


class TestLocalFetcherIntegration:
    """Integration tests for LocalFetcher with different content types."""

    def test_fetch_markdown_file(self, local_docs_dir):
        """Test fetching and parsing a Markdown file."""
        from datus.storage.document.fetcher.local_fetcher import LocalFetcher

        fetcher = LocalFetcher(platform="test", version="v1.0")
        doc = fetcher.fetch_single(str(local_docs_dir / "guide.md"))

        assert doc is not None
        assert doc.platform == "test"
        assert doc.version == "v1.0"
        assert doc.content_type == CONTENT_TYPE_MARKDOWN
        assert "# User Guide" in doc.raw_content
        assert doc.doc_path.endswith("guide.md")

    def test_fetch_html_file(self, local_docs_dir):
        """Test fetching and parsing an HTML file."""
        from datus.storage.document.fetcher.local_fetcher import LocalFetcher

        fetcher = LocalFetcher(platform="test", version="v1.0")
        doc = fetcher.fetch_single(str(local_docs_dir / "api.html"))

        assert doc is not None
        assert doc.content_type == CONTENT_TYPE_HTML
        assert "<h1>API Reference</h1>" in doc.raw_content
        assert doc.metadata.get("title") == "API Reference"

    def test_fetch_rst_file(self, local_docs_dir):
        """Test fetching and parsing an RST file."""
        from datus.storage.document.fetcher.local_fetcher import LocalFetcher

        fetcher = LocalFetcher(platform="test", version="v1.0")
        doc = fetcher.fetch_single(str(local_docs_dir / "changelog.rst"))

        assert doc is not None
        assert doc.content_type == CONTENT_TYPE_RST
        assert "Version 2.0.0" in doc.raw_content
        # RST title extraction should work
        assert doc.metadata.get("title") == "Changelog"

    def test_fetch_directory_recursive(self, local_docs_dir):
        """Test fetching all documents from a directory recursively."""
        from datus.storage.document.fetcher.local_fetcher import LocalFetcher

        fetcher = LocalFetcher(platform="test", version="v1.0")
        docs = fetcher.fetch(source=str(local_docs_dir), recursive=True)

        # Should find: guide.md, api.html, changelog.rst, advanced/plugins.md
        assert len(docs) == 4

        # Verify content types
        content_types = {doc.content_type for doc in docs}
        assert CONTENT_TYPE_MARKDOWN in content_types
        assert CONTENT_TYPE_HTML in content_types
        assert CONTENT_TYPE_RST in content_types

    def test_fetch_with_include_patterns(self, local_docs_dir):
        """Test fetching with include patterns (only Markdown)."""
        from datus.storage.document.fetcher.local_fetcher import LocalFetcher

        fetcher = LocalFetcher(platform="test", version="v1.0")
        docs = fetcher.fetch(
            source=str(local_docs_dir),
            recursive=True,
            include_patterns=["*.md"],
        )

        # Should find only: guide.md, advanced/plugins.md
        assert len(docs) == 2
        assert all(doc.content_type == CONTENT_TYPE_MARKDOWN for doc in docs)

    def test_fetch_with_exclude_patterns(self, local_docs_dir):
        """Test fetching with exclude patterns."""
        from datus.storage.document.fetcher.local_fetcher import LocalFetcher

        fetcher = LocalFetcher(platform="test", version="v1.0")
        docs = fetcher.fetch(
            source=str(local_docs_dir),
            recursive=True,
            exclude_patterns=["changelog.*", "advanced/*"],
        )

        # Should exclude changelog.rst and advanced/plugins.md
        assert len(docs) == 2
        doc_names = {Path(doc.doc_path).name for doc in docs}
        assert "changelog.rst" not in doc_names
        assert "plugins.md" not in doc_names


# =============================================================================
# Parser Integration Tests
# =============================================================================


class TestParserIntegration:
    """Integration tests for document parsers."""

    def test_markdown_parser_full_document(self, local_docs_dir):
        """Test Markdown parser with a full document."""
        from datus.storage.document.fetcher.local_fetcher import LocalFetcher
        from datus.storage.document.parser.markdown_parser import MarkdownParser

        fetcher = LocalFetcher(platform="test", version="v1.0")
        doc = fetcher.fetch_single(str(local_docs_dir / "guide.md"))

        parser = MarkdownParser()
        parsed = parser.parse(doc)

        assert parsed.title == "User Guide"
        assert len(parsed.sections) > 0

        # Check section structure (recursively collect all titles)
        def collect_titles(sections):
            titles = []
            for section in sections:
                titles.append(section.title)
                titles.extend(collect_titles(section.children))
            return titles

        section_titles = collect_titles(parsed.sections)

        assert "Getting Started" in section_titles
        assert "Installation" in section_titles
        assert "Advanced Usage" in section_titles

    def test_html_parser_full_document(self, local_docs_dir):
        """Test HTML parser with a full document."""
        from datus.storage.document.fetcher.local_fetcher import LocalFetcher
        from datus.storage.document.parser.html_parser import HTMLParser

        fetcher = LocalFetcher(platform="test", version="v1.0")
        doc = fetcher.fetch_single(str(local_docs_dir / "api.html"))

        parser = HTMLParser()
        parsed = parser.parse(doc)

        assert parsed.title == "API Reference"
        assert len(parsed.sections) > 0

    def test_rst_parsed_as_markdown(self, local_docs_dir):
        """Test that RST files are parsed using Markdown parser."""
        from datus.storage.document.cleaner.doc_cleaner import DocumentCleaner
        from datus.storage.document.fetcher.local_fetcher import LocalFetcher
        from datus.storage.document.parser.markdown_parser import MarkdownParser

        fetcher = LocalFetcher(platform="test", version="v1.0")
        doc = fetcher.fetch_single(str(local_docs_dir / "changelog.rst"))

        # Clean first (as done in pipeline)
        cleaner = DocumentCleaner()
        cleaned = cleaner.clean(doc)

        # RST should be parsed with Markdown parser
        parser = MarkdownParser()
        parsed = parser.parse(cleaned)

        # Should extract some structure
        assert parsed.title is not None
        assert len(parsed.sections) > 0


# =============================================================================
# Chunker Integration Tests
# =============================================================================


class TestChunkerIntegration:
    """Integration tests for semantic chunker."""

    def test_chunk_parsed_document(self, local_docs_dir):
        """Test chunking a parsed document."""
        from datus.storage.document.chunker.semantic_chunker import ChunkingConfig, SemanticChunker
        from datus.storage.document.cleaner.doc_cleaner import DocumentCleaner
        from datus.storage.document.fetcher.local_fetcher import LocalFetcher
        from datus.storage.document.parser.markdown_parser import MarkdownParser

        # Fetch and parse
        fetcher = LocalFetcher(platform="test", version="v1.0")
        doc = fetcher.fetch_single(str(local_docs_dir / "guide.md"))

        cleaner = DocumentCleaner()
        cleaned = cleaner.clean(doc)

        parser = MarkdownParser()
        parsed = parser.parse(cleaned)

        # Chunk
        config = ChunkingConfig(chunk_size=256, preserve_code_blocks=True)
        chunker = SemanticChunker(config=config)

        metadata = {
            "platform": doc.platform,
            "version": doc.version,
            "source_type": doc.source_type,
            "source_url": doc.source_url,
            "doc_path": doc.doc_path,
        }

        chunks = chunker.chunk(parsed, metadata)

        assert len(chunks) > 0
        assert all(isinstance(c, PlatformDocChunk) for c in chunks)

        # Verify chunk properties
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.chunk_text
            assert chunk.version == "v1.0"
            assert chunk.title

        # Verify code blocks are preserved
        code_chunks = [c for c in chunks if "```" in c.chunk_text]
        assert len(code_chunks) > 0


# =============================================================================
# Full Pipeline Integration Tests (init_platform_docs)
# =============================================================================


class TestFullPipelineIntegration:
    """Integration tests for the complete init_platform_docs pipeline."""

    def test_init_platform_docs_local_overwrite(self, local_docs_dir, temp_dir):
        """Test full pipeline with local source in overwrite mode."""
        db_path = str(Path(temp_dir) / "store")

        cfg = DocumentConfig(
            type="local",
            source=str(local_docs_dir),
            version="v1.0",
            chunk_size=256,
        )

        result = init_platform_docs(
            db_path=db_path,
            platform="test_local",
            cfg=cfg,
            build_mode="overwrite",
            pool_size=2,
        )

        assert result.success, f"Pipeline failed: {result.errors}"
        assert result.platform == "test_local"
        assert result.version == "v1.0"
        assert result.total_docs == 4  # guide.md, api.html, changelog.rst, plugins.md
        assert result.total_chunks > 0
        assert result.duration_seconds > 0

    def test_init_platform_docs_check_mode(self, local_docs_dir, temp_dir):
        """Test check mode returns existing stats without fetching."""
        db_path = str(Path(temp_dir) / "store")

        cfg = DocumentConfig(
            type="local",
            source=str(local_docs_dir),
            version="v1.0",
        )

        # First, populate the store
        init_platform_docs(
            db_path=db_path,
            platform="test_local",
            cfg=cfg,
            build_mode="overwrite",
        )

        # Now check mode
        result = init_platform_docs(
            db_path=db_path,
            platform="test_local",
            cfg=cfg,
            build_mode="check",
        )

        assert result.success
        assert result.total_chunks > 0

    def test_init_platform_docs_with_include_patterns(self, local_docs_dir, temp_dir):
        """Test pipeline with include patterns (only Markdown)."""
        db_path = str(Path(temp_dir) / "store")

        cfg = DocumentConfig(
            type="local",
            source=str(local_docs_dir),
            version="v1.0",
            include_patterns=["*.md"],
        )

        result = init_platform_docs(
            db_path=db_path,
            platform="test_local",
            cfg=cfg,
            build_mode="overwrite",
        )

        assert result.success
        assert result.total_docs == 2  # Only guide.md and plugins.md

    def test_init_platform_docs_empty_source(self, temp_dir):
        """Test pipeline with empty source directory."""
        db_path = str(Path(temp_dir) / "store")
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()

        cfg = DocumentConfig(
            type="local",
            source=str(empty_dir),
            version="v1.0",
        )

        result = init_platform_docs(
            db_path=db_path,
            platform="test_empty",
            cfg=cfg,
            build_mode="overwrite",
        )

        assert result.success
        assert result.total_docs == 0
        assert "No documents found" in result.errors[0]


# =============================================================================
# SearchTool Integration Tests
# =============================================================================


class TestSearchToolIntegration:
    """Integration tests for SearchTool after storing documents."""

    @pytest.fixture
    def populated_store(self, local_docs_dir, temp_dir, agent_config):
        """Create a populated store for search tests."""
        db_path = str(Path(temp_dir) / "store")

        cfg = DocumentConfig(
            type="local",
            source=str(local_docs_dir),
            version="v1.0",
            chunk_size=256,
        )

        result = init_platform_docs(
            db_path=db_path,
            platform="test_search",
            cfg=cfg,
            build_mode="overwrite",
        )
        assert result.success

        # Create a modified agent_config that uses our temp store
        class TestAgentConfig:
            def document_storage_path(self, platform):
                return db_path

        return TestAgentConfig(), db_path

    def test_list_document_nav(self, populated_store):
        """Test list_document_nav returns navigation tree."""
        test_config, _ = populated_store
        tool = SearchTool(agent_config=test_config)

        result = tool.list_document_nav(platform="test_search")

        assert result.success, f"list_document_nav failed: {result.error}"
        assert result.platform == "test_search"
        assert result.total_docs > 0
        assert len(result.nav_tree) > 0

        # Verify tree structure
        for item in result.nav_tree:
            assert "name" in item or "version" in item

    def test_get_document_by_title(self, populated_store):
        """Test get_document retrieves document chunks."""
        test_config, _ = populated_store
        tool = SearchTool(agent_config=test_config)

        # First get nav to find a title
        _nav_result = tool.list_document_nav(platform="test_search")
        assert _nav_result.success

        # Find a leaf node (document title)
        title = _find_nav_leaf(_nav_result.nav_tree)
        assert title, "Should find at least one document title"

        # Get document by title
        result = tool.get_document(platform="test_search", titles=[title])

        assert result.success, f"get_document failed: {result.error}"
        assert result.chunk_count > 0
        assert len(result.chunks) == result.chunk_count

    def test_search_document_semantic(self, populated_store):
        """Test search_document finds relevant documents."""
        test_config, _ = populated_store
        tool = SearchTool(agent_config=test_config)

        # Search for content we know exists
        result = tool.search_document(
            platform="test_search",
            keywords=["installation", "plugin"],
            top_n=3,
        )

        assert result.success, f"search_document failed: {result.error}"
        assert result.doc_count > 0

        # Verify results for each keyword
        for keyword in ["installation", "plugin"]:
            assert keyword in result.docs
            assert len(result.docs[keyword]) > 0

    def test_search_no_results(self, populated_store):
        """Test search with non-matching keywords."""
        test_config, _ = populated_store
        tool = SearchTool(agent_config=test_config)

        result = tool.search_document(
            platform="test_search",
            keywords=["xyznonexistent123"],
            top_n=3,
        )

        assert result.success
        # May return 0 or low-relevance results depending on embedding model

    def test_list_nav_nonexistent_platform(self, agent_config):
        """Test list_document_nav for non-existent platform."""
        tool = SearchTool(agent_config=agent_config)

        result = tool.list_document_nav(platform="nonexistent_xyz_platform")

        assert result.success
        assert result.total_docs == 0
        assert result.nav_tree == []


# =============================================================================
# Store Operations Integration Tests
# =============================================================================


class TestStoreOperationsIntegration:
    """Integration tests for DocumentStore operations."""

    def test_store_and_retrieve_chunks(self, temp_dir):
        """Test storing and retrieving chunks."""
        store = document_store(temp_dir)

        chunks = [
            PlatformDocChunk(
                chunk_id=PlatformDocChunk.generate_chunk_id("doc.md", i, "v1"),
                chunk_text=f"Test content {i}",
                chunk_index=i,
                title="Test Document",
                titles=["Test Document"],
                nav_path=["Guides"],
                group_name="Guides",
                hierarchy="Guides > Test Document",
                version="v1",
                source_type="local",
                source_url="",
                doc_path="doc.md",
            )
            for i in range(5)
        ]

        store.store_chunks(chunks)

        # Verify storage
        assert store.table.count_rows() == 5

        # Retrieve all rows
        rows = store.get_all_rows(select_fields=["chunk_id", "chunk_text", "version"])
        assert len(rows) == 5

    def test_store_search_with_version_filter(self, temp_dir):
        """Test search with version filtering."""
        store = document_store(temp_dir)

        # Store chunks for two versions
        for version in ["v1", "v2"]:
            chunks = [
                PlatformDocChunk(
                    chunk_id=PlatformDocChunk.generate_chunk_id(f"{version}.md", i, version),
                    chunk_text=f"Content for {version} chunk {i}",
                    chunk_index=i,
                    title=f"Doc {version}",
                    titles=[f"Doc {version}"],
                    nav_path=[],
                    group_name="",
                    hierarchy=f"Doc {version}",
                    version=version,
                    source_type="local",
                    source_url="",
                    doc_path=f"{version}.md",
                )
                for i in range(3)
            ]
            store.store_chunks(chunks)

        assert store.table.count_rows() == 6

        # Search with version filter
        results = store.search_docs(query="Content", version="v1", top_n=10)
        assert len(results) == 3
        assert all(r["version"] == "v1" for r in results)

    def test_store_delete_by_version(self, temp_dir):
        """Test deleting chunks by version."""
        store = document_store(temp_dir)

        # Store chunks for two versions
        for version in ["v1", "v2"]:
            chunks = [
                PlatformDocChunk(
                    chunk_id=PlatformDocChunk.generate_chunk_id(f"{version}.md", i, version),
                    chunk_text=f"Content {version}",
                    chunk_index=i,
                    title=f"Doc {version}",
                    titles=[f"Doc {version}"],
                    nav_path=[],
                    group_name="",
                    hierarchy=f"Doc {version}",
                    version=version,
                    source_type="local",
                    source_url="",
                    doc_path=f"{version}.md",
                )
                for i in range(3)
            ]
            store.store_chunks(chunks)

        # Delete v1
        deleted = store.delete_docs(version="v1")
        assert deleted == 3

        # Only v2 remains
        assert store.table.count_rows() == 3

    def test_store_list_versions(self, temp_dir):
        """Test listing available versions."""
        store = document_store(temp_dir)

        # Store chunks for multiple versions
        for version in ["v1.0", "v2.0", "v3.0"]:
            chunk = PlatformDocChunk(
                chunk_id=PlatformDocChunk.generate_chunk_id("doc.md", 0, version),
                chunk_text=f"Content {version}",
                chunk_index=0,
                title="Doc",
                titles=["Doc"],
                nav_path=[],
                group_name="",
                hierarchy="Doc",
                version=version,
                source_type="local",
                source_url="",
                doc_path="doc.md",
            )
            store.store_chunks([chunk])

        versions = store.list_versions()
        version_names = [v["version"] for v in versions]

        assert "v1.0" in version_names
        assert "v2.0" in version_names
        assert "v3.0" in version_names


# =============================================================================
# Streaming Processor Integration Tests
# =============================================================================


class TestStreamingProcessorIntegration:
    """Integration tests for StreamingDocProcessor."""

    def test_streaming_processor_local(self, local_docs_dir, temp_dir):
        """Test streaming processor with local documents."""
        from datus.storage.document.fetcher.local_fetcher import LocalFetcher

        db_path = str(Path(temp_dir) / "store")
        store = document_store(db_path)

        # Fetch documents
        fetcher = LocalFetcher(platform="test_streaming", version="v1.0")
        documents = fetcher.fetch(source=str(local_docs_dir), recursive=True)

        assert len(documents) == 4

        # Process with streaming processor
        processor = StreamingDocProcessor(
            store=store,
            chunk_size=256,
            pool_size=2,
        )

        stats = processor.process_local(
            fetcher=fetcher,
            documents=documents,
            version="v1.0",
            platform="test_streaming",
        )

        assert stats.total_docs == 4
        assert stats.total_chunks > 0
        assert len(stats.errors) == 0

        # Verify data in store
        assert store.table.count_rows() == stats.total_chunks

    def test_streaming_processor_progress_tracking(self, local_docs_dir, temp_dir):
        """Test that streaming processor tracks progress correctly."""
        from datus.storage.document.fetcher.local_fetcher import LocalFetcher

        db_path = str(Path(temp_dir) / "store")
        store = document_store(db_path)

        fetcher = LocalFetcher(platform="test_progress", version="v1.0")
        documents = fetcher.fetch(source=str(local_docs_dir), recursive=True)

        processor = StreamingDocProcessor(
            store=store,
            chunk_size=512,
            pool_size=4,
        )

        stats = processor.process_local(
            fetcher=fetcher,
            documents=documents,
            version="v1.0",
            platform="test_progress",
        )

        # Verify stats are consistent
        assert stats.total_docs == len(documents)
        assert stats.duration_seconds > 0


# =============================================================================
# GitHub Fetcher Integration Tests (Requires Token)
# =============================================================================


@pytest.mark.skipif(
    not os.environ.get("GITHUB_TOKEN"),
    reason="Requires GITHUB_TOKEN environment variable",
)
class TestGitHubFetcherIntegration:
    """Integration tests for GitHub fetcher (requires network and token)."""

    def test_github_fetch_repo_info(self):
        """Test fetching repository info from apache/polaris."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetcher

        fetcher = GitHubFetcher(platform="polaris", token=os.environ.get("GITHUB_TOKEN"))
        info = fetcher.get_repo_info("apache/polaris")

        assert info["full_name"] == "apache/polaris"
        assert "description" in info
        assert "default_branch" in info

    def test_github_collect_metadata(self):
        """Test collecting metadata from apache/polaris versioned-docs branch."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetcher

        fetcher = GitHubFetcher(
            platform="polaris",
            token=os.environ.get("GITHUB_TOKEN"),
            github_ref="versioned-docs",
        )

        metadata = fetcher.collect_metadata(source="apache/polaris", paths=["1.3.0"])

        assert metadata is not None
        assert len(metadata.file_paths) >= 1
        assert metadata.version is not None

    def test_streaming_processor_github(self, temp_dir):
        """Test streaming processor with GitHub source (apache/polaris)."""
        from datus.storage.document.fetcher.github_fetcher import GitHubFetcher

        db_path = str(Path(temp_dir) / "store")
        store = document_store(db_path)

        fetcher = GitHubFetcher(
            platform="polaris",
            token=os.environ.get("GITHUB_TOKEN"),
            github_ref="versioned-docs",
            pool_size=2,
        )

        metadata = fetcher.collect_metadata(source="apache/polaris", paths=["1.3.0"])
        assert len(metadata.file_paths) >= 1

        processor = StreamingDocProcessor(
            store=store,
            chunk_size=512,
            pool_size=2,
        )

        stats = processor.process_github(
            fetcher=fetcher,
            metadata=metadata,
            version=metadata.version,
            platform="polaris",
        )

        assert stats.total_docs >= 1
        assert stats.total_chunks >= 1

    def test_init_platform_docs_github_polaris(self, temp_dir):
        """Test full pipeline with GitHub source (apache/polaris versioned paths)."""
        db_path = str(Path(temp_dir) / "store")

        cfg = DocumentConfig(
            type="github",
            source="apache/polaris",
            paths=["1.3.0", "1.2.0"],
            github_token=os.environ.get("GITHUB_TOKEN"),
            github_ref="versioned-docs",
        )

        result = init_platform_docs(
            db_path=db_path,
            platform="polaris",
            cfg=cfg,
            build_mode="overwrite",
        )

        assert result.success, f"GitHub pipeline failed for polaris: {result.errors}"
        assert result.total_docs >= 1
        # Should detect multi-version from paths
        assert "1.3.0" in result.version or "1.2.0" in result.version

    def test_init_platform_docs_github_starrocks(self, temp_dir):
        """Test full pipeline with GitHub source (StarRocks/starrocks)."""
        db_path = str(Path(temp_dir) / "store")

        cfg = DocumentConfig(
            type="github",
            source="StarRocks/starrocks",
            paths=["docs/en", "README.md"],
            github_token=os.environ.get("GITHUB_TOKEN"),
            github_ref="4.0.5",
        )

        result = init_platform_docs(
            db_path=db_path,
            platform="starrocks",
            cfg=cfg,
            build_mode="overwrite",
        )

        assert result.success, f"GitHub pipeline failed for starrocks: {result.errors}"
        assert result.total_docs >= 1
        assert result.version == "4.0.5"


# =============================================================================
# Web Fetcher Integration Tests (Requires Network)
# =============================================================================


@pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK_TESTS", "").lower() in ("1", "true"),
    reason="Skipping network-dependent tests (SKIP_NETWORK_TESTS is set)",
)
class TestWebFetcherIntegration:
    """Integration tests for web fetcher (requires network access)."""

    def test_web_fetch_single_page(self):
        """Test fetching a single web page."""
        from datus.storage.document.fetcher.web_fetcher import WebFetcher

        fetcher = WebFetcher(platform="snowflake", version="latest")
        doc = fetcher.fetch_single("https://docs.snowflake.com/en")

        assert doc is not None
        assert doc.content_type == CONTENT_TYPE_HTML
        assert doc.source_type == SOURCE_TYPE_WEBSITE

    def test_web_fetch_with_crawling(self):
        """Test web crawling with depth > 0 using legacy fetch method."""
        from datus.storage.document.fetcher.web_fetcher import WebFetcher

        fetcher = WebFetcher(platform="snowflake", version="latest")
        docs = fetcher.fetch(
            source="https://docs.snowflake.com/en",
            max_depth=1,
            include_patterns=["en/sql-reference", "en/user-guide"],
        )

        # Should find multiple pages
        assert len(docs) >= 1

    def test_streaming_processor_website(self, temp_dir):
        """Test streaming processor with website source."""
        from datus.storage.document.fetcher.web_fetcher import WebFetcher

        db_path = str(Path(temp_dir) / "store")
        store = document_store(db_path)

        fetcher = WebFetcher(platform="snowflake", version="latest", pool_size=2)

        processor = StreamingDocProcessor(
            store=store,
            chunk_size=512,
            pool_size=2,
        )

        # Test with small depth to limit crawling
        stats = processor.process_website(
            fetcher=fetcher,
            base_url="https://docs.snowflake.com/en",
            version="latest",
            platform="snowflake",
            max_depth=1,
            include_patterns=["en/sql-reference"],
        )

        assert stats.total_docs >= 1
        assert stats.total_chunks >= 1

    def test_init_platform_docs_website(self, temp_dir):
        """Test full pipeline with website source (snowflake)."""
        db_path = str(Path(temp_dir) / "store")

        cfg = DocumentConfig(
            type="website",
            source="https://docs.snowflake.com/en/",
            max_depth=1,
            include_patterns=["en/sql-reference"],
        )

        result = init_platform_docs(
            db_path=db_path,
            platform="snowflake",
            cfg=cfg,
            build_mode="overwrite",
        )

        assert result.success, f"Website pipeline failed: {result.errors}"
        assert result.total_docs >= 1


# =============================================================================
# End-to-End Helpers
# =============================================================================


def _find_nav_leaf(nodes):
    """Recursively find a leaf document title from a navigation tree."""
    for node in nodes:
        if "tree" in node:
            leaf = _find_nav_leaf(node["tree"])
            if leaf:
                return leaf
        elif not node.get("children"):
            return node.get("name")
        else:
            leaf = _find_nav_leaf(node["children"])
            if leaf:
                return leaf
    return None


def _run_e2e_workflow(db_path, platform, cfg, search_keywords):
    """Run the complete E2E workflow: init -> list_nav -> search -> get_document.

    Returns (_init_result, _nav_result, _search_result, _doc_result) tuple.
    """
    # Step 1: Initialize platform docs
    _init_result = init_platform_docs(
        db_path=db_path,
        platform=platform,
        cfg=cfg,
        build_mode="overwrite",
    )
    assert _init_result.success, f"Init failed for {platform}: {_init_result.errors}"
    assert _init_result.total_docs >= 1, f"No docs found for {platform}"
    assert _init_result.total_chunks >= 1, f"No chunks created for {platform}"
    logger.info(
        f"[{platform}] Initialized {_init_result.total_chunks} chunks "
        f"from {_init_result.total_docs} docs (v{_init_result.version})"
    )

    # Step 2: Create SearchTool with test config
    class _TestConfig:
        def document_storage_path(self, _platform):
            return db_path

    tool = SearchTool(agent_config=_TestConfig())

    # Step 3: List navigation
    _nav_result = tool.list_document_nav(platform=platform)
    assert _nav_result.success, f"list_document_nav failed for {platform}: {_nav_result.error}"
    assert _nav_result.total_docs > 0, f"Nav tree empty for {platform}"
    logger.info(f"[{platform}] Navigation tree has {_nav_result.total_docs} documents")

    # Step 4: Search for content
    _search_result = tool.search_document(
        platform=platform,
        keywords=search_keywords,
        top_n=5,
    )
    assert _search_result.success, f"search_document failed for {platform}: {_search_result.error}"
    assert _search_result.doc_count > 0, f"Search returned no results for {platform}"
    logger.info(f"[{platform}] Search found {_search_result.doc_count} results")

    # Step 5: Get document by title from nav tree
    title = _find_nav_leaf(_nav_result.nav_tree)
    assert title, f"Could not find a leaf document title in nav tree for {platform}"

    _doc_result = tool.get_document(platform=platform, titles=[title])
    assert _doc_result.success, f"get_document failed for {platform}: {_doc_result.error}"
    assert _doc_result.chunk_count > 0, f"get_document returned no chunks for {platform}"
    logger.info(f"[{platform}] Got document '{title}' with {_doc_result.chunk_count} chunks")

    return _init_result, _nav_result, _search_result, _doc_result


# =============================================================================
# End-to-End Integration Test
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration test covering the complete workflow."""

    def test_complete_workflow(self, local_docs_dir, temp_dir, agent_config):
        """Test complete workflow: init → search → get_document."""
        db_path = str(Path(temp_dir) / "store")

        cfg = DocumentConfig(
            type="local",
            source=str(local_docs_dir),
            version="v1.0",
            chunk_size=256,
        )

        _init_result, _nav_result, _search_result, _doc_result = _run_e2e_workflow(
            db_path=db_path,
            platform="test_e2e",
            cfg=cfg,
            search_keywords=["installation", "configuration"],
        )

        # Verify chunk content
        for chunk in _doc_result.chunks[:3]:
            assert "chunk_text" in chunk
            assert chunk["chunk_text"]
            logger.info(f"  Chunk: {chunk.get('title', 'N/A')[:50]}...")

    def test_complete_workflow_local_multi_dir(self, temp_dir):
        """Test complete workflow with multiple separate local doc directories."""
        db_path = str(Path(temp_dir) / "store")
        root_dir = Path(temp_dir) / "multi_docs"
        root_dir.mkdir()

        # Directory 1: API documentation
        api_dir = root_dir / "api_reference"
        api_dir.mkdir()
        (api_dir / "endpoints.md").write_text(
            """# REST API Endpoints

## Authentication

All API requests require a Bearer token.

### POST /auth/login

Authenticates a user and returns an access token.

**Request Body:**
```json
{"username": "admin", "password": "secret"}
```

### GET /api/v1/users

Returns a list of registered users.

### POST /api/v1/queries

Submit a SQL query for execution.
"""
        )
        (api_dir / "errors.md").write_text(
            """# Error Handling

## Error Codes

| Code | Description |
|------|-------------|
| 400  | Bad Request |
| 401  | Unauthorized |
| 500  | Internal Server Error |

## Retry Policy

Failed requests should be retried with exponential backoff.
"""
        )

        # Directory 2: User guides
        guide_dir = root_dir / "user_guides"
        guide_dir.mkdir()
        (guide_dir / "quickstart.md").write_text(
            """# Quick Start Guide

## Prerequisites

- Python 3.9 or higher
- A running database instance

## Installation

```bash
pip install datus-agent
```

## First Query

Run your first natural language query:

```python
from datus import Agent
agent = Agent(config="agent.yml")
result = agent.query("Show me total sales by region")
```
"""
        )

        # Directory 3: Architecture docs
        arch_dir = root_dir / "architecture"
        arch_dir.mkdir()
        (arch_dir / "overview.md").write_text(
            """# System Architecture

## Components

### Query Parser
Converts natural language to structured intent.

### Schema Linker
Maps intent to database schema elements.

### SQL Generator
Produces optimized SQL from linked schema.

## Data Flow

User Query → Parser → Linker → Generator → Database → Results
"""
        )

        # Initialize with the root directory (recursive fetch finds all subdirs)
        cfg = DocumentConfig(
            type="local",
            source=str(root_dir),
            version="v1.0",
            chunk_size=256,
        )

        _init_result, _nav_result, _search_result, _doc_result = _run_e2e_workflow(
            db_path=db_path,
            platform="test_multi_dir",
            cfg=cfg,
            search_keywords=["authentication", "installation", "schema linker"],
        )

        # Multi-dir specific: should find docs from all 3 directories
        assert _init_result.total_docs == 4, f"Expected 4 docs from 3 directories, got {_init_result.total_docs}"
        assert _nav_result.total_docs == 4
        # Should find results for keywords spanning different directories
        for keyword in ["authentication", "installation", "schema linker"]:
            assert keyword in _search_result.docs, f"Missing search results for '{keyword}'"
            assert len(_search_result.docs[keyword]) > 0, f"No results for '{keyword}'"


# =============================================================================
# Real Platform End-to-End Integration Tests
# =============================================================================


class TestEndToEndRealPlatforms:
    """End-to-end integration tests with real platform documentation sources."""

    @pytest.mark.skipif(
        not os.environ.get("GITHUB_TOKEN"),
        reason="Requires GITHUB_TOKEN environment variable",
    )
    def test_complete_workflow_starrocks(self, temp_dir):
        """Test complete workflow with StarRocks GitHub documentation."""
        db_path = str(Path(temp_dir) / "store")

        cfg = DocumentConfig(
            type="github",
            source="StarRocks/starrocks",
            paths=["docs/en/sql-reference/sql-statements"],
            github_token=os.environ.get("GITHUB_TOKEN"),
            github_ref="4.0.5",
            chunk_size=512,
        )

        _init_result, _nav_result, _search_result, _doc_result = _run_e2e_workflow(
            db_path=db_path,
            platform="starrocks_e2e",
            cfg=cfg,
            search_keywords=["CREATE TABLE", "materialized view"],
        )

        # StarRocks-specific assertions
        assert _init_result.version == "4.0.5"

    @pytest.mark.skipif(
        not os.environ.get("GITHUB_TOKEN"),
        reason="Requires GITHUB_TOKEN environment variable",
    )
    def test_complete_workflow_starrocks_multi_dir(self, temp_dir):
        """Test complete workflow with StarRocks using multiple directory paths."""
        db_path = str(Path(temp_dir) / "store")

        cfg = DocumentConfig(
            type="github",
            source="StarRocks/starrocks",
            paths=["docs/en/sql-reference/sql-statements", "docs/en/loading"],
            github_token=os.environ.get("GITHUB_TOKEN"),
            github_ref="4.0.5",
            chunk_size=512,
        )

        _init_result, _nav_result, _search_result, _doc_result = _run_e2e_workflow(
            db_path=db_path,
            platform="starrocks_multi_e2e",
            cfg=cfg,
            search_keywords=["CREATE TABLE", "BROKER LOAD"],
        )

        # Multi-dir: should have docs from both sql-statements and loading
        assert _init_result.version == "4.0.5"
        assert _init_result.total_docs >= 2, "Should have docs from multiple directories"

    @pytest.mark.skipif(
        not os.environ.get("GITHUB_TOKEN"),
        reason="Requires GITHUB_TOKEN environment variable",
    )
    def test_complete_workflow_polaris(self, temp_dir):
        """Test complete workflow with Apache Polaris multi-version documentation."""
        db_path = str(Path(temp_dir) / "store")

        cfg = DocumentConfig(
            type="github",
            source="apache/polaris",
            paths=["1.3.0", "1.2.0"],
            github_token=os.environ.get("GITHUB_TOKEN"),
            github_ref="versioned-docs",
            chunk_size=512,
        )

        _init_result, _nav_result, _search_result, _doc_result = _run_e2e_workflow(
            db_path=db_path,
            platform="polaris_e2e",
            cfg=cfg,
            search_keywords=["catalog", "namespace"],
        )

        # Polaris-specific: multi-version paths should be detected
        assert "1.3.0" in _init_result.version
        assert "1.2.0" in _init_result.version

    @pytest.mark.skipif(
        os.environ.get("SKIP_NETWORK_TESTS", "").lower() in ("1", "true"),
        reason="Skipping network-dependent tests (SKIP_NETWORK_TESTS is set)",
    )
    def test_complete_workflow_snowflake(self, temp_dir):
        """Test complete workflow with Snowflake website documentation."""
        db_path = str(Path(temp_dir) / "store")

        cfg = DocumentConfig(
            type="website",
            source="https://docs.snowflake.com/en/",
            max_depth=1,
            include_patterns=["en/sql-reference"],
            chunk_size=512,
        )

        _run_e2e_workflow(
            db_path=db_path,
            platform="snowflake_e2e",
            cfg=cfg,
            search_keywords=["CREATE TABLE", "warehouse"],
        )
