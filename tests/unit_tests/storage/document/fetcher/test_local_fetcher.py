# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.document.fetcher.local_fetcher."""

import pytest

from datus.storage.document.fetcher.local_fetcher import LocalFetcher
from datus.storage.document.schemas import (
    CONTENT_TYPE_HTML,
    CONTENT_TYPE_MARKDOWN,
    CONTENT_TYPE_RST,
    SOURCE_TYPE_LOCAL,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fetcher():
    """Return a LocalFetcher with a fixed version to keep assertions deterministic."""
    return LocalFetcher(platform="test-platform", version="1.0.0")


@pytest.fixture()
def fetcher_no_version():
    """Fetcher without an explicit version (auto-generates from current date)."""
    return LocalFetcher(platform="auto-ver")


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestLocalFetcherInit:
    """Tests for LocalFetcher initialization."""

    @pytest.mark.ci
    def test_init_stores_platform_and_version(self):
        """Platform and version should be stored from constructor."""
        f = LocalFetcher(platform="snowflake", version="v3.0")
        assert f.platform == "snowflake"
        assert f.version == "v3.0"

    @pytest.mark.ci
    def test_init_version_default_none(self):
        """Version should default to None when omitted."""
        f = LocalFetcher(platform="duckdb")
        assert f.version is None

    @pytest.mark.ci
    def test_doc_extensions_class_attribute(self):
        """DOC_EXTENSIONS should contain all supported extensions."""
        expected = {".md", ".markdown", ".html", ".htm", ".rst", ".txt"}
        assert LocalFetcher.DOC_EXTENSIONS == expected


# ---------------------------------------------------------------------------
# fetch() -- directory scanning
# ---------------------------------------------------------------------------


class TestFetch:
    """Tests for LocalFetcher.fetch (directory scanning)."""

    @pytest.mark.ci
    def test_fetch_nonexistent_directory(self, fetcher, tmp_path):
        """Fetching from a nonexistent directory should return an empty list."""
        result = fetcher.fetch(str(tmp_path / "no_such_dir"))
        assert result == []

    @pytest.mark.ci
    def test_fetch_not_a_directory(self, fetcher, tmp_path):
        """Fetching from a file path (not directory) should return an empty list."""
        f = tmp_path / "file.txt"
        f.write_text("hello")
        result = fetcher.fetch(str(f))
        assert result == []

    @pytest.mark.ci
    def test_fetch_empty_directory(self, fetcher, tmp_path):
        """An empty directory should return zero documents."""
        result = fetcher.fetch(str(tmp_path))
        assert result == []

    @pytest.mark.ci
    def test_fetch_single_markdown_file(self, fetcher, tmp_path):
        """A directory with one .md file should yield exactly one FetchedDocument."""
        md = tmp_path / "guide.md"
        md.write_text("# Guide\n\nHello world")
        docs = fetcher.fetch(str(tmp_path))
        assert len(docs) == 1
        doc = docs[0]
        assert doc.platform == "test-platform"
        assert doc.version == "1.0.0"
        assert doc.source_type == SOURCE_TYPE_LOCAL
        assert doc.content_type == CONTENT_TYPE_MARKDOWN
        assert doc.raw_content == "# Guide\n\nHello world"
        assert doc.doc_path == "guide.md"

    @pytest.mark.ci
    def test_fetch_multiple_file_types(self, fetcher, tmp_path):
        """Should pick up .md, .html, .rst, .txt files."""
        (tmp_path / "a.md").write_text("# A")
        (tmp_path / "b.html").write_text("<h1>B</h1>")
        (tmp_path / "c.rst").write_text("C\n===")
        (tmp_path / "d.txt").write_text("D content")
        docs = fetcher.fetch(str(tmp_path))
        assert len(docs) == 4
        doc_paths = {d.doc_path for d in docs}
        assert doc_paths == {"a.md", "b.html", "c.rst", "d.txt"}

    @pytest.mark.ci
    def test_fetch_skips_non_doc_extensions(self, fetcher, tmp_path):
        """Non-doc files (.py, .csv, .png) should be excluded."""
        (tmp_path / "code.py").write_text("print('hi')")
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "note.md").write_text("# Note")
        docs = fetcher.fetch(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].doc_path == "note.md"

    @pytest.mark.ci
    def test_fetch_recursive(self, fetcher, tmp_path):
        """With recursive=True (default), nested directories should be scanned."""
        sub = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "nested.md").write_text("# Nested")
        (tmp_path / "top.md").write_text("# Top")
        docs = fetcher.fetch(str(tmp_path), recursive=True)
        assert len(docs) == 2

    @pytest.mark.ci
    def test_fetch_non_recursive(self, fetcher, tmp_path):
        """With recursive=False, only top-level files should be returned."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("# Nested")
        (tmp_path / "top.md").write_text("# Top")
        docs = fetcher.fetch(str(tmp_path), recursive=False)
        assert len(docs) == 1
        assert docs[0].doc_path == "top.md"

    @pytest.mark.ci
    def test_fetch_skips_empty_files(self, fetcher, tmp_path):
        """Empty files (or whitespace-only) should be skipped."""
        (tmp_path / "empty.md").write_text("")
        (tmp_path / "whitespace.md").write_text("   \n  \n")
        (tmp_path / "real.md").write_text("# Real content")
        docs = fetcher.fetch(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].doc_path == "real.md"

    @pytest.mark.ci
    def test_fetch_include_patterns(self, fetcher, tmp_path):
        """Only files matching include_patterns should be returned."""
        (tmp_path / "keep.md").write_text("# Keep")
        (tmp_path / "skip.html").write_text("<h1>Skip</h1>")
        docs = fetcher.fetch(str(tmp_path), include_patterns=["*.md"])
        assert len(docs) == 1
        assert docs[0].doc_path == "keep.md"

    @pytest.mark.ci
    def test_fetch_exclude_patterns(self, fetcher, tmp_path):
        """Files matching exclude_patterns should be excluded."""
        (tmp_path / "README.md").write_text("# README")
        (tmp_path / "guide.md").write_text("# Guide")
        docs = fetcher.fetch(str(tmp_path), exclude_patterns=["README.md"])
        assert len(docs) == 1
        assert docs[0].doc_path == "guide.md"

    @pytest.mark.ci
    def test_fetch_auto_version_when_not_set(self, fetcher_no_version, tmp_path):
        """When version is None, fetch should auto-generate a date-based version."""
        (tmp_path / "note.md").write_text("# Note")
        docs = fetcher_no_version.fetch(str(tmp_path))
        assert len(docs) == 1
        # Version should be YYYY-MM-DD format
        import re

        assert re.match(r"\d{4}-\d{2}-\d{2}", docs[0].version)

    @pytest.mark.ci
    def test_fetch_source_url_is_file_uri(self, fetcher, tmp_path):
        """source_url should be a file:// URI."""
        (tmp_path / "doc.md").write_text("# Doc")
        docs = fetcher.fetch(str(tmp_path))
        assert docs[0].source_url.startswith("file://")

    @pytest.mark.ci
    def test_fetch_metadata_fields(self, fetcher, tmp_path):
        """Metadata should contain title, file_name, file_size, last_modified, absolute_path."""
        (tmp_path / "info.md").write_text("# Info Title\n\nContent here")
        docs = fetcher.fetch(str(tmp_path))
        meta = docs[0].metadata
        assert "title" in meta
        assert meta["file_name"] == "info.md"
        assert isinstance(meta["file_size"], int)
        assert meta["file_size"] > 0
        assert "last_modified" in meta
        assert "absolute_path" in meta


# ---------------------------------------------------------------------------
# fetch_single()
# ---------------------------------------------------------------------------


class TestFetchSingle:
    """Tests for LocalFetcher.fetch_single."""

    @pytest.mark.ci
    def test_fetch_single_nonexistent_file(self, fetcher, tmp_path):
        """Should return None for a nonexistent file."""
        result = fetcher.fetch_single(str(tmp_path / "ghost.md"))
        assert result is None

    @pytest.mark.ci
    def test_fetch_single_directory_path(self, fetcher, tmp_path):
        """Should return None when path points to a directory."""
        result = fetcher.fetch_single(str(tmp_path))
        assert result is None

    @pytest.mark.ci
    def test_fetch_single_valid_markdown(self, fetcher, tmp_path):
        """Should return a FetchedDocument for a valid markdown file."""
        md = tmp_path / "single.md"
        md.write_text("# Single File\n\nContent")
        doc = fetcher.fetch_single(str(md))
        assert doc is not None
        assert doc.platform == "test-platform"
        assert doc.version == "1.0.0"
        assert doc.content_type == CONTENT_TYPE_MARKDOWN
        assert doc.raw_content == "# Single File\n\nContent"

    @pytest.mark.ci
    def test_fetch_single_with_base_url(self, fetcher, tmp_path):
        """When base_url is given, doc_path should be relative to it."""
        sub = tmp_path / "docs" / "api"
        sub.mkdir(parents=True)
        f = sub / "ref.md"
        f.write_text("# API Reference")
        doc = fetcher.fetch_single(str(f), base_url=str(tmp_path / "docs"))
        assert doc is not None
        assert doc.doc_path == "api/ref.md"

    @pytest.mark.ci
    def test_fetch_single_without_base_url(self, fetcher, tmp_path):
        """Without base_url, doc_path should default to just the filename."""
        f = tmp_path / "standalone.md"
        f.write_text("# Standalone")
        doc = fetcher.fetch_single(str(f))
        assert doc is not None
        assert doc.doc_path == "standalone.md"

    @pytest.mark.ci
    def test_fetch_single_html_file(self, fetcher, tmp_path):
        """HTML files should be detected with html content type."""
        f = tmp_path / "page.html"
        f.write_text("<html><head><title>My Page</title></head><body>Hello</body></html>")
        doc = fetcher.fetch_single(str(f))
        assert doc is not None
        assert doc.content_type == CONTENT_TYPE_HTML

    @pytest.mark.ci
    def test_fetch_single_auto_version(self, fetcher_no_version, tmp_path):
        """When version is None, should auto-generate a date-based version."""
        f = tmp_path / "auto.md"
        f.write_text("# Auto")
        doc = fetcher_no_version.fetch_single(str(f))
        assert doc is not None
        import re

        assert re.match(r"\d{4}-\d{2}-\d{2}", doc.version)


# ---------------------------------------------------------------------------
# _fetch_file() -- error handling
# ---------------------------------------------------------------------------


class TestFetchFileErrors:
    """Tests for _fetch_file error paths."""

    @pytest.mark.ci
    def test_non_utf8_file_returns_none(self, fetcher, tmp_path):
        """Binary / non-UTF-8 files should be skipped gracefully."""
        f = tmp_path / "binary.md"
        f.write_bytes(b"\x80\x81\x82\xff\xfe")
        doc = fetcher._fetch_file(f, tmp_path, "1.0.0")
        assert doc is None

    @pytest.mark.ci
    def test_empty_file_returns_none(self, fetcher, tmp_path):
        """An empty file should return None."""
        f = tmp_path / "empty.md"
        f.write_text("")
        doc = fetcher._fetch_file(f, tmp_path, "1.0.0")
        assert doc is None

    @pytest.mark.ci
    def test_whitespace_only_file_returns_none(self, fetcher, tmp_path):
        """A whitespace-only file should return None."""
        f = tmp_path / "ws.md"
        f.write_text("   \n\t\n  ")
        doc = fetcher._fetch_file(f, tmp_path, "1.0.0")
        assert doc is None


# ---------------------------------------------------------------------------
# _extract_title()
# ---------------------------------------------------------------------------


class TestExtractTitle:
    """Tests for LocalFetcher._extract_title."""

    @pytest.mark.ci
    def test_markdown_atx_heading(self, fetcher):
        """Should extract title from ATX-style # heading."""
        title = fetcher._extract_title("# My Title\n\nContent", "file.md", CONTENT_TYPE_MARKDOWN)
        assert title == "My Title"

    @pytest.mark.ci
    def test_markdown_setext_heading_equals(self, fetcher):
        """Should extract title from setext-style heading with ===."""
        title = fetcher._extract_title("My Setext Title\n================\n\nContent", "f.md", CONTENT_TYPE_MARKDOWN)
        assert title == "My Setext Title"

    @pytest.mark.ci
    def test_markdown_setext_heading_dashes(self, fetcher):
        """Should extract title from setext-style heading with ---."""
        title = fetcher._extract_title("Dash Title\n----------\n\nContent", "f.md", CONTENT_TYPE_MARKDOWN)
        assert title == "Dash Title"

    @pytest.mark.ci
    def test_rst_setext_heading_tilde(self, fetcher):
        """Should extract title from RST setext-style heading with ~~~."""
        title = fetcher._extract_title("RST Title\n~~~~~~~~~\n\nContent", "f.rst", CONTENT_TYPE_RST)
        assert title == "RST Title"

    @pytest.mark.ci
    def test_rst_setext_heading_caret(self, fetcher):
        """Should extract title from RST setext-style heading with ^^^."""
        title = fetcher._extract_title("Caret Title\n^^^^^^^^^^^\n\nContent", "f.rst", CONTENT_TYPE_RST)
        assert title == "Caret Title"

    @pytest.mark.ci
    def test_html_title_tag(self, fetcher):
        """Should extract title from <title> tag in HTML."""
        html = "<html><head><title>Page Title</title></head><body></body></html>"
        title = fetcher._extract_title(html, "page.html", CONTENT_TYPE_HTML)
        assert title == "Page Title"

    @pytest.mark.ci
    def test_html_h1_tag(self, fetcher):
        """Should extract title from <h1> when no <title> is present."""
        html = "<html><body><h1>Heading One</h1></body></html>"
        title = fetcher._extract_title(html, "page.html", CONTENT_TYPE_HTML)
        assert title == "Heading One"

    @pytest.mark.ci
    def test_html_h1_with_nested_tags(self, fetcher):
        """Should strip inner HTML tags from <h1> content."""
        html = "<html><body><h1><span>Bold</span> Title</h1></body></html>"
        title = fetcher._extract_title(html, "page.html", CONTENT_TYPE_HTML)
        assert title == "Bold Title"

    @pytest.mark.ci
    def test_fallback_to_filename(self, fetcher):
        """When no heading is found, should title-case the filename stem."""
        title = fetcher._extract_title("Just some text", "my_doc-file.md", CONTENT_TYPE_MARKDOWN)
        assert title == "My Doc File"

    @pytest.mark.ci
    def test_html_fallback_to_filename(self, fetcher):
        """HTML without title or h1 should fall back to filename."""
        html = "<html><body><p>Just a paragraph</p></body></html>"
        title = fetcher._extract_title(html, "some-page.html", CONTENT_TYPE_HTML)
        assert title == "Some Page"


# ---------------------------------------------------------------------------
# _detect_content_type (inherited from BaseFetcher, but exercised via fetch)
# ---------------------------------------------------------------------------


class TestContentTypeDetection:
    """Verify content type detection through the fetch pipeline."""

    @pytest.mark.ci
    def test_markdown_extension(self, fetcher, tmp_path):
        """'.md' should be detected as markdown."""
        (tmp_path / "doc.md").write_text("# MD")
        docs = fetcher.fetch(str(tmp_path))
        assert docs[0].content_type == CONTENT_TYPE_MARKDOWN

    @pytest.mark.ci
    def test_markdown_long_extension(self, fetcher, tmp_path):
        """'.markdown' should be detected as markdown."""
        (tmp_path / "doc.markdown").write_text("# MD long")
        docs = fetcher.fetch(str(tmp_path))
        assert docs[0].content_type == CONTENT_TYPE_MARKDOWN

    @pytest.mark.ci
    def test_html_extension(self, fetcher, tmp_path):
        """'.html' should be detected as html."""
        (tmp_path / "page.html").write_text("<h1>Page</h1>")
        docs = fetcher.fetch(str(tmp_path))
        assert docs[0].content_type == CONTENT_TYPE_HTML

    @pytest.mark.ci
    def test_htm_extension(self, fetcher, tmp_path):
        """'.htm' should be detected as html."""
        (tmp_path / "page.htm").write_text("<h1>Page</h1>")
        docs = fetcher.fetch(str(tmp_path))
        assert docs[0].content_type == CONTENT_TYPE_HTML

    @pytest.mark.ci
    def test_rst_extension(self, fetcher, tmp_path):
        """'.rst' should be detected as rst."""
        (tmp_path / "doc.rst").write_text("RST Title\n=========\n\nContent")
        docs = fetcher.fetch(str(tmp_path))
        assert docs[0].content_type == CONTENT_TYPE_RST
