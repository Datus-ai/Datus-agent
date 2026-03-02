# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.document.fetcher.base_fetcher."""

from typing import List, Optional

import pytest

from datus.storage.document.fetcher.base_fetcher import BaseFetcher
from datus.storage.document.schemas import FetchedDocument

# ---------------------------------------------------------------------------
# Concrete subclass for testing (BaseFetcher is abstract)
# ---------------------------------------------------------------------------


class _StubFetcher(BaseFetcher):
    """Minimal concrete fetcher so we can test non-abstract helper methods."""

    def fetch(self, source: str, **kwargs) -> List[FetchedDocument]:
        return []

    def fetch_single(self, path: str, **kwargs) -> Optional[FetchedDocument]:
        return None


# ---------------------------------------------------------------------------
# _is_doc_file
# ---------------------------------------------------------------------------


class TestIsDocFile:
    """Tests for BaseFetcher._is_doc_file."""

    @pytest.fixture()
    def fetcher(self):
        return _StubFetcher(platform="test")

    @pytest.mark.parametrize(
        "filename",
        [
            "README.md",
            "guide.rst",
            "notes.txt",
            "page.html",
            "index.htm",
        ],
    )
    def test_is_doc_file_known_extensions(self, fetcher, filename):
        """Files with standard doc extensions should be recognized."""
        assert fetcher._is_doc_file(filename) is True

    @pytest.mark.parametrize(
        "filename",
        [
            "README.MD",
            "GUIDE.RST",
            "NOTES.TXT",
            "PAGE.HTML",
            "INDEX.HTM",
        ],
    )
    def test_is_doc_file_case_insensitive(self, fetcher, filename):
        """Extension matching should be case-insensitive."""
        assert fetcher._is_doc_file(filename) is True

    @pytest.mark.parametrize(
        "filename",
        [
            "readme",
            "CHANGELOG",
            "contributing",
            "LICENSE",
        ],
    )
    def test_is_doc_file_special_filenames(self, fetcher, filename):
        """Well-known filenames without extensions should be recognized."""
        assert fetcher._is_doc_file(filename) is True

    @pytest.mark.parametrize(
        "filename",
        [
            "readme.cfg",
            "CHANGELOG.json",
        ],
    )
    def test_is_doc_file_special_name_with_non_doc_extension(self, fetcher, filename):
        """Special names with non-doc extensions should still match by name part."""
        assert fetcher._is_doc_file(filename) is True

    @pytest.mark.parametrize(
        "filename",
        [
            "main.py",
            "config.yaml",
            "data.csv",
            "image.png",
            "Makefile",
        ],
    )
    def test_is_doc_file_non_doc_files(self, fetcher, filename):
        """Non-documentation files should return False."""
        assert fetcher._is_doc_file(filename) is False


# ---------------------------------------------------------------------------
# _detect_content_type
# ---------------------------------------------------------------------------


class TestDetectContentType:
    """Tests for BaseFetcher._detect_content_type."""

    @pytest.fixture()
    def fetcher(self):
        return _StubFetcher(platform="test")

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("guide.md", "markdown"),
            ("guide.markdown", "markdown"),
            ("page.html", "html"),
            ("page.htm", "html"),
            ("doc.rst", "rst"),
        ],
    )
    def test_detect_content_type_by_extension(self, fetcher, filename, expected):
        """Should detect content type from well-known file extensions."""
        result = fetcher._detect_content_type(filename, "some content")
        assert result == expected

    def test_detect_content_type_extension_case_insensitive(self, fetcher):
        """Extension detection should be case-insensitive."""
        assert fetcher._detect_content_type("DOC.MD", "content") == "markdown"
        assert fetcher._detect_content_type("PAGE.HTML", "content") == "html"
        assert fetcher._detect_content_type("DOC.RST", "content") == "rst"

    def test_detect_content_type_html_from_doctype(self, fetcher):
        """HTML content starting with <!DOCTYPE should be detected."""
        content = "<!DOCTYPE html>\n<html><body>Hello</body></html>"
        result = fetcher._detect_content_type("unknown_file", content)
        assert result == "html"

    def test_detect_content_type_html_from_html_tag(self, fetcher):
        """HTML content starting with <html should be detected."""
        content = "<html>\n<head><title>Test</title></head><body></body></html>"
        result = fetcher._detect_content_type("unknown_file", content)
        assert result == "html"

    def test_detect_content_type_html_from_content_with_whitespace(self, fetcher):
        """HTML detection should handle leading whitespace in content."""
        content = "   \n  <!DOCTYPE html>\n<html></html>"
        result = fetcher._detect_content_type("unknown_file", content)
        assert result == "html"

    def test_detect_content_type_default_markdown(self, fetcher):
        """Unknown extension and non-HTML content should default to markdown."""
        result = fetcher._detect_content_type("readme", "# Some heading\nSome text")
        assert result == "markdown"

    def test_detect_content_type_txt_defaults_to_markdown(self, fetcher):
        """A .txt file is not in the extension map, so should default to markdown."""
        result = fetcher._detect_content_type("notes.txt", "plain text")
        assert result == "markdown"


# ---------------------------------------------------------------------------
# __init__ attributes
# ---------------------------------------------------------------------------


class TestBaseFetcherInit:
    """Tests for BaseFetcher initialization."""

    def test_init_platform_and_version(self):
        """platform and version attributes should be set from constructor."""
        f = _StubFetcher(platform="snowflake", version="v3.2.1")
        assert f.platform == "snowflake"
        assert f.version == "v3.2.1"

    def test_init_version_default_none(self):
        """version should default to None when not provided."""
        f = _StubFetcher(platform="duckdb")
        assert f.version is None
