# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus/storage/document/nav_resolver/__init__.py -- supplemental coverage.

Supplements tests/unit_tests/storage/test_nav_resolver.py with additional coverage
for NavResolverPipeline._extract_frontmatter_context, _parse_simple_frontmatter,
and RESOLVER_MAP.
"""

from datus.storage.document.nav_resolver import RESOLVER_MAP, NavResolverPipeline, _parse_simple_frontmatter
from datus.storage.document.nav_resolver.detector import FRAMEWORK_DOCUSAURUS, FRAMEWORK_HUGO, FRAMEWORK_MKDOCS
from datus.storage.document.nav_resolver.docusaurus_resolver import DocusaurusResolver
from datus.storage.document.nav_resolver.hugo_resolver import HugoResolver
from datus.storage.document.nav_resolver.mkdocs_resolver import MkDocsResolver
from datus.storage.document.schemas import FetchedDocument

# ============================================================
# RESOLVER_MAP
# ============================================================


class TestResolverMap:
    """Tests for the RESOLVER_MAP constant."""

    def test_resolver_map_has_three_entries(self):
        """RESOLVER_MAP should contain exactly 3 framework entries."""
        assert len(RESOLVER_MAP) == 3

    def test_docusaurus_maps_to_resolver(self):
        """Docusaurus should map to DocusaurusResolver."""
        assert RESOLVER_MAP[FRAMEWORK_DOCUSAURUS] is DocusaurusResolver

    def test_hugo_maps_to_resolver(self):
        """Hugo should map to HugoResolver."""
        assert RESOLVER_MAP[FRAMEWORK_HUGO] is HugoResolver

    def test_mkdocs_maps_to_resolver(self):
        """MkDocs should map to MkDocsResolver."""
        assert RESOLVER_MAP[FRAMEWORK_MKDOCS] is MkDocsResolver


# ============================================================
# _parse_simple_frontmatter
# ============================================================


class TestParseSimpleFrontmatter:
    """Tests for _parse_simple_frontmatter function."""

    def test_standard_frontmatter(self):
        """Standard YAML frontmatter should be parsed correctly."""
        content = """---
title: My Page
weight: 100
sidebar_position: 5
---
# Content here
"""
        result = _parse_simple_frontmatter(content)
        assert result["title"] == "My Page"
        assert result["weight"] == "100"
        assert result["sidebar_position"] == "5"

    def test_no_frontmatter(self):
        """Content without frontmatter should return empty dict."""
        result = _parse_simple_frontmatter("# Just a heading\nSome content")
        assert result == {}

    def test_empty_frontmatter(self):
        """Empty frontmatter block should return empty dict."""
        result = _parse_simple_frontmatter("---\n---\nContent")
        assert result == {}

    def test_frontmatter_with_quoted_values(self):
        """Values with quotes should have quotes stripped."""
        content = "---\ntitle: \"My Quoted Page\"\nauthor: 'Jane Doe'\n---\n"
        result = _parse_simple_frontmatter(content)
        assert result["title"] == "My Quoted Page"
        assert result["author"] == "Jane Doe"

    def test_frontmatter_ignores_comments(self):
        """Lines starting with # inside frontmatter should be skipped."""
        content = "---\ntitle: Page\n# This is a comment\nweight: 10\n---\n"
        result = _parse_simple_frontmatter(content)
        assert "title" in result
        assert "weight" in result
        assert len(result) == 2

    def test_frontmatter_ignores_empty_lines(self):
        """Empty lines inside frontmatter should be skipped."""
        content = "---\ntitle: Page\n\nweight: 10\n---\n"
        result = _parse_simple_frontmatter(content)
        assert result["title"] == "Page"
        assert result["weight"] == "10"

    def test_frontmatter_skips_empty_key_or_value(self):
        """Lines with empty key or value should be skipped."""
        content = "---\ntitle: Page\n: empty_key\nempty_value:\n---\n"
        result = _parse_simple_frontmatter(content)
        assert "title" in result
        # ": empty_key" has empty key, should be skipped
        # "empty_value:" has empty value, should be skipped
        assert len(result) == 1

    def test_frontmatter_value_with_colon(self):
        """Values containing colons should be handled correctly."""
        content = "---\ntitle: My Page: A Subtitle\n---\n"
        result = _parse_simple_frontmatter(content)
        assert result["title"] == "My Page: A Subtitle"

    def test_frontmatter_not_at_start(self):
        """Frontmatter not at the very start should not be parsed."""
        content = "\n---\ntitle: Page\n---\n"
        result = _parse_simple_frontmatter(content)
        assert result == {}

    def test_empty_content(self):
        """Empty string should return empty dict."""
        result = _parse_simple_frontmatter("")
        assert result == {}


# ============================================================
# NavResolverPipeline._extract_frontmatter_context
# ============================================================


class TestExtractFrontmatterContext:
    """Tests for NavResolverPipeline._extract_frontmatter_context."""

    def _make_doc(self, doc_path: str, content: str, content_type: str = "markdown") -> FetchedDocument:
        """Create a FetchedDocument for testing."""
        return FetchedDocument(
            platform="test",
            version="v1.0.0",
            source_url=f"https://example.com/{doc_path}",
            source_type="github",
            doc_path=doc_path,
            raw_content=content,
            content_type=content_type,
        )

    def test_extract_markdown_frontmatter(self):
        """Markdown documents should have frontmatter extracted."""
        docs = [
            self._make_doc(
                "docs/_index.md",
                "---\ntitle: Getting Started\nweight: 100\n---\n# Content",
            ),
        ]
        context = NavResolverPipeline._extract_frontmatter_context(docs)
        assert "docs/_index.md" in context
        assert context["docs/_index.md"]["title"] == "Getting Started"
        assert context["docs/_index.md"]["weight"] == "100"

    def test_extract_rst_frontmatter(self):
        """RST documents should also have frontmatter extracted."""
        docs = [
            self._make_doc(
                "docs/guide.rst",
                "---\ntitle: Guide\n---\nSome RST content",
                content_type="rst",
            ),
        ]
        context = NavResolverPipeline._extract_frontmatter_context(docs)
        assert "docs/guide.rst" in context
        assert context["docs/guide.rst"]["title"] == "Guide"

    def test_skip_html_documents(self):
        """HTML documents should be skipped (not markdown/rst)."""
        docs = [
            self._make_doc(
                "docs/page.html",
                "---\ntitle: HTML Page\n---\n<html></html>",
                content_type="html",
            ),
        ]
        context = NavResolverPipeline._extract_frontmatter_context(docs)
        assert context == {}

    def test_skip_doc_without_frontmatter(self):
        """Documents without frontmatter should not appear in context."""
        docs = [
            self._make_doc("docs/plain.md", "# Just a heading\nNo frontmatter here."),
        ]
        context = NavResolverPipeline._extract_frontmatter_context(docs)
        assert context == {}

    def test_multiple_docs(self):
        """Multiple documents should all be processed."""
        docs = [
            self._make_doc("docs/a.md", "---\ntitle: Page A\n---\n"),
            self._make_doc("docs/b.md", "---\ntitle: Page B\n---\n"),
            self._make_doc("docs/c.md", "# No frontmatter"),
        ]
        context = NavResolverPipeline._extract_frontmatter_context(docs)
        assert len(context) == 2
        assert context["docs/a.md"]["title"] == "Page A"
        assert context["docs/b.md"]["title"] == "Page B"

    def test_empty_docs_list(self):
        """Empty document list should return empty context."""
        context = NavResolverPipeline._extract_frontmatter_context([])
        assert context == {}


# ============================================================
# NavResolverPipeline instantiation
# ============================================================


class TestNavResolverPipelineInit:
    """Tests for NavResolverPipeline that do not require GitHub API."""

    def test_pipeline_can_be_instantiated(self):
        """NavResolverPipeline should be instantiable without arguments."""
        pipeline = NavResolverPipeline()
        assert pipeline is not None
