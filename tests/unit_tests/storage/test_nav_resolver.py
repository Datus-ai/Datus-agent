# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the nav_resolver module."""

import json

from datus.storage.document.nav_resolver.docusaurus_resolver import DocusaurusResolver
from datus.storage.document.nav_resolver.fallback_resolver import FallbackResolver
from datus.storage.document.nav_resolver.hugo_resolver import HugoResolver
from datus.storage.document.nav_resolver.mkdocs_resolver import MkDocsResolver

# =============================================================================
# FallbackResolver Tests
# =============================================================================


class TestFallbackResolver:
    def setup_method(self):
        self.resolver = FallbackResolver()

    def test_simple_path(self):
        result = self.resolver.resolve(
            config_content="",
            file_paths=["docs/en/sql-reference/ddl/CREATE_TABLE.md"],
            content_root="docs/en/",
        )
        assert result["docs/en/sql-reference/ddl/CREATE_TABLE.md"] == ["Sql Reference", "Ddl"]

    def test_strips_content_root(self):
        result = self.resolver.resolve("", ["docs/en/intro.md"], "docs/en/")
        assert result["docs/en/intro.md"] == []

    def test_skip_boilerplate_dirs(self):
        result = self.resolver.resolve("", ["docs/en/content/guides/setup.md"], "")
        # "docs", "en", "content" are all in _SKIP_SEGMENTS
        assert result["docs/en/content/guides/setup.md"] == ["Guides"]

    def test_preserves_uppercase(self):
        result = self.resolver.resolve("", ["API/reference/endpoints.md"], "")
        assert result["API/reference/endpoints.md"] == ["API", "Reference"]

    def test_empty_path(self):
        result = self.resolver.resolve("", ["README.md"], "")
        assert result["README.md"] == []

    def test_multiple_files(self):
        files = [
            "docs/en/loading/stream-load.md",
            "docs/en/loading/broker-load.md",
        ]
        result = self.resolver.resolve("", files, "docs/en/")
        assert result["docs/en/loading/stream-load.md"] == ["Loading"]
        assert result["docs/en/loading/broker-load.md"] == ["Loading"]

    def test_clean_segment_hyphens(self):
        assert FallbackResolver._clean_segment("sql-reference") == "Sql Reference"

    def test_clean_segment_underscores(self):
        assert FallbackResolver._clean_segment("getting_started") == "Getting Started"

    def test_clean_segment_all_caps(self):
        assert FallbackResolver._clean_segment("DDL") == "DDL"


# =============================================================================
# DocusaurusResolver Tests
# =============================================================================


class TestDocusaurusResolver:
    def setup_method(self):
        self.resolver = DocusaurusResolver()

    def test_simple_sidebar(self):
        sidebars = {
            "docs": [
                {
                    "type": "category",
                    "label": "Getting Started",
                    "items": ["intro", "quick-start"],
                },
                {
                    "type": "category",
                    "label": "SQL Reference",
                    "items": [
                        {
                            "type": "category",
                            "label": "DDL",
                            "items": ["sql-reference/ddl/CREATE_TABLE"],
                        }
                    ],
                },
            ]
        }
        config = json.dumps(sidebars)
        files = [
            "docs/en/intro.md",
            "docs/en/quick-start.md",
            "docs/en/sql-reference/ddl/CREATE_TABLE.md",
        ]
        result = self.resolver.resolve(config, files, "docs/en/")

        assert result["docs/en/intro.md"] == ["Getting Started"]
        assert result["docs/en/quick-start.md"] == ["Getting Started"]
        assert result["docs/en/sql-reference/ddl/CREATE_TABLE.md"] == ["SQL Reference", "DDL"]

    def test_doc_object_items(self):
        sidebars = {
            "docs": [
                {
                    "type": "category",
                    "label": "Guides",
                    "items": [
                        {"type": "doc", "id": "guides/overview", "label": "Overview"},
                        {"type": "doc", "id": "guides/setup"},
                    ],
                }
            ]
        }
        config = json.dumps(sidebars)
        files = ["docs/guides/overview.md", "docs/guides/setup.md"]
        result = self.resolver.resolve(config, files, "docs/")

        assert result["docs/guides/overview.md"] == ["Guides"]
        assert result["docs/guides/setup.md"] == ["Guides"]

    def test_invalid_json(self):
        result = self.resolver.resolve("not json", ["file.md"], "")
        assert result == {}

    def test_empty_config(self):
        result = self.resolver.resolve("", ["file.md"], "")
        assert result == {}

    def test_link_items_skipped(self):
        sidebars = {
            "docs": [
                {"type": "link", "label": "External", "href": "https://example.com"},
                "intro",
            ]
        }
        config = json.dumps(sidebars)
        result = self.resolver.resolve(config, ["docs/intro.md"], "docs/")
        assert result["docs/intro.md"] == []


# =============================================================================
# MkDocsResolver Tests
# =============================================================================


class TestMkDocsResolver:
    def setup_method(self):
        self.resolver = MkDocsResolver()

    def test_simple_nav(self):
        config = """
site_name: My Docs
docs_dir: docs
nav:
  - Home: index.md
  - User Guide:
    - Getting Started: user-guide/getting-started.md
    - Configuration: user-guide/configuration.md
  - API Reference:
    - Overview: api/overview.md
"""
        files = [
            "docs/index.md",
            "docs/user-guide/getting-started.md",
            "docs/user-guide/configuration.md",
            "docs/api/overview.md",
        ]
        result = self.resolver.resolve(config, files, "docs/")

        assert result["docs/index.md"] == []
        assert result["docs/user-guide/getting-started.md"] == ["User Guide"]
        assert result["docs/user-guide/configuration.md"] == ["User Guide"]
        assert result["docs/api/overview.md"] == ["API Reference"]

    def test_nested_nav(self):
        config = """
nav:
  - SQL Reference:
    - DDL:
      - CREATE TABLE: sql/ddl/create-table.md
      - ALTER TABLE: sql/ddl/alter-table.md
"""
        files = ["docs/sql/ddl/create-table.md", "docs/sql/ddl/alter-table.md"]
        result = self.resolver.resolve(config, files, "docs/")

        assert result["docs/sql/ddl/create-table.md"] == ["SQL Reference", "DDL"]
        assert result["docs/sql/ddl/alter-table.md"] == ["SQL Reference", "DDL"]

    def test_no_nav_section(self):
        config = """
site_name: My Docs
theme:
  name: material
"""
        result = self.resolver.resolve(config, ["docs/index.md"], "docs/")
        assert result == {}

    def test_invalid_yaml(self):
        result = self.resolver.resolve("not: valid: yaml: {{", ["file.md"], "")
        assert result == {}

    def test_empty_config(self):
        result = self.resolver.resolve("", ["file.md"], "")
        assert result == {}


# =============================================================================
# HugoResolver Tests
# =============================================================================


class TestHugoResolver:
    def setup_method(self):
        self.resolver = HugoResolver()

    def test_with_index_titles(self):
        files = [
            "site/content/getting-started/_index.md",
            "site/content/getting-started/quick-start.md",
            "site/content/getting-started/install.md",
        ]
        extra = {
            "site/content/getting-started/_index.md": {"title": "Getting Started", "weight": "100"},
        }
        result = self.resolver.resolve("", files, "site/content/", extra)

        assert result["site/content/getting-started/quick-start.md"] == ["Getting Started"]
        assert result["site/content/getting-started/install.md"] == ["Getting Started"]
        # _index.md itself gets parent nav_path (empty since it's top-level)
        assert result["site/content/getting-started/_index.md"] == []

    def test_nested_sections(self):
        files = [
            "content/security/_index.md",
            "content/security/auth/_index.md",
            "content/security/auth/oauth.md",
        ]
        extra = {
            "content/security/_index.md": {"title": "Security"},
            "content/security/auth/_index.md": {"title": "Authentication"},
        }
        result = self.resolver.resolve("", files, "content/", extra)

        assert result["content/security/auth/oauth.md"] == ["Security", "Authentication"]
        assert result["content/security/auth/_index.md"] == ["Security"]
        assert result["content/security/_index.md"] == []

    def test_fallback_to_dir_name(self):
        """When no _index.md exists, use cleaned directory name."""
        files = ["content/my-guide/intro.md"]
        result = self.resolver.resolve("", files, "content/", {})

        assert result["content/my-guide/intro.md"] == ["My Guide"]

    def test_linktitle_preferred(self):
        """linkTitle should be preferred over title."""
        files = ["content/docs/_index.md", "content/docs/page.md"]
        extra = {
            "content/docs/_index.md": {"title": "Full Documentation Title", "linkTitle": "Docs"},
        }
        result = self.resolver.resolve("", files, "content/", extra)
        assert result["content/docs/page.md"] == ["Docs"]

    def test_empty_files(self):
        result = self.resolver.resolve("", [], "content/", {})
        assert result == {}


# =============================================================================
# NavResolverPipeline Tests
# =============================================================================


class TestNavResolverPipeline:
    def test_frontmatter_extraction(self):
        """Test _parse_simple_frontmatter helper."""
        from datus.storage.document.nav_resolver import _parse_simple_frontmatter

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

    def test_frontmatter_no_frontmatter(self):
        from datus.storage.document.nav_resolver import _parse_simple_frontmatter

        result = _parse_simple_frontmatter("# Just a heading\nSome content")
        assert result == {}

    def test_frontmatter_empty(self):
        from datus.storage.document.nav_resolver import _parse_simple_frontmatter

        result = _parse_simple_frontmatter("---\n---\nContent")
        assert result == {}
