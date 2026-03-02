# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for MarkdownParser."""

from datus.storage.document.parser.markdown_parser import MarkdownParser
from datus.storage.document.schemas import FetchedDocument


def _make_doc(raw_content: str, doc_path: str = "docs/test.md") -> FetchedDocument:
    """Create a minimal FetchedDocument for testing."""
    return FetchedDocument(
        platform="test",
        version="1.0",
        source_url="https://example.com/docs/test.md",
        source_type="github",
        doc_path=doc_path,
        raw_content=raw_content,
        content_type="markdown",
    )


# ---------------------------------------------------------------------------
# Frontmatter extraction
# ---------------------------------------------------------------------------


class TestExtractFrontmatter:
    """Tests for _extract_frontmatter."""

    def test_extract_frontmatter_basic_yaml(self):
        """Frontmatter with simple key-value pairs is extracted correctly."""
        parser = MarkdownParser()
        content = '---\ntitle: "My Title"\nauthor: John\n---\n\nBody text.'
        metadata, remaining = parser._extract_frontmatter(content)

        assert metadata["title"] == "My Title"
        assert metadata["author"] == "John"
        assert "Body text." in remaining

    def test_extract_frontmatter_no_frontmatter(self):
        """Content without frontmatter returns empty metadata and unchanged content."""
        parser = MarkdownParser()
        content = "# Hello\n\nSome text."
        metadata, remaining = parser._extract_frontmatter(content)

        assert metadata == {}
        assert remaining == content

    def test_extract_frontmatter_nav_hints(self):
        """Nav-hint keys (sidebar_position, weight, slug, etc.) populate _nav_hints."""
        parser = MarkdownParser()
        content = "---\ntitle: Page\nsidebar_position: 3\nweight: 10\nslug: my-page\n---\n\nContent."
        metadata, _ = parser._extract_frontmatter(content)

        assert "_nav_hints" in metadata
        assert metadata["_nav_hints"]["sidebar_position"] == "3"
        assert metadata["_nav_hints"]["weight"] == "10"
        assert metadata["_nav_hints"]["slug"] == "my-page"

    def test_extract_frontmatter_no_nav_hints(self):
        """When no nav-hint keys present, _nav_hints is not added."""
        parser = MarkdownParser()
        content = "---\ntitle: Page\nauthor: Alice\n---\n\nContent."
        metadata, _ = parser._extract_frontmatter(content)

        assert "_nav_hints" not in metadata

    def test_extract_frontmatter_quoted_values(self):
        """Single and double quoted values in frontmatter are stripped of quotes."""
        parser = MarkdownParser()
        content = "---\ntitle: 'Single Quoted'\ndesc: \"Double Quoted\"\n---\n\nBody."
        metadata, _ = parser._extract_frontmatter(content)

        assert metadata["title"] == "Single Quoted"
        assert metadata["desc"] == "Double Quoted"

    def test_extract_frontmatter_incomplete_fence(self):
        """Content starting with --- but missing closing --- is not treated as frontmatter."""
        parser = MarkdownParser()
        content = "---\ntitle: Test\nSome normal content without closing fence."
        metadata, remaining = parser._extract_frontmatter(content)

        # Without a closing ---, split("---", 2) yields only 2 parts
        assert metadata == {}


# ---------------------------------------------------------------------------
# parse() integration tests
# ---------------------------------------------------------------------------


class TestParseBasic:
    """Tests for parse() with various markdown content."""

    def test_parse_simple_headings(self):
        """Document with h1 and h2 headings produces correct hierarchy."""
        parser = MarkdownParser()
        doc = _make_doc("# Title\n\nIntro text.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B.")
        result = parser.parse(doc)

        assert result.title == "Title"
        titles = result.get_section_titles()
        assert "Title" in titles
        assert "Section A" in titles
        assert "Section B" in titles

    def test_parse_nested_sections(self):
        """Nested headings (h1 > h2 > h3) create a child hierarchy."""
        parser = MarkdownParser()
        doc = _make_doc("# Top\n\n## Mid\n\n### Deep\n\nDeep content.")
        result = parser.parse(doc)

        assert result.title == "Top"
        # Top section should have Mid as child
        top = result.sections[0]
        assert top.title == "Top"
        assert len(top.children) >= 1
        mid = top.children[0]
        assert mid.title == "Mid"
        assert len(mid.children) >= 1
        assert mid.children[0].title == "Deep"

    def test_parse_code_blocks_preserved(self):
        """Fenced code blocks are preserved in section content."""
        parser = MarkdownParser()
        md = "# Code Example\n\n```python\nprint('hello')\n```\n"
        doc = _make_doc(md)
        result = parser.parse(doc)

        section = result.sections[0]
        all_content = section.get_all_content()
        assert "```python" in all_content
        assert "print('hello')" in all_content

    def test_parse_title_from_frontmatter(self):
        """Title from frontmatter takes precedence over first h1."""
        parser = MarkdownParser()
        md = "---\ntitle: FM Title\n---\n\n# Heading Title\n\nBody."
        doc = _make_doc(md)
        result = parser.parse(doc)

        assert result.title == "FM Title"

    def test_parse_title_fallback_to_doc_path(self):
        """When no h1 and no frontmatter title, title falls back to doc_path."""
        parser = MarkdownParser()
        doc = _make_doc("## Only h2\n\nSome text.", doc_path="docs/my-page.md")
        result = parser.parse(doc)

        # Fallback logic: doc_path.split("/")[-1].replace(".md","").replace("-"," ").title()
        assert result.title == "My Page"

    def test_parse_content_before_first_heading(self):
        """Content before the first heading is included in the first section's content."""
        parser = MarkdownParser()
        md = "Some preamble text.\n\n# First Heading\n\nBody."
        doc = _make_doc(md)
        result = parser.parse(doc)

        # markdown-it parser attaches preamble to the first heading section
        first_section = result.sections[0]
        assert "preamble" in first_section.content.lower() or "preamble" in first_section.get_all_content().lower()

    def test_parse_content_only_no_heading(self):
        """Content without any heading creates a level-0 section."""
        parser = MarkdownParser()
        md = "Just some text without any heading.\n\nAnother paragraph."
        doc = _make_doc(md, doc_path="docs/no-heading.md")
        result = parser.parse(doc)

        assert len(result.sections) == 1
        assert result.sections[0].level == 0
        assert "text without any heading" in result.sections[0].content

    def test_parse_multiple_heading_levels(self):
        """Document with h1, h2, h3, h4 levels is parsed correctly."""
        parser = MarkdownParser()
        md = "# H1\n\n## H2\n\n### H3\n\n#### H4\n\nContent at h4."
        doc = _make_doc(md)
        result = parser.parse(doc)

        titles = result.get_section_titles()
        assert titles == ["H1", "H2", "H3", "H4"]

    def test_parse_source_doc_reference(self):
        """Parsed document keeps a reference to the source FetchedDocument."""
        parser = MarkdownParser()
        doc = _make_doc("# Test\n\nContent.")
        result = parser.parse(doc)

        assert result.source_doc is doc

    def test_parse_metadata_from_frontmatter(self):
        """Metadata from frontmatter is available in parsed result."""
        parser = MarkdownParser()
        md = "---\ntitle: Meta Title\nauthor: Bob\n---\n\n# Heading\n\nBody."
        doc = _make_doc(md)
        result = parser.parse(doc)

        assert result.metadata["author"] == "Bob"


# ---------------------------------------------------------------------------
# _parse_with_regex fallback
# ---------------------------------------------------------------------------


class TestParseWithRegex:
    """Tests for the regex fallback parser."""

    def test_regex_simple_headings(self):
        """Regex parser extracts heading hierarchy correctly."""
        parser = MarkdownParser()
        content = "# Title\n\nIntro.\n\n## Section\n\nSection text."
        sections = parser._parse_with_regex(content)

        assert len(sections) >= 1
        assert sections[0].title == "Title"

    def test_regex_content_before_heading(self):
        """Regex parser captures content before the first heading as level-0."""
        parser = MarkdownParser()
        content = "Preamble text.\n\n# First\n\nBody."
        sections = parser._parse_with_regex(content)

        assert sections[0].level == 0
        assert "Preamble" in sections[0].content

    def test_regex_nested_sections(self):
        """Regex parser nests lower-level headings under higher-level ones."""
        parser = MarkdownParser()
        content = "# Parent\n\nParent content.\n\n## Child\n\nChild content."
        sections = parser._parse_with_regex(content)

        parent = sections[0]
        assert parent.title == "Parent"
        assert len(parent.children) == 1
        assert parent.children[0].title == "Child"

    def test_regex_remaining_content_appended(self):
        """Remaining content after last heading is appended to that section."""
        parser = MarkdownParser()
        content = "# Section\n\nMiddle text.\n\nTrailing text."
        sections = parser._parse_with_regex(content)

        assert "Trailing text" in sections[0].content

    def test_regex_multiple_same_level_headings(self):
        """Multiple h2 sections at the same level are sibling children under h1."""
        parser = MarkdownParser()
        content = "# Top\n\n## A\n\nContent A.\n\n## B\n\nContent B."
        sections = parser._parse_with_regex(content)

        top = sections[0]
        assert top.title == "Top"
        assert len(top.children) == 2
        assert top.children[0].title == "A"
        assert top.children[1].title == "B"

    def test_regex_empty_content(self):
        """Regex parser handles empty content gracefully."""
        parser = MarkdownParser()
        sections = parser._parse_with_regex("")
        assert sections == []


# ---------------------------------------------------------------------------
# Lists and tables (via markdown-it)
# ---------------------------------------------------------------------------


class TestListsAndTables:
    """Tests for list and table extraction during parse."""

    def test_parse_unordered_list(self):
        """Unordered list items appear in section content."""
        parser = MarkdownParser()
        md = "# List Test\n\n- Item A\n- Item B\n- Item C\n"
        doc = _make_doc(md)
        result = parser.parse(doc)

        content = result.sections[0].get_all_content()
        assert "Item A" in content
        assert "Item B" in content

    def test_parse_ordered_list(self):
        """Ordered list items appear with numeric prefixes."""
        parser = MarkdownParser()
        md = "# Ordered\n\n1. First\n2. Second\n3. Third\n"
        doc = _make_doc(md)
        result = parser.parse(doc)

        content = result.sections[0].get_all_content()
        assert "First" in content
        assert "Second" in content

    def test_parse_table(self):
        """Markdown table is extracted and preserved."""
        parser = MarkdownParser()
        md = "# Table Test\n\n| Name | Value |\n|------|-------|\n| A    | 1     |\n| B    | 2     |\n"
        doc = _make_doc(md)
        result = parser.parse(doc)

        content = result.sections[0].get_all_content()
        assert "Name" in content
        assert "Value" in content
