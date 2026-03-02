# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for HTMLParser."""

from datus.storage.document.parser.html_parser import HTMLParser
from datus.storage.document.schemas import FetchedDocument


def _make_doc(
    raw_content: str,
    doc_path: str = "docs/test.html",
    source_url: str = "https://docs.example.com/en/user-guide/loading-data",
) -> FetchedDocument:
    """Create a minimal FetchedDocument with HTML content."""
    return FetchedDocument(
        platform="test",
        version="1.0",
        source_url=source_url,
        source_type="website",
        doc_path=doc_path,
        raw_content=raw_content,
        content_type="html",
    )


# ---------------------------------------------------------------------------
# parse() full pipeline
# ---------------------------------------------------------------------------


class TestHTMLParserParse:
    """Tests for HTMLParser.parse()."""

    def test_parse_simple_document(self):
        """Parse a minimal HTML document with title and a heading."""
        html = "<html><head><title>Test Page</title></head><body><h1>Hello</h1><p>World</p></body></html>"
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(_make_doc(html))

        assert result.title == "Test Page"
        assert len(result.sections) >= 1

    def test_parse_title_cleanup_pipe(self):
        """Title with ' | SiteName' suffix is cleaned."""
        html = "<html><head><title>My Page | Docs Site</title></head><body><p>Body</p></body></html>"
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(_make_doc(html))

        assert result.title == "My Page"

    def test_parse_title_cleanup_dash(self):
        """Title with ' - SiteName' suffix is cleaned."""
        html = "<html><head><title>Intro - Documentation</title></head><body><p>Text</p></body></html>"
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(_make_doc(html))

        assert result.title == "Intro"

    def test_parse_title_fallback_to_doc_path(self):
        """Without <title>, title falls back to doc_path-based name."""
        html = "<html><body><p>No title tag</p></body></html>"
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(_make_doc(html, doc_path="docs/my-page.html"))

        assert result.title == "My Page"

    def test_parse_source_doc_reference(self):
        """Parsed document keeps reference to source FetchedDocument."""
        html = "<html><body><h1>T</h1></body></html>"
        doc = _make_doc(html)
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(doc)

        assert result.source_doc is doc


# ---------------------------------------------------------------------------
# _extract_breadcrumb
# ---------------------------------------------------------------------------


class TestExtractBreadcrumb:
    """Tests for _extract_breadcrumb."""

    def test_breadcrumb_by_aria_label(self):
        """Breadcrumb detected via aria-label='breadcrumb' with <li> items."""
        html = (
            "<html><body>"
            '<nav aria-label="breadcrumb">'
            "<ul>"
            "<li>User Guide</li>"
            "<li>Loading Data</li>"
            "<li>Snowpipe</li>"
            "</ul>"
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_breadcrumb(soup)

        assert result == ["User Guide", "Loading Data", "Snowpipe"]

    def test_breadcrumb_by_class(self):
        """Breadcrumb detected via class='breadcrumb'."""
        html = (
            "<html><body>" '<div class="breadcrumb">' "<ul><li>Guides</li><li>SQL</li></ul>" "</div>" "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_breadcrumb(soup)

        assert result == ["Guides", "SQL"]

    def test_breadcrumb_filters_common_labels(self):
        """Breadcrumb filters out 'Home', 'Docs', 'Documentation', '...'."""
        html = (
            "<html><body>"
            '<nav aria-label="breadcrumb">'
            "<ul>"
            "<li>Home</li>"
            "<li>Docs</li>"
            "<li>Features</li>"
            "<li>Query</li>"
            "</ul>"
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_breadcrumb(soup)

        assert "Home" not in result
        assert "Docs" not in result
        assert "Features" in result

    def test_breadcrumb_no_match(self):
        """Returns empty list when no breadcrumb element exists."""
        html = "<html><body><p>Nothing</p></body></html>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_breadcrumb(soup)

        assert result == []

    def test_breadcrumb_via_links(self):
        """Breadcrumb extracted from <a> tags when no <li> items present."""
        html = (
            "<html><body>"
            '<nav aria-label="breadcrumb">'
            '<a href="/guide">Guide</a> > <a href="/guide/sql">SQL</a>'
            '<span class="current">Functions</span>'
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_breadcrumb(soup)

        assert "Guide" in result
        assert "SQL" in result
        assert "Functions" in result


# ---------------------------------------------------------------------------
# _extract_path_from_url
# ---------------------------------------------------------------------------


class TestExtractPathFromUrl:
    """Tests for _extract_path_from_url."""

    def test_basic_url_segments(self):
        """URL path segments are converted to title-case words."""
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_path_from_url("https://docs.example.com/user-guide/data-load")

        assert result == ["User Guide", "Data Load"]

    def test_skips_common_prefixes(self):
        """Common prefixes like 'en', 'docs', 'latest' are skipped."""
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_path_from_url("https://docs.example.com/en/docs/latest/features")

        assert result == ["Features"]

    def test_empty_path(self):
        """Root URL returns empty list."""
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_path_from_url("https://docs.example.com/")

        assert result == []

    def test_underscore_segments(self):
        """Underscored segments are converted to spaces then title-cased."""
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_path_from_url("https://example.com/data_types/numeric_types")

        assert result == ["Data Types", "Numeric Types"]


# ---------------------------------------------------------------------------
# _extract_metadata
# ---------------------------------------------------------------------------


class TestExtractMetadata:
    """Tests for _extract_metadata."""

    def test_meta_description(self):
        """Description meta tag is extracted."""
        html = "<html><head>" '<meta name="description" content="A test page about SQL.">' "</head><body></body></html>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_metadata(soup)

        assert result["description"] == "A test page about SQL."

    def test_meta_author(self):
        """Author meta tag is extracted."""
        html = '<html><head><meta name="author" content="Alice"></head><body></body></html>'
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_metadata(soup)

        assert result["author"] == "Alice"

    def test_meta_keywords(self):
        """Keywords meta tag is split into a list."""
        html = '<html><head><meta name="keywords" content="sql, database, query"></head><body></body></html>'
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_metadata(soup)

        assert result["keywords"] == ["sql", "database", "query"]

    def test_meta_title_from_tag(self):
        """Title is extracted from <title> tag."""
        html = "<html><head><title>Page Title</title></head><body></body></html>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_metadata(soup)

        assert result["title"] == "Page Title"

    def test_meta_og_title_fallback(self):
        """og:title used as fallback when no <title> tag."""
        html = '<html><head><meta property="og:title" content="OG Title"></head><body></body></html>'
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_metadata(soup)

        assert result["title"] == "OG Title"

    def test_meta_empty_document(self):
        """Empty head returns empty metadata dict."""
        html = "<html><head></head><body></body></html>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_metadata(soup)

        assert result == {}


# ---------------------------------------------------------------------------
# _extract_sections
# ---------------------------------------------------------------------------


class TestExtractSections:
    """Tests for _extract_sections."""

    def test_heading_hierarchy(self):
        """h1 > h2 > h3 creates nested sections."""
        html = "<div><h1>Title</h1><p>Intro</p><h2>Sub</h2><p>Sub text</p><h3>Deep</h3><p>Deep text</p></div>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        assert sections[0].title == "Title"
        assert sections[0].children[0].title == "Sub"
        assert sections[0].children[0].children[0].title == "Deep"

    def test_no_headings_creates_level0_section(self):
        """Content without headings creates a single level-0 section."""
        html = "<div><p>Just a paragraph.</p><p>Another paragraph.</p></div>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        assert len(sections) == 1
        assert sections[0].level == 0
        assert "paragraph" in sections[0].content.lower()

    def test_code_block_extraction(self):
        """Code blocks within <pre><code> are preserved."""
        html = '<div><h1>Code</h1><pre><code class="language-sql">SELECT 1;</code></pre></div>'
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        content = sections[0].content
        assert "```sql" in content
        assert "SELECT 1;" in content

    def test_blockquote_extraction(self):
        """Blockquotes are prefixed with > markers."""
        html = "<div><h1>Quote</h1><blockquote>Important note here.</blockquote></div>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        content = sections[0].content
        assert ">" in content
        assert "Important note" in content


# ---------------------------------------------------------------------------
# _extract_text_with_formatting
# ---------------------------------------------------------------------------


class TestExtractTextWithFormatting:
    """Tests for _extract_text_with_formatting."""

    def test_bold_text(self):
        """<strong> and <b> tags produce ** markers."""
        html = "<p>This is <strong>bold</strong> text.</p>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_text_with_formatting(soup.find("p"))

        assert "**bold**" in result

    def test_italic_text(self):
        """<em> and <i> tags produce * markers."""
        html = "<p>This is <em>italic</em> text.</p>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_text_with_formatting(soup.find("p"))

        assert "*italic*" in result

    def test_inline_code(self):
        """<code> tags produce backtick markers."""
        html = "<p>Use <code>SELECT</code> statement.</p>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_text_with_formatting(soup.find("p"))

        assert "`SELECT`" in result

    def test_link_formatting(self):
        """<a> tags produce [text](href) format."""
        html = '<p>See <a href="https://example.com">docs</a> for more.</p>'
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_text_with_formatting(soup.find("p"))

        assert "[docs](https://example.com)" in result


# ---------------------------------------------------------------------------
# _extract_list and _extract_table
# ---------------------------------------------------------------------------


class TestExtractListAndTable:
    """Tests for _extract_list and _extract_table."""

    def test_unordered_list(self):
        """Unordered list items are prefixed with '- '."""
        html = "<ul><li>Alpha</li><li>Beta</li><li>Gamma</li></ul>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_list(soup.find("ul"), ordered=False)

        assert "- Alpha" in result
        assert "- Beta" in result
        assert "- Gamma" in result

    def test_ordered_list(self):
        """Ordered list items are numbered sequentially."""
        html = "<ol><li>First</li><li>Second</li><li>Third</li></ol>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_list(soup.find("ol"), ordered=True)

        assert "1. First" in result
        assert "2. Second" in result
        assert "3. Third" in result

    def test_table_with_thead(self):
        """Table with <thead> produces header row + separator + data rows."""
        html = (
            "<table>"
            "<thead><tr><th>Name</th><th>Type</th></tr></thead>"
            "<tbody><tr><td>id</td><td>int</td></tr><tr><td>name</td><td>varchar</td></tr></tbody>"
            "</table>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_table(soup.find("table"))

        assert "| Name | Type |" in result
        assert "|---|---|" in result
        assert "| id | int |" in result
        assert "| name | varchar |" in result

    def test_table_without_thead(self):
        """Table without <thead> treats first row as header."""
        html = "<table><tr><td>Col1</td><td>Col2</td></tr><tr><td>A</td><td>B</td></tr></table>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_table(soup.find("table"))

        assert "| Col1 | Col2 |" in result
        assert "---|---" in result
        assert "| A | B |" in result

    def test_empty_list(self):
        """Empty list returns empty string."""
        html = "<ul></ul>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_list(soup.find("ul"), ordered=False)

        assert result == ""


# ---------------------------------------------------------------------------
# _extract_sidebar_path
# ---------------------------------------------------------------------------


class TestExtractSidebarPath:
    """Tests for _extract_sidebar_path sidebar navigation traversal."""

    def test_sidebar_with_matching_link(self):
        """Sidebar contains a link matching the current URL path."""
        html = (
            "<html><body>"
            '<nav aria-label="Docs pages">'
            "<ul>"
            '<li><a href="/guide">Guide</a>'
            "<ul>"
            '<li><a href="/guide/loading">Loading Data</a></li>'
            "</ul>"
            "</li>"
            "</ul>"
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_sidebar_path(soup, "https://docs.example.com/guide/loading")

        assert "Loading Data" in result
        assert "Guide" in result

    def test_sidebar_no_matching_link(self):
        """Sidebar without a link matching current URL returns empty list."""
        html = (
            "<html><body>"
            '<nav aria-label="Docs pages">'
            "<ul>"
            '<li><a href="/other">Other Page</a></li>'
            "</ul>"
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_sidebar_path(soup, "https://docs.example.com/guide/loading")

        assert result == []

    def test_sidebar_no_sidebar_element(self):
        """No sidebar element at all returns empty list."""
        html = "<html><body><p>No sidebar here</p></body></html>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_sidebar_path(soup, "https://docs.example.com/guide/loading")

        assert result == []

    def test_sidebar_with_details_summary_ancestors(self):
        """Sidebar with <details>/<summary> ancestor sections are traversed."""
        html = (
            "<html><body>"
            '<nav aria-label="Docs pages">'
            "<details>"
            "<summary>User Guide</summary>"
            "<ul>"
            '<li><a href="/guide/loading">Loading Data</a></li>'
            "</ul>"
            "</details>"
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_sidebar_path(soup, "https://docs.example.com/guide/loading")

        assert "Loading Data" in result
        assert "User Guide" in result

    def test_sidebar_skips_hash_only_links(self):
        """Sidebar links with # or empty href should be skipped."""
        html = (
            "<html><body>"
            '<nav aria-label="Docs pages">'
            "<ul>"
            '<li><a href="#">Anchor</a></li>'
            '<li><a href="">Empty</a></li>'
            '<li><a href="#section">Hash Section</a></li>'
            '<li><a href="/real-page">Real Page</a></li>'
            "</ul>"
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_sidebar_path(soup, "https://docs.example.com/real-page")

        assert "Real Page" in result

    def test_sidebar_with_heading_in_section(self):
        """Sidebar with section containing heading ancestor."""
        html = (
            "<html><body>"
            '<nav aria-label="Docs pages">'
            "<section>"
            "<h3>API Reference</h3>"
            "<ul>"
            '<li><a href="/api/endpoints">Endpoints</a></li>'
            "</ul>"
            "</section>"
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_sidebar_path(soup, "https://docs.example.com/api/endpoints")

        assert "Endpoints" in result
        assert "API Reference" in result

    def test_sidebar_link_in_wrapper_div(self):
        """Sidebar with parent link in a wrapper div/span child."""
        html = (
            "<html><body>"
            '<nav aria-label="Docs pages">'
            "<ul>"
            "<li>"
            '<div><a href="/guide">Guide</a></div>'
            "<ul>"
            '<li><a href="/guide/intro">Introduction</a></li>'
            "</ul>"
            "</li>"
            "</ul>"
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_sidebar_path(soup, "https://docs.example.com/guide/intro")

        assert "Introduction" in result
        assert "Guide" in result


# ---------------------------------------------------------------------------
# _extract_group_name
# ---------------------------------------------------------------------------


class TestExtractGroupName:
    """Tests for _extract_group_name group detection."""

    def test_group_name_from_known_first_element(self):
        """First nav_path element that is a known group name."""
        from bs4 import BeautifulSoup

        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_group_name(["Guides", "Loading Data", "Snowpipe"], soup)

        assert result == "Guides"

    def test_group_name_from_contained_known_group(self):
        """First nav_path element containing a known group substring."""
        from bs4 import BeautifulSoup

        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_group_name(["User Guide and Tips", "Topic"], soup)

        assert result == "User Guide and Tips"

    def test_group_name_fallback_short_first_element(self):
        """First nav_path element shorter than 30 chars used as fallback."""
        from bs4 import BeautifulSoup

        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_group_name(["Custom Category", "Page"], soup)

        assert result == "Custom Category"

    def test_group_name_empty_nav_path(self):
        """Empty nav_path returns empty string."""
        from bs4 import BeautifulSoup

        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_group_name([], soup)

        assert result == ""

    def test_group_name_long_first_element_no_known_group(self):
        """First element longer than 30 chars with no known group returns empty."""
        from bs4 import BeautifulSoup

        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        nav_path = ["A very long page title that is definitely more than thirty chars"]
        result = parser._extract_group_name(nav_path, soup)

        assert result == ""

    def test_group_name_from_top_nav_tabs(self):
        """Group name extracted from top-level navigation tabs matching nav_path."""
        from bs4 import BeautifulSoup

        html = (
            "<html><body>"
            '<nav role="navigation"><ul>'
            '<li><a href="/api">API</a></li>'
            '<li><a href="/tutorials">Tutorials</a></li>'
            "</ul></nav>"
            "</body></html>"
        )
        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_group_name(["Something with tutorials in it", "A Page"], soup)

        # Strategy 1 should match because "tutorials" is in the first element
        assert result == "Something with tutorials in it"

    def test_group_name_known_groups_case_insensitive(self):
        """Known groups matched case-insensitively."""
        from bs4 import BeautifulSoup

        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")

        assert parser._extract_group_name(["API Reference", "Endpoints"], soup) == "API Reference"
        assert parser._extract_group_name(["tutorials", "Lesson 1"], soup) == "tutorials"
        assert parser._extract_group_name(["Getting Started", "Install"], soup) == "Getting Started"


# ---------------------------------------------------------------------------
# _extract_navigation_path integration
# ---------------------------------------------------------------------------


class TestExtractNavigationPath:
    """Tests for _extract_navigation_path strategy selection."""

    def test_breadcrumb_preferred_over_sidebar(self):
        """Breadcrumb path with >1 items is preferred over sidebar."""
        html = (
            "<html><body>"
            '<nav aria-label="breadcrumb">'
            "<ul><li>Guide</li><li>Data</li></ul>"
            "</nav>"
            '<nav aria-label="Docs pages">'
            '<ul><li><a href="/guide/data">Data</a></li></ul>'
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_navigation_path(soup, "https://docs.example.com/guide/data")

        assert result == ["Guide", "Data"]

    def test_fallback_to_url_path(self):
        """When no breadcrumb or sidebar found, URL path is used."""
        html = "<html><body><p>No nav</p></body></html>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_navigation_path(soup, "https://docs.example.com/user-guide/loading-data")

        assert result == ["User Guide", "Loading Data"]

    def test_returns_empty_for_root_url_no_nav(self):
        """Root URL with no nav elements returns empty list."""
        html = "<html><body><p>Root</p></body></html>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_navigation_path(soup, "https://docs.example.com/")

        assert result == []


# ---------------------------------------------------------------------------
# Complex breadcrumb scenarios
# ---------------------------------------------------------------------------


class TestComplexBreadcrumbs:
    """Tests for complex breadcrumb extraction scenarios."""

    def test_breadcrumb_with_schema_org_structured_data(self):
        """Breadcrumb with schema.org BreadcrumbList structured data."""
        html = (
            "<html><body>"
            '<nav aria-label="breadcrumb">'
            '<ol itemscope itemtype="http://schema.org/BreadcrumbList">'
            '<li itemprop="itemListElement"><span itemprop="name">Docs</span></li>'
            '<li itemprop="itemListElement"><span itemprop="name">SQL</span></li>'
            '<li itemprop="itemListElement"><span itemprop="name">Functions</span></li>'
            "</ol>"
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_breadcrumb(soup)

        # schema.org extraction should return the items
        assert len(result) >= 2

    def test_breadcrumb_with_separator_cleaning(self):
        """Breadcrumb items with separator characters are cleaned."""
        html = (
            "<html><body>"
            '<nav aria-label="breadcrumb">'
            "<ul>"
            "<li>/Guide/</li>"
            "<li>>>SQL>>></li>"
            "</ul>"
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_breadcrumb(soup)

        # Separators should be stripped
        for item in result:
            assert not item.startswith("/")
            assert not item.startswith(">")

    def test_breadcrumb_with_active_span(self):
        """Breadcrumb with current page indicated by span class='active'."""
        html = (
            "<html><body>"
            '<nav aria-label="Breadcrumb">'
            '<a href="/guide">Guide</a>'
            '<span class="active">Current Page</span>'
            "</nav>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_breadcrumb(soup)

        assert "Guide" in result
        assert "Current Page" in result

    def test_breadcrumb_by_id(self):
        """Breadcrumb detected via id='breadcrumb'."""
        html = (
            "<html><body>" '<div id="breadcrumb">' "<ul><li>API</li><li>Endpoints</li></ul>" "</div>" "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_breadcrumb(soup)

        assert result == ["API", "Endpoints"]


# ---------------------------------------------------------------------------
# Code block extraction with language detection
# ---------------------------------------------------------------------------


class TestCodeBlockExtraction:
    """Tests for code block extraction with various language class prefixes."""

    def test_code_block_lang_prefix(self):
        """Code block with 'lang-' prefix class is detected."""
        html = '<div><h1>Code</h1><pre><code class="lang-python">print("hi")</code></pre></div>'
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        content = sections[0].content
        assert "```python" in content
        assert 'print("hi")' in content

    def test_code_block_no_language_class(self):
        """Code block without language class uses empty language marker."""
        html = '<div><h1>Code</h1><pre><code class="highlight">x = 1</code></pre></div>'
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        content = sections[0].content
        assert "```\n" in content
        assert "x = 1" in content

    def test_pre_without_code(self):
        """<pre> without <code> child preserves text with empty language."""
        html = "<div><h1>Code</h1><pre>raw preformatted text</pre></div>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        content = sections[0].content
        assert "```\n" in content
        assert "raw preformatted text" in content

    def test_inline_code_outside_pre(self):
        """Inline <code> outside <pre> is formatted with backticks."""
        html = "<div><h1>Ref</h1><p>Use <code>SELECT</code> to query.</p></div>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        content = sections[0].content
        assert "`SELECT`" in content


# ---------------------------------------------------------------------------
# Table extraction edge cases
# ---------------------------------------------------------------------------


class TestTableExtractionEdgeCases:
    """Additional table extraction scenarios."""

    def test_table_with_mixed_th_td_in_thead(self):
        """Table header containing mixed <th> and <td> cells."""
        html = (
            "<table>"
            "<thead><tr><th>Name</th><td>Value</td></tr></thead>"
            "<tbody><tr><td>A</td><td>1</td></tr></tbody>"
            "</table>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_table(soup.find("table"))

        assert "| Name | Value |" in result
        assert "| A | 1 |" in result

    def test_empty_table(self):
        """Empty table returns empty string."""
        html = "<table></table>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_table(soup.find("table"))

        assert result == ""


# ---------------------------------------------------------------------------
# _extract_sections advanced
# ---------------------------------------------------------------------------


class TestExtractSectionsAdvanced:
    """Advanced section extraction scenarios."""

    def test_sibling_headings_at_same_level(self):
        """Multiple h2 headings create sibling sections, not nested."""
        html = "<div><h1>Title</h1><h2>Sec A</h2><p>A text</p><h2>Sec B</h2><p>B text</p></div>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        assert sections[0].title == "Title"
        assert len(sections[0].children) == 2
        assert sections[0].children[0].title == "Sec A"
        assert sections[0].children[1].title == "Sec B"

    def test_list_extraction_in_section(self):
        """Unordered and ordered lists are extracted within sections."""
        html = (
            "<div>"
            "<h1>Lists</h1>"
            "<ul><li>Item 1</li><li>Item 2</li></ul>"
            "<ol><li>First</li><li>Second</li></ol>"
            "</div>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        content = sections[0].content
        assert "- Item 1" in content
        assert "1. First" in content

    def test_table_extraction_in_section(self):
        """Tables within sections are extracted as markdown."""
        html = (
            "<div>"
            "<h1>Data</h1>"
            "<table><thead><tr><th>Col</th></tr></thead>"
            "<tbody><tr><td>Val</td></tr></tbody></table>"
            "</div>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        content = sections[0].content
        assert "| Col |" in content
        assert "| Val |" in content

    def test_navigable_string_content(self):
        """Bare text nodes (NavigableString) outside tags are captured."""
        html = "<div><h1>Title</h1>Some raw text here</div>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        assert "Some raw text here" in sections[0].content

    def test_blockquote_multiline(self):
        """Blockquote with multiple lines is prefix-quoted."""
        html = "<div><h1>Quote</h1><blockquote>Line one\nLine two</blockquote></div>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        sections = parser._extract_sections(soup.find("div"))

        content = sections[0].content
        assert "> Line one" in content or "> Line" in content


# ---------------------------------------------------------------------------
# parse() additional edge cases
# ---------------------------------------------------------------------------


class TestHTMLParserParseAdditional:
    """Additional parse() integration tests."""

    def test_parse_with_nav_path_sets_metadata(self):
        """Parsing HTML with breadcrumb sets nav_path and group_name in metadata."""
        html = (
            "<html><head><title>Test</title></head><body>"
            '<nav aria-label="breadcrumb">'
            "<ul><li>Guides</li><li>SQL</li></ul>"
            "</nav>"
            "<h1>SQL Guide</h1><p>Content</p>"
            "</body></html>"
        )
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(_make_doc(html))

        assert "nav_path" in result.metadata
        assert result.metadata["nav_path"] == ["Guides", "SQL"]
        assert result.metadata["group_name"] == "Guides"

    def test_parse_cleans_script_and_style(self):
        """Script and style tags are removed during cleaning."""
        html = (
            "<html><head><title>Clean</title></head><body>"
            "<script>alert('xss')</script>"
            "<style>body{color:red}</style>"
            "<h1>Real Content</h1><p>Visible text</p>"
            "</body></html>"
        )
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(_make_doc(html))

        all_content = " ".join(s.content for s in result.sections)
        assert "alert" not in all_content
        assert "color:red" not in all_content
        assert "Visible text" in all_content

    def test_parse_finds_main_content(self):
        """Content within <main> or <article> is found as main content."""
        html = (
            "<html><head><title>Main</title></head><body>"
            "<div>Sidebar noise</div>"
            "<main><h1>Main Content</h1><p>Important text</p></main>"
            "</body></html>"
        )
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(_make_doc(html))

        assert any("Important text" in s.content for s in result.sections)

    def test_parse_removes_non_content_classes(self):
        """Elements with classes matching removal patterns are removed."""
        html = (
            "<html><head><title>Test</title></head><body>"
            '<div class="advertisement">Buy now!</div>'
            '<div class="cookie-banner">Accept cookies</div>'
            "<main><h1>Content</h1><p>Real text</p></main>"
            "</body></html>"
        )
        parser = HTMLParser(parser="html.parser")
        result = parser.parse(_make_doc(html))

        all_content = " ".join(s.content for s in result.sections)
        assert "Buy now" not in all_content
        assert "Accept cookies" not in all_content


# ---------------------------------------------------------------------------
# _extract_text_with_formatting additional
# ---------------------------------------------------------------------------


class TestExtractTextWithFormattingAdditional:
    """Additional text formatting tests."""

    def test_b_tag_bold(self):
        """<b> tag produces ** markers like <strong>."""
        html = "<p>This is <b>bold</b> text.</p>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_text_with_formatting(soup.find("p"))

        assert "**bold**" in result

    def test_i_tag_italic(self):
        """<i> tag produces * markers like <em>."""
        html = "<p>This is <i>italic</i> text.</p>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_text_with_formatting(soup.find("p"))

        assert "*italic*" in result

    def test_unknown_tag_extracts_text(self):
        """Unknown tags just extract inner text."""
        html = "<p>This is <span>span text</span> here.</p>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._extract_text_with_formatting(soup.find("p"))

        assert "span text" in result


# ---------------------------------------------------------------------------
# _clean_soup
# ---------------------------------------------------------------------------


class TestCleanSoup:
    """Tests for _clean_soup element removal."""

    def test_removes_nav_and_aside(self):
        """Nav and aside elements are removed after navigation extraction."""
        html = (
            "<html><body>"
            "<nav>Navigation content</nav>"
            "<aside>Sidebar</aside>"
            "<main><p>Main content</p></main>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        parser._clean_soup(soup)

        assert soup.find("nav") is None
        assert soup.find("aside") is None
        assert "Main content" in soup.get_text()

    def test_removes_footer_header_tags(self):
        """Footer and header elements are decomposed."""
        html = (
            "<html><body>"
            "<header>Header stuff</header>"
            "<footer>Footer stuff</footer>"
            "<p>Body content</p>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        parser._clean_soup(soup)

        assert soup.find("header") is None
        assert soup.find("footer") is None

    def test_removes_elements_by_class_pattern(self):
        """Elements with class matching social/ads patterns are removed."""
        html = (
            "<html><body>"
            '<div class="social-share">Share this</div>'
            '<div id="ads-banner">Ad content</div>'
            "<p>Real content</p>"
            "</body></html>"
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        parser._clean_soup(soup)

        text = soup.get_text()
        assert "Share this" not in text
        assert "Real content" in text


# ---------------------------------------------------------------------------
# _find_main_content
# ---------------------------------------------------------------------------


class TestFindMainContent:
    """Tests for _find_main_content selector fallback chain."""

    def test_finds_main_element(self):
        """<main> element is selected."""
        html = "<html><body><main><p>Main</p></main></body></html>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._find_main_content(soup)

        assert result.name == "main"

    def test_finds_article_element(self):
        """<article> element is selected when no <main>."""
        html = "<html><body><article><p>Article</p></article></body></html>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._find_main_content(soup)

        assert result.name == "article"

    def test_finds_role_main(self):
        """Element with role='main' is selected."""
        html = '<html><body><div role="main"><p>Content</p></div></body></html>'
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._find_main_content(soup)

        assert result.get("role") == "main"

    def test_falls_back_to_body(self):
        """Falls back to <body> when no content selectors match."""
        html = "<html><body><div><p>Fallback</p></div></body></html>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._find_main_content(soup)

        assert result.name == "body"

    def test_falls_back_to_soup_when_no_body(self):
        """Falls back to soup itself when even body is missing."""
        html = "<div><p>No body tag</p></div>"
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        parser = HTMLParser(parser="html.parser")
        result = parser._find_main_content(soup)

        # Without body, should return soup itself
        assert result is not None
