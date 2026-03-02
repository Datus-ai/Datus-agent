# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.document.fetcher.web_fetcher."""

import re
from unittest.mock import MagicMock

import pytest

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

pytestmark = [
    pytest.mark.ci,
    pytest.mark.skipif(not BS4_AVAILABLE, reason="beautifulsoup4 not installed"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fetcher(**kwargs):
    """Create a WebFetcher with a mock httpx client so no real HTTP is done."""
    from datus.storage.document.fetcher.rate_limiter import RateLimiter

    rl = RateLimiter()
    # Zero out all wait times so tests are fast
    rl.wait = MagicMock(return_value=0.0)

    defaults = dict(platform="testplatform", rate_limiter=rl, pool_size=1, timeout=5.0)
    defaults.update(kwargs)

    from datus.storage.document.fetcher.web_fetcher import WebFetcher

    fetcher = WebFetcher(**defaults)
    return fetcher


# ---------------------------------------------------------------------------
# _extract_path_prefix
# ---------------------------------------------------------------------------


class TestExtractPathPrefix:
    """Tests for WebFetcher._extract_path_prefix."""

    def test_root_returns_none(self):
        """Root path should return None."""
        f = _make_fetcher()
        assert f._extract_path_prefix("/") is None

    def test_empty_returns_none(self):
        """Empty path should return None."""
        f = _make_fetcher()
        assert f._extract_path_prefix("") is None

    def test_single_segment(self):
        """Single segment path returns prefix with slashes."""
        f = _make_fetcher()
        assert f._extract_path_prefix("/docs") == "/docs/"

    def test_multi_segment(self):
        """Multi-segment path uses only the first segment."""
        f = _make_fetcher()
        assert f._extract_path_prefix("/en/docs/overview") == "/en/"

    def test_version_path(self):
        """Version-like path should use first segment."""
        f = _make_fetcher()
        assert f._extract_path_prefix("/docs/v1.2/guide") == "/docs/"

    def test_developer_path(self):
        """Developer section path."""
        f = _make_fetcher()
        assert f._extract_path_prefix("/developer/guide") == "/developer/"


# ---------------------------------------------------------------------------
# _detect_version_from_url
# ---------------------------------------------------------------------------


class TestDetectVersionFromUrl:
    """Tests for WebFetcher._detect_version_from_url."""

    def test_version_with_v_prefix(self):
        """URL with /v1.2.3/ should extract version."""
        f = _make_fetcher()
        assert f._detect_version_from_url("https://example.com/v1.2.3/docs") == "1.2.3"

    def test_version_without_v_prefix(self):
        """URL with /1.2.3/ should extract version."""
        f = _make_fetcher()
        assert f._detect_version_from_url("https://example.com/1.2.3/docs") == "1.2.3"

    def test_version_two_part(self):
        """URL with /v1.2/ should extract two-part version."""
        f = _make_fetcher()
        assert f._detect_version_from_url("https://example.com/v1.2/guide") == "1.2"

    def test_version_explicit_path(self):
        """URL with /version/1.2/ should extract version."""
        f = _make_fetcher()
        assert f._detect_version_from_url("https://example.com/version/1.2/guide") == "1.2"

    def test_docs_number(self):
        """URL with /docs/15/ should extract version as 15."""
        f = _make_fetcher()
        assert f._detect_version_from_url("https://example.com/docs/15/") == "15"

    def test_date_pattern(self):
        """URL with /2024-01/ should extract date."""
        f = _make_fetcher()
        assert f._detect_version_from_url("https://example.com/2024-01/ref") == "2024-01"

    def test_no_version_falls_back_to_date(self):
        """URL without version returns current date."""
        f = _make_fetcher()
        result = f._detect_version_from_url("https://example.com/docs/guide")
        # Should be a date string in YYYY-MM-DD format
        assert re.match(r"\d{4}-\d{2}-\d{2}", result)


# ---------------------------------------------------------------------------
# _should_include
# ---------------------------------------------------------------------------


class TestShouldInclude:
    """Tests for WebFetcher._should_include."""

    def test_no_patterns_includes_all(self):
        """With no include/exclude patterns, everything is included."""
        f = _make_fetcher()
        assert f._should_include("https://example.com/any", [], []) is True

    def test_exclude_pattern_blocks(self):
        """A matching exclude pattern should block the URL."""
        f = _make_fetcher()
        exclude = [re.compile(r"/api/")]
        assert f._should_include("https://example.com/api/v1", [], exclude) is False

    def test_exclude_non_matching_passes(self):
        """A non-matching exclude pattern should allow the URL."""
        f = _make_fetcher()
        exclude = [re.compile(r"/api/")]
        assert f._should_include("https://example.com/docs/guide", [], exclude) is True

    def test_include_pattern_matches(self):
        """A matching include pattern should allow the URL."""
        f = _make_fetcher()
        include = [re.compile(r"/docs/")]
        assert f._should_include("https://example.com/docs/guide", include, []) is True

    def test_include_pattern_non_matching_blocks(self):
        """When include patterns exist, a non-matching URL is blocked."""
        f = _make_fetcher()
        include = [re.compile(r"/docs/")]
        assert f._should_include("https://example.com/blog/post", include, []) is False

    def test_exclude_takes_precedence_over_include(self):
        """Exclude patterns are checked before include patterns."""
        f = _make_fetcher()
        include = [re.compile(r"/docs/")]
        exclude = [re.compile(r"internal")]
        assert f._should_include("https://example.com/docs/internal", include, exclude) is False


# ---------------------------------------------------------------------------
# _extract_links
# ---------------------------------------------------------------------------


class TestExtractLinks:
    """Tests for WebFetcher._extract_links."""

    def test_extracts_same_domain_links(self):
        """Should extract links from the same domain."""
        f = _make_fetcher()
        html = '<html><body><a href="/page2">Link</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        links = f._extract_links(soup, "https://example.com/page1", "example.com")
        assert "https://example.com/page2" in links

    def test_skips_external_links(self):
        """Should skip links to different domains."""
        f = _make_fetcher()
        html = '<html><body><a href="https://other.com/page">Link</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        links = f._extract_links(soup, "https://example.com/page1", "example.com")
        assert len(links) == 0

    def test_skips_anchors(self):
        """Should skip anchor links (#fragment)."""
        f = _make_fetcher()
        html = '<html><body><a href="#section">Anchor</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        links = f._extract_links(soup, "https://example.com/page1", "example.com")
        assert len(links) == 0

    def test_skips_javascript_links(self):
        """Should skip javascript: links."""
        f = _make_fetcher()
        html = '<html><body><a href="javascript:void(0)">JS</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        links = f._extract_links(soup, "https://example.com/page1", "example.com")
        assert len(links) == 0

    def test_skips_mailto_links(self):
        """Should skip mailto: links."""
        f = _make_fetcher()
        html = '<html><body><a href="mailto:test@test.com">Email</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        links = f._extract_links(soup, "https://example.com/page1", "example.com")
        assert len(links) == 0

    def test_skips_non_doc_extensions(self):
        """Should skip links to images, CSS, JS and other non-doc files."""
        f = _make_fetcher()
        html = """<html><body>
        <a href="/img.png">PNG</a>
        <a href="/style.css">CSS</a>
        <a href="/script.js">JS</a>
        <a href="/file.pdf">PDF</a>
        <a href="/font.woff2">Font</a>
        </body></html>"""
        soup = BeautifulSoup(html, "html.parser")
        links = f._extract_links(soup, "https://example.com/", "example.com")
        assert len(links) == 0

    def test_removes_fragment_from_urls(self):
        """Should strip fragment identifiers from extracted URLs."""
        f = _make_fetcher()
        html = '<html><body><a href="/page2#section">Link</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        links = f._extract_links(soup, "https://example.com/page1", "example.com")
        assert "https://example.com/page2" in links
        # No URL should contain a fragment
        assert all("#" not in link for link in links)

    def test_preserves_query_string(self):
        """Should preserve query strings in extracted URLs."""
        f = _make_fetcher()
        html = '<html><body><a href="/page2?tab=overview">Link</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        links = f._extract_links(soup, "https://example.com/page1", "example.com")
        assert "https://example.com/page2?tab=overview" in links

    def test_deduplicates_links(self):
        """Duplicate hrefs should produce unique links."""
        f = _make_fetcher()
        html = """<html><body>
        <a href="/page2">Link 1</a>
        <a href="/page2">Link 2</a>
        </body></html>"""
        soup = BeautifulSoup(html, "html.parser")
        links = f._extract_links(soup, "https://example.com/page1", "example.com")
        assert len(links) == 1

    def test_resolves_relative_urls(self):
        """Relative URLs should be resolved against the current URL."""
        f = _make_fetcher()
        html = '<html><body><a href="subpage">Link</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        links = f._extract_links(soup, "https://example.com/docs/", "example.com")
        assert "https://example.com/docs/subpage" in links


# ---------------------------------------------------------------------------
# URL normalization in fetch / fetch_single
# ---------------------------------------------------------------------------


class TestUrlNormalization:
    """Tests for URL normalization in fetch/fetch_single."""

    def test_fetch_adds_https_prefix(self):
        """fetch() should add https:// to bare URLs."""
        f = _make_fetcher()
        # Patch _fetch_page to capture the source URL that was built
        calls = []

        def mock_fetch_page(url, depth, base_domain, version):
            calls.append(url)
            return None

        f._fetch_page = mock_fetch_page

        f.fetch("example.com/docs", max_depth=0)
        assert calls[0].startswith("https://")

    def test_fetch_single_adds_https(self):
        """fetch_single() should add https:// to bare URLs."""
        f = _make_fetcher()

        calls = []

        def mock_fetch_page(url, depth, base_domain, version):
            calls.append(url)
            return None

        f._fetch_page = mock_fetch_page

        f.fetch_single("example.com/docs/page")
        assert calls[0].startswith("https://")

    def test_fetch_single_with_base_url(self):
        """fetch_single() should join relative paths with base_url."""
        f = _make_fetcher()

        calls = []

        def mock_fetch_page(url, depth, base_domain, version):
            calls.append(url)
            return None

        f._fetch_page = mock_fetch_page

        f.fetch_single("guide/intro", base_url="https://example.com/docs/")
        assert calls[0] == "https://example.com/docs/guide/intro"


# ---------------------------------------------------------------------------
# _fetch_page
# ---------------------------------------------------------------------------


class TestFetchPage:
    """Tests for WebFetcher._fetch_page with mocked httpx."""

    def test_fetch_page_html_success(self):
        """Successful HTML page fetch should return (doc, links)."""
        f = _make_fetcher()

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.text = "<html><head><title>Test Page</title></head><body><a href='/other'>Link</a></body></html>"
        mock_response.raise_for_status = MagicMock()

        f._client = MagicMock()
        f._client.get.return_value = mock_response

        result = f._fetch_page("https://example.com/docs/page", 1, "example.com", "1.0")
        assert result is not None
        doc, links = result
        assert doc.platform == "testplatform"
        assert doc.version == "1.0"
        assert doc.metadata["title"] == "Test Page"
        assert doc.metadata["depth"] == 1
        assert doc.source_url == "https://example.com/docs/page"
        assert doc.content_type == "html"

    def test_fetch_page_non_html_returns_none(self):
        """Non-HTML content types should be skipped."""
        f = _make_fetcher()

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status = MagicMock()

        f._client = MagicMock()
        f._client.get.return_value = mock_response

        result = f._fetch_page("https://example.com/api/data", 0, "example.com", "1.0")
        assert result is None

    def test_fetch_page_xhtml_accepted(self):
        """application/xhtml content type should be accepted."""
        f = _make_fetcher()

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/xhtml+xml"}
        mock_response.text = "<html><head><title>XHTML</title></head><body></body></html>"
        mock_response.raise_for_status = MagicMock()

        f._client = MagicMock()
        f._client.get.return_value = mock_response

        result = f._fetch_page("https://example.com/page", 0, "example.com", "1.0")
        assert result is not None

    def test_fetch_page_404_returns_none(self):
        """404 errors should return None."""
        import httpx

        f = _make_fetcher()

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        f._client = MagicMock()
        f._client.get.return_value = mock_response

        result = f._fetch_page("https://example.com/missing", 0, "example.com", "1.0")
        assert result is None

    def test_fetch_page_500_returns_none(self):
        """5xx errors should return None."""
        import httpx

        f = _make_fetcher()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )

        f._client = MagicMock()
        f._client.get.return_value = mock_response

        result = f._fetch_page("https://example.com/error", 0, "example.com", "1.0")
        assert result is None

    def test_fetch_page_request_error_returns_none(self):
        """Connection errors should return None."""
        import httpx

        f = _make_fetcher()

        f._client = MagicMock()
        f._client.get.side_effect = httpx.RequestError("Connection refused", request=MagicMock())

        result = f._fetch_page("https://example.com/timeout", 0, "example.com", "1.0")
        assert result is None

    def test_fetch_page_doc_path_construction(self):
        """doc_path should be extracted from URL path, trailing / gets index.html."""
        f = _make_fetcher()

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html><head><title>T</title></head><body></body></html>"
        mock_response.raise_for_status = MagicMock()

        f._client = MagicMock()
        f._client.get.return_value = mock_response

        result = f._fetch_page("https://example.com/docs/", 0, "example.com", "1.0")
        assert result is not None
        doc, _ = result
        assert doc.doc_path == "/docs/index.html"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """Tests for context manager protocol."""

    def test_enter_returns_self(self):
        """__enter__ should return the fetcher itself."""
        f = _make_fetcher()
        assert f.__enter__() is f

    def test_exit_closes_client(self):
        """__exit__ should close the HTTP client."""
        f = _make_fetcher()
        f._client = MagicMock()
        f.__exit__(None, None, None)
        assert f._client is None

    def test_close_idempotent(self):
        """close() should be safe to call multiple times."""
        f = _make_fetcher()
        f.close()
        f.close()  # Should not raise
        assert f._client is None


# ---------------------------------------------------------------------------
# fetch with crawling
# ---------------------------------------------------------------------------


class TestFetchCrawling:
    """Tests for the main fetch() crawl loop."""

    def test_fetch_max_depth_0_single_page(self):
        """max_depth=0 should fetch only the starting URL."""
        f = _make_fetcher()

        call_urls = []

        def mock_fetch_page(url, depth, base_domain, version):
            call_urls.append(url)
            from datus.storage.document.schemas import CONTENT_TYPE_HTML, SOURCE_TYPE_WEBSITE, FetchedDocument

            doc = FetchedDocument(
                platform="testplatform",
                version="1.0",
                source_url=url,
                source_type=SOURCE_TYPE_WEBSITE,
                doc_path="/page",
                raw_content="<html></html>",
                content_type=CONTENT_TYPE_HTML,
                metadata={"title": "Test", "depth": depth, "content_length": 10},
            )
            links = ["https://example.com/page2", "https://example.com/page3"]
            return (doc, links)

        f._fetch_page = mock_fetch_page

        docs = f.fetch("https://example.com/docs", max_depth=0)
        # Only the starting URL should be fetched (depth=0, and discovered links are depth=1 which exceeds max_depth=0)
        assert len(call_urls) == 1
        assert len(docs) == 1

    def test_fetch_respects_exclude_patterns(self):
        """Fetched pages matching exclude patterns should not appear in results."""
        f = _make_fetcher()

        def mock_fetch_page(url, depth, base_domain, version):
            from datus.storage.document.schemas import CONTENT_TYPE_HTML, SOURCE_TYPE_WEBSITE, FetchedDocument

            doc = FetchedDocument(
                platform="testplatform",
                version="1.0",
                source_url=url,
                source_type=SOURCE_TYPE_WEBSITE,
                doc_path=url.replace("https://example.com", ""),
                raw_content="<html></html>",
                content_type=CONTENT_TYPE_HTML,
                metadata={"title": "Test", "depth": depth, "content_length": 10},
            )
            return (doc, [])

        f._fetch_page = mock_fetch_page

        docs = f.fetch("https://example.com/docs/internal/page", max_depth=0, exclude_patterns=[r"internal"])
        assert len(docs) == 0

    def test_fetch_skips_visited_urls(self):
        """Already visited URLs should not be fetched again."""
        f = _make_fetcher()

        call_count = 0

        def mock_fetch_page(url, depth, base_domain, version):
            nonlocal call_count
            call_count += 1
            from datus.storage.document.schemas import CONTENT_TYPE_HTML, SOURCE_TYPE_WEBSITE, FetchedDocument

            doc = FetchedDocument(
                platform="testplatform",
                version="1.0",
                source_url=url,
                source_type=SOURCE_TYPE_WEBSITE,
                doc_path="/page",
                raw_content="<html></html>",
                content_type=CONTENT_TYPE_HTML,
                metadata={"title": "Test", "depth": depth, "content_length": 10},
            )
            # Return link back to self to test dedup
            return (doc, [url])

        f._fetch_page = mock_fetch_page

        f.fetch("https://example.com/docs", max_depth=2)
        # Should only be called once despite the self-referencing link
        assert call_count == 1
