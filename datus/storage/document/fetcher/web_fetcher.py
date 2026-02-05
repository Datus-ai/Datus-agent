# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Website Documentation Fetcher

Fetches documentation from official websites by crawling HTML pages.
Supports recursive crawling with depth limits and URL filtering.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx

from datus.storage.document.fetcher.base_fetcher import BaseFetcher
from datus.storage.document.fetcher.rate_limiter import RateLimiter, get_rate_limiter
from datus.storage.document.schemas import CONTENT_TYPE_HTML, SOURCE_TYPE_WEBSITE, FetchedDocument
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Check if BeautifulSoup is available
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None


class WebFetcher(BaseFetcher):
    """Fetcher for website documentation.

    Crawls documentation websites to fetch HTML pages.
    Supports:
    - Recursive crawling with configurable depth
    - URL pattern filtering (include/exclude)
    - Rate limiting
    - Parallel fetching

    Example:
        >>> fetcher = WebFetcher(platform="duckdb")
        >>> docs = fetcher.fetch("https://duckdb.org/docs/", max_depth=2)
    """

    DEFAULT_HEADERS = {
        "User-Agent": ("Mozilla/5.0 (compatible; DatusDocBot/1.0; " "+https://github.com/Datus-ai/Datus-agent)"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(
        self,
        platform: str,
        version: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        pool_size: int = 4,
        timeout: float = 30.0,
    ):
        """Initialize the web fetcher.

        Args:
            platform: Target platform name
            version: Target version (optional)
            rate_limiter: Rate limiter instance
            pool_size: Thread pool size for parallel fetching
            timeout: HTTP request timeout in seconds
        """
        super().__init__(platform=platform, version=version)

        if not BS4_AVAILABLE:
            raise ImportError(
                "BeautifulSoup4 is required for web fetching. " "Install it with: pip install beautifulsoup4 lxml"
            )

        self.rate_limiter = rate_limiter or get_rate_limiter()
        self.pool_size = pool_size
        self.timeout = timeout

        self._client = httpx.Client(
            headers=self.DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close HTTP client."""
        self.close()
        return False

    def close(self):
        """Close the HTTP client and release resources."""
        if hasattr(self, "_client") and self._client:
            self._client.close()
            self._client = None

    def __del__(self):
        """Clean up HTTP client."""
        self.close()

    def fetch(
        self,
        source: str,
        max_depth: int = 2,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        enforce_source_prefix: bool = True,
        **kwargs,
    ) -> List[FetchedDocument]:
        """Fetch documentation from a website.

        Args:
            source: Starting URL
            max_depth: Maximum crawl depth (0 = only starting URL)
            include_patterns: URL patterns to include (regex)
            exclude_patterns: URL patterns to exclude (regex)
            enforce_source_prefix: If True, only follow links that share
                the source URL's path prefix (e.g., /en/ won't explore /de/)
            **kwargs: Additional parameters

        Returns:
            List of fetched documents
        """
        # Normalize URL
        if not source.startswith(("http://", "https://")):
            source = "https://" + source

        parsed = urlparse(source)
        base_domain = parsed.netloc

        # Extract path prefix for filtering (e.g., /en/ from /en/docs/overview)
        source_path_prefix = self._extract_path_prefix(parsed.path) if enforce_source_prefix else None

        # Compile patterns
        include_re = [re.compile(p) for p in (include_patterns or [])]
        exclude_re = [re.compile(p) for p in (exclude_patterns or [])]

        # Track visited URLs
        visited: Set[str] = set()
        to_visit: List[tuple] = [(source, 0)]  # (url, depth)

        # Detect version from URL or page
        version = self.version or self._detect_version_from_url(source)

        documents = []

        while to_visit:
            # Get next batch of URLs at current depth
            current_batch = []
            next_batch = []

            for url, depth in to_visit:
                if url in visited:
                    continue
                if depth <= max_depth:
                    current_batch.append((url, depth))
                else:
                    next_batch.append((url, depth))

            to_visit = next_batch

            if not current_batch:
                continue

            # Log progress
            logger.info(
                f"Fetching batch: {len(current_batch)} URLs at depth {current_batch[0][1]}, "
                f"visited: {len(visited)}, queued: {len(to_visit)}"
            )

            # Fetch current batch in parallel
            with ThreadPoolExecutor(max_workers=self.pool_size) as executor:
                futures = {
                    executor.submit(
                        self._fetch_page,
                        url,
                        depth,
                        base_domain,
                        version,
                    ): (url, depth)
                    for url, depth in current_batch
                    if url not in visited
                }

                for future in as_completed(futures):
                    url, depth = futures[future]
                    visited.add(url)

                    try:
                        result = future.result()
                        if result is None:
                            continue

                        doc, links = result

                        # Check if document matches filters
                        if self._should_include(url, include_re, exclude_re):
                            documents.append(doc)

                        # Add discovered links
                        if depth < max_depth:
                            for link in links:
                                if link not in visited:
                                    # Only follow links on same domain
                                    link_parsed = urlparse(link)
                                    if link_parsed.netloc == base_domain:
                                        # Enforce source path prefix if enabled
                                        if source_path_prefix and not link_parsed.path.startswith(source_path_prefix):
                                            continue
                                        if self._should_include(link, include_re, exclude_re):
                                            to_visit.append((link, depth + 1))

                    except Exception as e:
                        logger.warning(f"Failed to fetch {url}: {e}")

        logger.info(f"Fetched {len(documents)} pages from {base_domain}")
        return documents

    def fetch_single(
        self,
        path: str,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> Optional[FetchedDocument]:
        """Fetch a single page.

        Args:
            path: URL or path to fetch
            base_url: Base URL for relative paths
            **kwargs: Additional parameters

        Returns:
            Fetched document or None
        """
        # Construct full URL
        if base_url and not path.startswith(("http://", "https://")):
            url = urljoin(base_url, path)
        else:
            url = path

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        version = self.version or self._detect_version_from_url(url)
        result = self._fetch_page(url, 0, urlparse(url).netloc, version)

        if result:
            return result[0]
        return None

    def _fetch_page(
        self,
        url: str,
        depth: int,
        base_domain: str,
        version: str,
    ) -> Optional[tuple]:
        """Fetch a single page and extract links.

        Args:
            url: URL to fetch
            depth: Current crawl depth
            base_domain: Base domain for relative links
            version: Version string

        Returns:
            Tuple of (FetchedDocument, List[links]) or None
        """
        try:
            # Rate limit
            self.rate_limiter.wait(base_domain)

            # Fetch page
            response = self._client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            # Only process HTML
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                logger.debug(f"Skipping non-HTML content: {url}")
                return None

            raw_content = response.text

            # Parse HTML
            soup = BeautifulSoup(raw_content, "lxml")

            # Extract title
            title = ""
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Extract links
            links = self._extract_links(soup, url, base_domain)

            # Determine doc_path from URL
            parsed = urlparse(url)
            doc_path = parsed.path or "/"
            if doc_path.endswith("/"):
                doc_path += "index.html"

            # Build document
            doc = FetchedDocument(
                platform=self.platform,
                version=version,
                source_url=url,
                source_type=SOURCE_TYPE_WEBSITE,
                doc_path=doc_path,
                raw_content=raw_content,
                content_type=CONTENT_TYPE_HTML,
                metadata={
                    "title": title,
                    "depth": depth,
                    "content_length": len(raw_content),
                },
            )

            return (doc, links)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"Page not found: {url}")
            else:
                logger.warning(f"HTTP error for {url}: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"Request error for {url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error fetching {url}: {e}")
            return None

    def _extract_links(
        self,
        soup: "BeautifulSoup",
        current_url: str,
        base_domain: str,
    ) -> List[str]:
        """Extract documentation links from a page.

        Args:
            soup: Parsed HTML
            current_url: Current page URL
            base_domain: Base domain for filtering

        Returns:
            List of absolute URLs
        """
        links = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Skip anchors, javascript, etc.
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            # Convert to absolute URL
            absolute_url = urljoin(current_url, href)

            # Remove fragment
            parsed = urlparse(absolute_url)
            absolute_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                absolute_url += f"?{parsed.query}"

            # Only include same-domain links
            if parsed.netloc == base_domain:
                # Skip non-doc files
                path_lower = parsed.path.lower()
                skip_extensions = {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".svg",
                    ".ico",
                    ".css",
                    ".js",
                    ".json",
                    ".xml",
                    ".pdf",
                    ".zip",
                    ".tar",
                    ".gz",
                    ".woff",
                    ".woff2",
                    ".ttf",
                    ".eot",
                }
                if not any(path_lower.endswith(ext) for ext in skip_extensions):
                    links.add(absolute_url)

        return list(links)

    def _should_include(
        self,
        url: str,
        include_patterns: List[re.Pattern],
        exclude_patterns: List[re.Pattern],
    ) -> bool:
        """Check if URL should be included based on patterns.

        Args:
            url: URL to check
            include_patterns: Patterns to match for inclusion
            exclude_patterns: Patterns to match for exclusion

        Returns:
            True if URL should be included
        """
        # Check exclude patterns first
        for pattern in exclude_patterns:
            if pattern.search(url):
                return False

        # If include patterns specified, URL must match at least one
        if include_patterns:
            return any(pattern.search(url) for pattern in include_patterns)

        return True

    def _extract_path_prefix(self, path: str) -> Optional[str]:
        """Extract path prefix for URL filtering.

        Extracts the first significant path segment to use as a prefix filter.
        This prevents crawling into other language versions or unrelated sections.

        Examples:
            /en/docs/overview → /en/
            /docs/v1.2/guide → /docs/
            /developer/guide → /developer/
            / → None

        Args:
            path: URL path to analyze

        Returns:
            Path prefix string or None if no meaningful prefix
        """
        if not path or path == "/":
            return None

        # Split path into segments
        segments = [s for s in path.split("/") if s]
        if not segments:
            return None

        # Use first segment as prefix
        first_segment = segments[0]

        # Return prefix with leading and trailing slashes
        return f"/{first_segment}/"

    def _detect_version_from_url(self, url: str) -> str:
        """Try to detect version from URL.

        Args:
            url: URL to analyze

        Returns:
            Version string or current date
        """
        # Common patterns: /v1.2.3/, /version/1.2/, /docs/15/, etc.
        patterns = [
            r"/v?(\d+\.\d+(?:\.\d+)?)/",
            r"/version/(\d+\.\d+(?:\.\d+)?)/",
            r"/docs/(\d+)/",
            r"/(\d{4}-\d{2})/",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
