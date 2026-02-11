# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
HTML Document Parser

Parses HTML documents into a structured format with hierarchical sections.
Uses BeautifulSoup4 for DOM traversal.
"""

import re
from typing import Any, Dict, List

from datus.storage.document.schemas import FetchedDocument, ParsedDocument, ParsedSection
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Check if BeautifulSoup is available
try:
    from bs4 import BeautifulSoup, NavigableString, Tag

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None


class HTMLParser:
    """Parser for HTML documents.

    Converts HTML content into a hierarchical structure of sections,
    extracting text content and preserving code blocks.

    Features:
    - Hierarchical section extraction based on headings (h1-h6)
    - Code block preservation (pre, code tags)
    - Navigation path extraction (breadcrumbs, sidebar)
    - Navigation and boilerplate removal
    - Clean text extraction

    Example:
        >>> parser = HTMLParser()
        >>> parsed = parser.parse(fetched_doc)
        >>> print(parsed.title)
        >>> print(parsed.metadata.get("nav_path"))  # Site navigation path
    """

    # Tags to remove entirely
    REMOVE_TAGS = {
        "script",
        "style",
        "header",
        "footer",
        "noscript",
        "iframe",
        "svg",
        "form",
    }

    # Classes/IDs that typically indicate non-content
    REMOVE_PATTERNS = [
        r"footer",
        r"header",
        r"social",
        r"share",
        r"comment",
        r"advertisement",
        r"ads",
        r"banner",
        r"cookie",
        r"popup",
        r"modal",
    ]

    # Breadcrumb selectors (ordered by priority)
    BREADCRUMB_SELECTORS = [
        '[aria-label="breadcrumb"]',
        '[aria-label="Breadcrumb"]',
        ".breadcrumb",
        ".breadcrumbs",
        "#breadcrumb",
        "#breadcrumbs",
        '[class*="breadcrumb"]',
        "nav.crumbs",
        ".crumbs",
    ]

    # Sidebar navigation selectors
    SIDEBAR_SELECTORS = [
        '[aria-label="Docs pages"]',
        '[aria-label="docs pages"]',
        '[aria-label="Documentation"]',
        '[aria-label="Side navigation"]',
        ".docs-sidebar",
        ".doc-sidebar",
        ".sidebar-nav",
        ".side-nav",
        ".toc-sidebar",
        "aside nav",
        "nav.sidebar",
        '[class*="sidebar"] nav',
        '[class*="sidebar"] ul',
        ".left-nav",
        ".leftnav",
    ]

    def __init__(self, parser: str = "lxml"):
        """Initialize the HTML parser.

        Args:
            parser: BeautifulSoup parser to use ('lxml', 'html.parser', etc.)
        """
        if not BS4_AVAILABLE:
            raise ImportError(
                "BeautifulSoup4 is required for HTML parsing. " "Install with: pip install beautifulsoup4 lxml"
            )

        self._parser = parser
        self._remove_pattern = re.compile("|".join(self.REMOVE_PATTERNS), re.IGNORECASE)

    def parse(self, doc: FetchedDocument) -> ParsedDocument:
        """Parse a fetched HTML document.

        Args:
            doc: Fetched document with HTML content

        Returns:
            Parsed document with hierarchical sections
        """
        soup = BeautifulSoup(doc.raw_content, self._parser)

        # Extract metadata
        metadata = self._extract_metadata(soup)

        # *** Extract navigation BEFORE cleaning ***
        nav_path = self._extract_navigation_path(soup, doc.source_url)
        if nav_path:
            metadata["nav_path"] = nav_path
            # Extract group_name from nav_path
            group_name = self._extract_group_name(nav_path, soup)
            metadata["group_name"] = group_name
            logger.debug(f"Extracted nav_path: {nav_path}, group_name: {group_name}")

        # Clean the document
        self._clean_soup(soup)

        # Find main content area
        main_content = self._find_main_content(soup)

        # Extract sections
        sections = self._extract_sections(main_content)

        # Extract title
        title = metadata.get("title", "")
        if not title:
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

        # Clean up title (remove site name suffix)
        if " | " in title:
            title = title.split(" | ")[0].strip()
        elif " - " in title:
            title = title.split(" - ")[0].strip()

        if not title:
            title = doc.doc_path.split("/")[-1].replace(".html", "").replace("-", " ").title()

        return ParsedDocument(
            title=title,
            sections=sections,
            metadata=metadata,
            source_doc=doc,
        )

    def _extract_navigation_path(self, soup: "BeautifulSoup", current_url: str) -> List[str]:
        """Extract site navigation path for current page.

        Tries multiple strategies:
        1. Breadcrumb navigation
        2. Sidebar navigation (find current page's ancestors)
        3. URL path fallback

        Args:
            soup: Parsed HTML (before cleaning)
            current_url: Current page URL

        Returns:
            List of navigation titles from root to current page
        """
        # Strategy 1: Try breadcrumb navigation
        breadcrumb_path = self._extract_breadcrumb(soup)
        if breadcrumb_path and len(breadcrumb_path) > 1:
            logger.debug(f"Found breadcrumb path: {breadcrumb_path}")
            return breadcrumb_path

        # Strategy 2: Try sidebar navigation
        sidebar_path = self._extract_sidebar_path(soup, current_url)
        if sidebar_path and len(sidebar_path) > 1:
            logger.debug(f"Found sidebar path: {sidebar_path}")
            return sidebar_path

        # Strategy 3: URL path fallback
        url_path = self._extract_path_from_url(current_url)
        if url_path:
            logger.debug(f"Using URL path: {url_path}")
            return url_path

        return []

    def _extract_breadcrumb(self, soup: "BeautifulSoup") -> List[str]:
        """Extract breadcrumb navigation.

        Args:
            soup: Parsed HTML

        Returns:
            List of breadcrumb items
        """
        for selector in self.BREADCRUMB_SELECTORS:
            try:
                breadcrumb = soup.select_one(selector)
                if not breadcrumb:
                    continue

                # Extract items from breadcrumb
                items = []

                # Try structured data (schema.org BreadcrumbList)
                for item in breadcrumb.select('[itemtype*="BreadcrumbList"] [itemprop="name"]'):
                    text = item.get_text(strip=True)
                    if text:
                        items.append(text)

                if items:
                    return items

                # Try list items
                for li in breadcrumb.find_all("li"):
                    text = li.get_text(strip=True)
                    # Clean separators
                    text = text.strip("/›»>|")
                    if text and text not in ("Home", "Docs", "Documentation", "..."):
                        items.append(text)

                if items:
                    return items

                # Try links
                for a in breadcrumb.find_all("a"):
                    text = a.get_text(strip=True)
                    if text and text not in ("Home", "Docs", "Documentation", "..."):
                        items.append(text)

                # Add current page (often not a link)
                current = breadcrumb.find("span", class_=re.compile(r"current|active"))
                if current:
                    text = current.get_text(strip=True)
                    if text:
                        items.append(text)

                if items:
                    return items

            except Exception as e:
                logger.debug(f"Breadcrumb extraction failed for {selector}: {e}")
                continue

        return []

    def _extract_sidebar_path(self, soup: "BeautifulSoup", current_url: str) -> List[str]:
        """Extract path from sidebar navigation by finding current page.

        Args:
            soup: Parsed HTML
            current_url: Current page URL to locate in sidebar

        Returns:
            List of ancestor titles leading to current page
        """
        from urllib.parse import urlparse

        current_path = urlparse(current_url).path.rstrip("/")

        for selector in self.SIDEBAR_SELECTORS:
            try:
                sidebar = soup.select_one(selector)
                if not sidebar:
                    continue

                # Find the link matching current URL
                current_link = None
                for a in sidebar.find_all("a", href=True):
                    href = a["href"]
                    # Skip empty or anchor-only hrefs
                    if not href or href == "#" or href.startswith("#"):
                        continue
                    # Normalize href
                    if href.startswith("/"):
                        href_path = href.rstrip("/")
                    else:
                        href_path = urlparse(href).path.rstrip("/")

                    if href_path == current_path or current_path.endswith(href_path):
                        current_link = a
                        break

                if not current_link:
                    continue

                # Traverse up to find ancestor items
                path = []
                element = current_link

                # First, add current page title
                current_text = current_link.get_text(strip=True)
                if current_text:
                    path.append(current_text)

                # Walk up the DOM to find parent navigation items
                while element:
                    element = element.parent
                    if not element:
                        break

                    # Check if this is a navigation list item
                    if element.name == "li":
                        # Look for a direct link (parent category)
                        parent_link = element.find("a", recursive=False)
                        if not parent_link:
                            # Try to find in a wrapper div/span
                            for child in element.children:
                                if isinstance(child, Tag):
                                    if child.name == "a":
                                        parent_link = child
                                        break
                                    elif child.name in ("div", "span"):
                                        parent_link = child.find("a")
                                        if parent_link:
                                            break

                        if parent_link and parent_link != current_link:
                            text = parent_link.get_text(strip=True)
                            if text and text not in path:
                                path.append(text)

                    # Also check for section headers
                    if element.name in ("details", "section", "div"):
                        # Look for summary or heading
                        summary = element.find("summary")
                        if summary:
                            text = summary.get_text(strip=True)
                            if text and text not in path:
                                path.append(text)
                        else:
                            heading = element.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                            if heading:
                                text = heading.get_text(strip=True)
                                if text and text not in path:
                                    path.append(text)

                if path:
                    # Reverse to get root -> current order
                    path.reverse()
                    return path

            except Exception as e:
                logger.debug(f"Sidebar extraction failed for {selector}: {e}")
                continue

        return []

    def _extract_path_from_url(self, url: str) -> List[str]:
        """Extract navigation path from URL structure.

        Converts URL path segments to readable titles.
        e.g., /en/user-guide/data-load-snowpipe-intro
              -> ["User Guide", "Data Load", "Snowpipe Intro"]

        Args:
            url: Page URL

        Returns:
            List of path segments as titles
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        path = parsed.path

        # Split and filter path segments
        segments = [s for s in path.split("/") if s]

        # Remove common prefixes
        skip_segments = {"en", "docs", "documentation", "latest", "stable", "v1", "v2", "api"}
        segments = [s for s in segments if s.lower() not in skip_segments]

        if not segments:
            return []

        # Convert to readable titles
        path_titles = []
        for segment in segments:
            # Convert slug to title
            # data-load-snowpipe-intro -> Data Load Snowpipe Intro
            title = segment.replace("-", " ").replace("_", " ")
            title = " ".join(word.capitalize() for word in title.split())
            path_titles.append(title)

        return path_titles

    def _extract_group_name(self, nav_path: List[str], soup: "BeautifulSoup") -> str:
        """Extract the top-level documentation group name.

        Common groups include: Guides, Get Started, Developers, Reference,
        API Reference, Tutorials, etc.

        Args:
            nav_path: Extracted navigation path
            soup: Parsed HTML (for fallback detection)

        Returns:
            Group name or empty string if not identified
        """
        # Common documentation group names (case-insensitive matching)
        KNOWN_GROUPS = {
            "guides",
            "guide",
            "get started",
            "getting started",
            "quickstart",
            "quick start",
            "tutorials",
            "tutorial",
            "developers",
            "developer",
            "reference",
            "api reference",
            "api",
            "concepts",
            "overview",
            "user guide",
            "user manual",
            "administration",
            "admin",
            "security",
            "release notes",
            "releases",
            "faq",
            "faqs",
            "troubleshooting",
            "best practices",
            "examples",
            "samples",
            "integrations",
        }

        # Strategy 1: Check first element of nav_path
        if nav_path:
            first = nav_path[0].lower()
            if first in KNOWN_GROUPS:
                return nav_path[0]

            # Check if first element contains a known group
            for group in KNOWN_GROUPS:
                if group in first:
                    return nav_path[0]

        # Strategy 2: Look for top-level navigation tabs/links
        # Many doc sites have tabs like "Guides | API | Reference"
        top_nav_selectors = [
            '[role="navigation"] > ul > li > a',
            ".nav-tabs a",
            ".top-nav a",
            "header nav a",
        ]

        for selector in top_nav_selectors:
            try:
                for link in soup.select(selector):
                    text = link.get_text(strip=True).lower()
                    if text in KNOWN_GROUPS:
                        # Check if current page is under this group
                        href = link.get("href", "")
                        if href and nav_path:
                            # Simple check: if nav_path contains this text
                            for item in nav_path:
                                if text in item.lower():
                                    return link.get_text(strip=True)
            except Exception:
                continue

        # Strategy 3: Return first nav_path element if it looks like a category
        # (not a specific page title - usually shorter than 30 chars)
        if nav_path and len(nav_path[0]) < 30:
            return nav_path[0]

        return ""

    def _extract_metadata(self, soup: "BeautifulSoup") -> Dict[str, Any]:
        """Extract metadata from HTML head.

        Args:
            soup: Parsed HTML

        Returns:
            Metadata dictionary
        """
        metadata = {}

        # Title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        # Meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name", meta.get("property", ""))
            content = meta.get("content", "")

            if name and content:
                # Normalize common meta names
                name_lower = name.lower()
                if "description" in name_lower:
                    metadata["description"] = content
                elif "author" in name_lower:
                    metadata["author"] = content
                elif "keywords" in name_lower:
                    metadata["keywords"] = [k.strip() for k in content.split(",")]
                elif "og:title" in name_lower:
                    metadata.setdefault("title", content)

        return metadata

    def _clean_soup(self, soup: "BeautifulSoup"):
        """Remove non-content elements from the document.

        Note: This is called AFTER navigation extraction, so nav/aside
        elements can be safely removed here.

        Args:
            soup: Parsed HTML (modified in place)
        """
        # Remove unwanted tags (including nav/aside after navigation extraction)
        tags_to_remove = self.REMOVE_TAGS | {"nav", "aside"}
        for tag_name in tags_to_remove:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove elements with non-content classes/IDs
        for element in soup.find_all(True):
            if element.attrs is None:
                continue
            classes = element.get("class", [])
            element_id = element.get("id", "")

            # Check if class or ID matches removal patterns
            class_str = " ".join(classes) if isinstance(classes, list) else str(classes)
            combined = f"{class_str} {element_id}"

            if self._remove_pattern.search(combined):
                element.decompose()

    def _find_main_content(self, soup: "BeautifulSoup") -> "Tag":
        """Find the main content area of the page.

        Args:
            soup: Cleaned HTML

        Returns:
            Main content element
        """
        # Try common main content selectors
        selectors = [
            "main",
            "article",
            '[role="main"]',
            ".content",
            ".main-content",
            ".doc-content",
            ".documentation",
            ".markdown-body",
            "#content",
            "#main",
            "#docs",
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element

        # Fall back to body
        body = soup.find("body")
        return body if body else soup

    def _extract_sections(self, content: "Tag") -> List[ParsedSection]:
        """Extract hierarchical sections from content.

        Args:
            content: Main content element

        Returns:
            List of parsed sections
        """
        sections = []
        section_stack: List[ParsedSection] = []
        current_content_parts: List[str] = []

        def save_current_content():
            """Save accumulated content to current section."""
            if current_content_parts and section_stack:
                section_stack[-1].content += "\n\n".join(current_content_parts)
                current_content_parts.clear()

        def process_element(element):
            """Process a single element."""
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    current_content_parts.append(text)
                return

            if not isinstance(element, Tag):
                return

            tag_name = element.name.lower()

            # Handle headings
            if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                save_current_content()

                level = int(tag_name[1])
                title = element.get_text(strip=True)

                new_section = ParsedSection(
                    level=level,
                    title=title,
                    content="",
                    children=[],
                )

                # Find parent
                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()

                if section_stack:
                    section_stack[-1].children.append(new_section)
                else:
                    sections.append(new_section)

                section_stack.append(new_section)
                return

            # Handle code blocks
            if tag_name == "pre":
                code = element.find("code")
                if code:
                    lang = ""
                    code_classes = code.get("class", [])
                    for cls in code_classes:
                        if cls.startswith("language-"):
                            lang = cls[9:]
                            break
                        elif cls.startswith("lang-"):
                            lang = cls[5:]
                            break

                    code_text = code.get_text()
                    code_block = f"```{lang}\n{code_text}\n```"
                    current_content_parts.append(code_block)
                else:
                    code_text = element.get_text()
                    current_content_parts.append(f"```\n{code_text}\n```")
                return

            # Handle inline code
            if tag_name == "code" and element.parent.name != "pre":
                current_content_parts.append(f"`{element.get_text()}`")
                return

            # Handle paragraphs
            if tag_name == "p":
                text = self._extract_text_with_formatting(element)
                if text:
                    current_content_parts.append(text)
                return

            # Handle lists
            if tag_name in ("ul", "ol"):
                list_text = self._extract_list(element, tag_name == "ol")
                if list_text:
                    current_content_parts.append(list_text)
                return

            # Handle tables
            if tag_name == "table":
                table_text = self._extract_table(element)
                if table_text:
                    current_content_parts.append(table_text)
                return

            # Handle blockquotes
            if tag_name == "blockquote":
                quote_text = element.get_text(strip=True)
                if quote_text:
                    quoted_lines = [f"> {line}" for line in quote_text.split("\n")]
                    current_content_parts.append("\n".join(quoted_lines))
                return

            # Recurse into other elements
            for child in element.children:
                process_element(child)

        # Process all elements
        for child in content.children:
            process_element(child)

        # Save any remaining content
        save_current_content()

        # If no sections found, create one from all content
        if not sections and current_content_parts:
            sections.append(
                ParsedSection(
                    level=0,
                    title="",
                    content="\n\n".join(current_content_parts),
                    children=[],
                )
            )

        return sections

    def _extract_text_with_formatting(self, element: "Tag") -> str:
        """Extract text preserving basic formatting.

        Args:
            element: HTML element

        Returns:
            Formatted text
        """
        parts = []

        for child in element.children:
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif isinstance(child, Tag):
                tag_name = child.name.lower()
                text = child.get_text()

                if tag_name in ("strong", "b"):
                    parts.append(f"**{text}**")
                elif tag_name in ("em", "i"):
                    parts.append(f"*{text}*")
                elif tag_name == "code":
                    parts.append(f"`{text}`")
                elif tag_name == "a":
                    href = child.get("href", "")
                    parts.append(f"[{text}]({href})")
                else:
                    parts.append(text)

        return "".join(parts).strip()

    def _extract_list(self, element: "Tag", ordered: bool = False) -> str:
        """Extract list as markdown.

        Args:
            element: List element (ul or ol)
            ordered: Whether it's an ordered list

        Returns:
            Markdown list text
        """
        items = []
        for i, li in enumerate(element.find_all("li", recursive=False), 1):
            prefix = f"{i}. " if ordered else "- "
            text = li.get_text(strip=True)
            items.append(f"{prefix}{text}")

        return "\n".join(items)

    def _extract_table(self, element: "Tag") -> str:
        """Extract table as markdown.

        Args:
            element: Table element

        Returns:
            Markdown table text
        """
        rows = []

        # Extract header
        thead = element.find("thead")
        if thead:
            header_cells = []
            for th in thead.find_all(["th", "td"]):
                header_cells.append(th.get_text(strip=True))
            if header_cells:
                rows.append("| " + " | ".join(header_cells) + " |")
                rows.append("|" + "|".join(["---"] * len(header_cells)) + "|")

        # Extract body
        tbody = element.find("tbody") or element
        for tr in tbody.find_all("tr"):
            cells = []
            for td in tr.find_all(["td", "th"]):
                cells.append(td.get_text(strip=True))
            if cells:
                # Add header separator if no thead
                if len(rows) == 0:
                    rows.append("| " + " | ".join(cells) + " |")
                    rows.append("|" + "|".join(["---"] * len(cells)) + "|")
                else:
                    rows.append("| " + " | ".join(cells) + " |")

        return "\n".join(rows)
