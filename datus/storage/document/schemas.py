# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Data models for Platform Documentation module.

This module defines the data structures used throughout the platform documentation
pipeline: fetching, parsing, chunking, and storage.
"""

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# =============================================================================
# Fetching Stage Models
# =============================================================================


@dataclass
class FetchedDocument:
    """Result of fetching a document from GitHub or website.

    Attributes:
        platform: Target platform (e.g., "snowflake", "duckdb")
        version: Document version (e.g., "v1.2.3" or "2025-01-29")
        source_url: Full URL of the source
        source_type: Source type ("github" or "website")
        doc_path: Path within the repository or site (e.g., "docs/guide/intro.md")
        raw_content: Raw content (Markdown or HTML)
        content_type: Content format ("markdown" or "html")
        metadata: Additional metadata (e.g., last_modified, commit_sha)
        fetch_timestamp: When the document was fetched
    """

    platform: str
    version: str
    source_url: str
    source_type: str  # "github" | "website"
    doc_path: str
    raw_content: str
    content_type: str  # "markdown" | "html"
    metadata: Dict[str, Any] = field(default_factory=dict)
    fetch_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Parsing Stage Models
# =============================================================================


@dataclass
class ParsedSection:
    """A section of a parsed document.

    Represents a hierarchical section with a title and content,
    potentially containing nested child sections.

    Attributes:
        level: Heading level (1-6, where 1 is h1)
        title: Section title text
        content: Text content of this section (excluding children)
        children: Nested subsections
    """

    level: int
    title: str
    content: str
    children: List["ParsedSection"] = field(default_factory=list)

    def get_all_content(self) -> str:
        """Get all content including children."""
        parts = [self.content]
        for child in self.children:
            parts.append(child.get_all_content())
        return "\n\n".join(filter(None, parts))


@dataclass
class ParsedDocument:
    """Result of parsing a document.

    Attributes:
        title: Document title (usually the first h1)
        sections: Top-level sections
        metadata: Extracted metadata. May include:
            - nav_path: List[str] - Site navigation path from breadcrumb/sidebar
              (e.g., ["User Guide", "Data Loading", "Snowpipe"])
            - description: str - Page description from meta tags
            - author: str - Document author
            - keywords: List[str] - Meta keywords
        source_doc: Reference to the original fetched document
    """

    title: str
    sections: List[ParsedSection]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_doc: Optional[FetchedDocument] = None

    def get_section_titles(self) -> List[str]:
        """Get all section titles recursively."""
        titles = []

        def collect_titles(section: ParsedSection):
            if section.title:
                titles.append(section.title)
            for child in section.children:
                collect_titles(child)

        for section in self.sections:
            collect_titles(section)
        return titles


# =============================================================================
# Chunking Stage Models
# =============================================================================


@dataclass
class PlatformDocChunk:
    """A chunk of platform documentation ready for storage.

    This is the final output format that will be stored in the vector database.

    Attributes:
        chunk_id: Unique identifier (MD5 hash)
        chunk_text: Text content of this chunk
        chunk_index: Index of this chunk within the document

        title: Current section title (the immediate heading)
        titles: Page-internal heading hierarchy (h1→h2→h3)
            e.g., ["Snowpipe", "Overview", "Key Features"]
        nav_path: Site navigation path from sidebar/breadcrumb
            e.g., ["Guides", "User Guide", "Loading Data", "Snowpipe"]
        group_name: Top-level documentation group/category
            e.g., "Guides", "Get Started", "Developers", or "" if none
        hierarchy: Full combined path (auto-generated for display/search)
            e.g., "Guides > User Guide > Loading Data > Snowpipe > Overview"

        version: Document version
        source_type: Source type (github/website)
        source_url: Full source URL
        doc_path: Path within repository/site

        keywords: Extracted keywords for search
        language: Language code (en, zh, etc.)
        content_hash: MD5 hash of the raw document content (same for all chunks from one doc)
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    chunk_id: str
    chunk_text: str
    chunk_index: int

    title: str
    titles: List[str]  # Page-internal headings (h1, h2, h3...)
    nav_path: List[str]  # Site navigation path
    group_name: str  # Top-level group (Guides, Get Started, etc.)
    hierarchy: str  # Full combined path for display

    version: str
    source_type: str
    source_url: str
    doc_path: str

    keywords: List[str] = field(default_factory=list)
    language: str = "en"
    content_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)

    @staticmethod
    def generate_chunk_id(
        doc_path: str,
        chunk_index: int,
        version: str,
    ) -> str:
        """Generate a unique chunk ID.

        The ID is deterministic based on path, index, and version,
        ensuring idempotent upserts.

        Args:
            doc_path: Document path
            chunk_index: Chunk index
            version: Document version

        Returns:
            MD5 hash string
        """
        key = f"{doc_path}:{chunk_index}:{version}"
        return hashlib.md5(key.encode()).hexdigest()


# =============================================================================
# Constants
# =============================================================================


# Source types
SOURCE_TYPE_GITHUB = "github"
SOURCE_TYPE_WEBSITE = "website"
SOURCE_TYPE_LOCAL = "local"

# Content types
CONTENT_TYPE_MARKDOWN = "markdown"
CONTENT_TYPE_HTML = "html"
CONTENT_TYPE_RST = "rst"

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_MIN_CHUNK_SIZE = 256
DEFAULT_MAX_CHUNK_SIZE = 2048
