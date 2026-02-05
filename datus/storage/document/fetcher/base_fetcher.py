# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Base Fetcher Abstract Class

Defines the interface for all document fetchers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from datus.storage.document.schemas import FetchedDocument


class BaseFetcher(ABC):
    """Abstract base class for document fetchers.

    All fetchers must implement the fetch method to retrieve
    documents from their respective sources.

    Attributes:
        platform: Target platform name
        version: Target version (optional)
    """

    def __init__(
        self,
        platform: str,
        version: Optional[str] = None,
    ):
        """Initialize the fetcher.

        Args:
            platform: Target platform name (e.g., "snowflake")
            version: Target version (optional, will be auto-detected if not provided)
        """
        self.platform = platform
        self.version = version

    @abstractmethod
    def fetch(self, source: str, **kwargs) -> List[FetchedDocument]:
        """Fetch documents from the source.

        Args:
            source: Source identifier (repo name for GitHub, URL for web)
            **kwargs: Additional source-specific parameters

        Returns:
            List of fetched documents
        """
        pass

    @abstractmethod
    def fetch_single(self, path: str, **kwargs) -> Optional[FetchedDocument]:
        """Fetch a single document.

        Args:
            path: Path to the document
            **kwargs: Additional parameters

        Returns:
            Fetched document or None if not found
        """
        pass

    def _is_doc_file(self, filename: str) -> bool:
        """Check if a file is a documentation file.

        Args:
            filename: Name of the file

        Returns:
            True if the file is a documentation file
        """
        doc_extensions = {".md", ".rst", ".txt", ".html", ".htm"}
        lower_name = filename.lower()

        # Check extension
        for ext in doc_extensions:
            if lower_name.endswith(ext):
                return True

        # Special files without extension
        special_files = {"readme", "changelog", "contributing", "license"}
        name_without_ext = lower_name.rsplit(".", 1)[0] if "." in lower_name else lower_name

        return name_without_ext in special_files

    def _detect_content_type(self, filename: str, content: str) -> str:
        """Detect the content type of a document.

        Args:
            filename: Name of the file
            content: File content

        Returns:
            Content type ("markdown", "html", or "rst")
        """
        lower_name = filename.lower()

        # Check extension first
        if lower_name.endswith((".md", ".markdown")):
            return "markdown"
        if lower_name.endswith((".html", ".htm")):
            return "html"
        if lower_name.endswith(".rst"):
            return "rst"

        # Try to detect from content
        if content.strip().startswith("<!DOCTYPE") or content.strip().startswith("<html"):
            return "html"

        # Default to markdown
        return "markdown"
