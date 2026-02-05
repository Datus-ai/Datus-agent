# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Local Document Fetcher

Fetches documentation from local file system directories.
Supports Markdown, HTML, and other text-based documentation formats.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set

from datus.storage.document.fetcher.base_fetcher import BaseFetcher
from datus.storage.document.schemas import (
    CONTENT_TYPE_HTML,
    CONTENT_TYPE_MARKDOWN,
    CONTENT_TYPE_RST,
    SOURCE_TYPE_LOCAL,
    FetchedDocument,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class LocalFetcher(BaseFetcher):
    """Fetcher for local documentation files.

    Reads documentation files from local directories.
    Supports:
    - Recursive directory traversal
    - File pattern filtering (include/exclude)
    - Multiple file formats (Markdown, HTML, RST)

    Example:
        >>> fetcher = LocalFetcher(platform="custom")
        >>> docs = fetcher.fetch("/path/to/docs", recursive=True)
    """

    # Supported documentation file extensions
    DOC_EXTENSIONS = {".md", ".markdown", ".html", ".htm", ".rst", ".txt"}

    def __init__(
        self,
        platform: str,
        version: Optional[str] = None,
    ):
        """Initialize the local fetcher.

        Args:
            platform: Platform name for the documentation
            version: Version string (optional, defaults to current date)
        """
        super().__init__(platform=platform, version=version)

    def fetch(
        self,
        source: str,
        recursive: bool = True,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        **kwargs,
    ) -> List[FetchedDocument]:
        """Fetch documentation from a local directory.

        Args:
            source: Path to the directory containing documentation
            recursive: Whether to recursively scan subdirectories
            include_patterns: File patterns to include (e.g., ["*.md"])
            exclude_patterns: File patterns to exclude (e.g., ["README.md"])
            **kwargs: Additional parameters (unused)

        Returns:
            List of fetched documents
        """
        source_path = Path(source).resolve()

        if not source_path.exists():
            logger.error(f"Directory not found: {source}")
            return []

        if not source_path.is_dir():
            logger.error(f"Not a directory: {source}")
            return []

        # Default version to current date if not provided
        version = self.version or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        documents = []
        visited_files: Set[Path] = set()

        # Collect all documentation files
        if recursive:
            file_iterator = source_path.rglob("*")
        else:
            file_iterator = source_path.glob("*")

        for file_path in file_iterator:
            # Skip directories
            if file_path.is_dir():
                continue

            # Skip non-documentation files
            if file_path.suffix.lower() not in self.DOC_EXTENSIONS:
                continue

            # Skip already processed files (in case of symlinks)
            resolved_path = file_path.resolve()
            if resolved_path in visited_files:
                continue
            visited_files.add(resolved_path)

            # Check include patterns
            if include_patterns:
                if not any(file_path.match(p) for p in include_patterns):
                    continue

            # Check exclude patterns
            if exclude_patterns:
                if any(file_path.match(p) for p in exclude_patterns):
                    continue

            # Fetch the file
            doc = self._fetch_file(file_path, source_path, version)
            if doc:
                documents.append(doc)

        logger.info(f"Fetched {len(documents)} documents from {source_path}")
        return documents

    def fetch_single(
        self,
        path: str,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> Optional[FetchedDocument]:
        """Fetch a single document file.

        Args:
            path: Path to the document file
            base_url: Base directory path (for relative path calculation)
            **kwargs: Additional parameters (unused)

        Returns:
            Fetched document or None if not found
        """
        file_path = Path(path).resolve()

        if not file_path.exists():
            logger.warning(f"File not found: {path}")
            return None

        if not file_path.is_file():
            logger.warning(f"Not a file: {path}")
            return None

        # Determine base path for relative path calculation
        if base_url:
            base_path = Path(base_url).resolve()
        else:
            base_path = file_path.parent

        # Default version to current date if not provided
        version = self.version or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return self._fetch_file(file_path, base_path, version)

    def _fetch_file(
        self,
        file_path: Path,
        base_path: Path,
        version: str,
    ) -> Optional[FetchedDocument]:
        """Fetch a single file and create a FetchedDocument.

        Args:
            file_path: Path to the file
            base_path: Base directory for relative path calculation
            version: Version string

        Returns:
            FetchedDocument or None if reading fails
        """
        try:
            # Read file content
            content = file_path.read_text(encoding="utf-8")

            # Skip empty files
            if not content.strip():
                logger.debug(f"Skipping empty file: {file_path}")
                return None

            # Calculate relative path from base
            try:
                doc_path = str(file_path.relative_to(base_path))
            except ValueError:
                # If file is outside base_path, use filename
                doc_path = file_path.name

            # Detect content type
            content_type = self._detect_content_type(file_path.name, content)

            # Extract title from content (first heading or filename)
            title = self._extract_title(content, file_path.name, content_type)

            # Get file modification time
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)

            # Create source URL as file:// URI
            source_url = file_path.as_uri()

            return FetchedDocument(
                platform=self.platform,
                version=version,
                source_url=source_url,
                source_type=SOURCE_TYPE_LOCAL,
                doc_path=doc_path,
                raw_content=content,
                content_type=content_type,
                metadata={
                    "title": title,
                    "file_name": file_path.name,
                    "file_size": len(content),
                    "last_modified": mtime.isoformat(),
                    "absolute_path": str(file_path),
                },
            )

        except UnicodeDecodeError:
            logger.warning(f"Cannot decode file (not UTF-8): {file_path}")
            return None
        except PermissionError:
            logger.warning(f"Permission denied: {file_path}")
            return None
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return None

    def _extract_title(
        self,
        content: str,
        filename: str,
        content_type: str,
    ) -> str:
        """Extract document title from content.

        Args:
            content: File content
            filename: Filename for fallback
            content_type: Content type (markdown/html)

        Returns:
            Extracted title or filename-based fallback
        """
        lines = content.strip().split("\n")

        if content_type in (CONTENT_TYPE_MARKDOWN, CONTENT_TYPE_RST):
            # Look for first heading
            # Markdown ATX-style: # Title
            # Markdown/RST setext-style: Title followed by === or ---
            for idx, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("# "):
                    return stripped[2:].strip()
                # Check for setext-style heading (used by both Markdown and RST)
                if stripped and idx + 1 < len(lines):
                    next_line = lines[idx + 1].strip()
                    # RST allows =, -, ~, ^, etc. as underline chars
                    if next_line and len(next_line) >= 3 and next_line[0] in "=-~^":
                        if all(c == next_line[0] for c in next_line):
                            return stripped

        elif content_type == CONTENT_TYPE_HTML:
            # Simple title extraction from HTML
            import re

            title_match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
            if title_match:
                return title_match.group(1).strip()

            # Try h1
            h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", content, re.IGNORECASE | re.DOTALL)
            if h1_match:
                # Remove any HTML tags inside
                return re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()

        # Fallback: use filename without extension
        return Path(filename).stem.replace("_", " ").replace("-", " ").title()
