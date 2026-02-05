# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Base class for documentation framework nav path resolvers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseNavResolver(ABC):
    """Abstract base for framework-specific nav path resolvers.

    Each resolver parses a framework's config to build a mapping
    from doc_path to navigation path (breadcrumb hierarchy).
    """

    @abstractmethod
    def resolve(
        self,
        config_content: str,
        file_paths: List[str],
        content_root: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """Build mapping from doc_path -> nav_path.

        Args:
            config_content: Raw content of the framework config file
            file_paths: All document file paths collected by the fetcher
            content_root: Content root directory (e.g., "docs/en/")
            extra_context: Additional context (e.g., frontmatter data for Hugo)

        Returns:
            Dict mapping file_path to nav_path list.
            e.g., {"docs/en/sql-reference/ddl/CREATE_TABLE.md":
                   ["SQL Reference", "DDL", "CREATE TABLE"]}
        """

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path separators to forward slashes.

        Ensures consistent path handling across different operating systems.

        Args:
            path: File path that may contain backslashes

        Returns:
            Path with all backslashes converted to forward slashes
        """
        return path.replace("\\", "/")

    def _strip_content_root(self, file_path: str, content_root: str) -> str:
        """Strip content root prefix from a file path.

        Args:
            file_path: Full file path (e.g., "docs/en/sql-reference/ddl/CREATE_TABLE.md")
            content_root: Content root (e.g., "docs/en/")

        Returns:
            Relative path (e.g., "sql-reference/ddl/CREATE_TABLE.md")
        """
        # Normalize both paths to use forward slashes
        file_path = self._normalize_path(file_path)
        content_root = self._normalize_path(content_root)

        if content_root and file_path.startswith(content_root):
            return file_path[len(content_root) :]
        return file_path

    def _strip_extension(self, path: str) -> str:
        """Strip common doc file extensions.

        Args:
            path: File path or segment

        Returns:
            Path without extension
        """
        for ext in (".md", ".mdx", ".rst", ".html", ".htm", ".adoc"):
            if path.endswith(ext):
                return path[: -len(ext)]
        return path
