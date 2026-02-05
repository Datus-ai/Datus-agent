# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Fallback nav resolver that derives nav_path from directory structure.

Used when no documentation framework is detected, or for files
not covered by a framework's nav config.
"""

import re
from typing import Any, Dict, List, Optional

from datus.storage.document.nav_resolver.base_resolver import BaseNavResolver

# Segments to skip in nav_path (common boilerplate directories)
_SKIP_SEGMENTS = {"docs", "doc", "en", "zh", "content", "src", "site", "pages"}

# Pattern to clean segment names
_SEPARATOR_RE = re.compile(r"[-_]+")


class FallbackResolver(BaseNavResolver):
    """Derives nav_path from file directory structure.

    Transforms a file path like ``docs/en/sql-reference/ddl/CREATE_TABLE.md``
    into ``["SQL Reference", "DDL"]`` by:
    1. Stripping the content root prefix
    2. Removing the filename (last segment becomes part of titles, not nav)
    3. Cleaning directory names (replace separators, title-case)
    """

    def resolve(
        self,
        config_content: str,  # noqa: ARG002
        file_paths: List[str],
        content_root: str,
        extra_context: Optional[Dict[str, Any]] = None,  # noqa: ARG002
    ) -> Dict[str, List[str]]:
        nav_map: Dict[str, List[str]] = {}
        for file_path in file_paths:
            nav_map[file_path] = self._path_to_nav(file_path, content_root)
        return nav_map

    def _path_to_nav(self, file_path: str, content_root: str) -> List[str]:
        """Convert a file path to a nav_path list."""
        relative = self._strip_content_root(file_path, content_root)

        # Split into segments and remove filename
        parts = relative.replace("\\", "/").split("/")
        if not parts:
            return []

        # Remove the filename (last segment) - it becomes part of titles, not nav
        dir_parts = parts[:-1]

        nav_path = []
        for segment in dir_parts:
            # Skip common boilerplate directory names
            if segment.lower() in _SKIP_SEGMENTS:
                continue
            cleaned = self._clean_segment(segment)
            if cleaned:
                nav_path.append(cleaned)

        return nav_path

    @staticmethod
    def _clean_segment(segment: str) -> str:
        """Clean a directory name into a human-readable label.

        Examples:
            "sql-reference" -> "SQL Reference"
            "getting_started" -> "Getting Started"
            "ddl" -> "DDL"
        """
        # Replace separators with spaces
        cleaned = _SEPARATOR_RE.sub(" ", segment)
        # Title case, but preserve all-caps segments (DDL, SQL, API)
        words = cleaned.split()
        result = []
        for word in words:
            if word.isupper() and len(word) > 1:
                result.append(word)
            else:
                result.append(word.capitalize())
        return " ".join(result)
