# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Hugo documentation nav resolver.

Hugo uses a distributed navigation model:
- Directory structure defines the hierarchy
- ``_index.md`` files define section titles and ordering via frontmatter
- Individual pages use ``weight`` and ``title`` in frontmatter for ordering/naming

This resolver builds nav_path from the directory structure by reading
``_index.md`` titles from already-fetched documents.
"""

import re
from typing import Any, Dict, List, Optional

from datus.storage.document.nav_resolver.base_resolver import BaseNavResolver
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_SEPARATOR_RE = re.compile(r"[-_]+")


class HugoResolver(BaseNavResolver):
    """Resolves nav_path from Hugo directory structure and frontmatter.

    Requires ``extra_context`` containing a dict mapping file paths to their
    frontmatter metadata (extracted from already-fetched documents).
    """

    def resolve(
        self,
        config_content: str,  # noqa: ARG002
        file_paths: List[str],
        content_root: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        # extra_context: {file_path: {"title": "...", "weight": 100, ...}}
        frontmatter_map = extra_context or {}

        # Build section title index from _index.md files
        # Maps directory path -> title (from _index.md frontmatter)
        section_titles: Dict[str, str] = {}
        for fp, fm in frontmatter_map.items():
            relative = self._strip_content_root(fp, content_root)
            basename = relative.rsplit("/", 1)[-1] if "/" in relative else relative
            if basename in ("_index.md", "_index.adoc"):
                # Directory path for this _index.md
                dir_path = relative.rsplit("/", 1)[0] if "/" in relative else ""
                title = fm.get("linkTitle") or fm.get("title") or ""
                if title:
                    section_titles[dir_path] = title

        # Build nav_path for each file
        nav_map: Dict[str, List[str]] = {}
        for fp in file_paths:
            relative = self._strip_content_root(fp, content_root)
            basename = relative.rsplit("/", 1)[-1] if "/" in relative else relative

            # Skip _index.md files themselves (they define sections, not pages)
            if basename in ("_index.md", "_index.adoc"):
                # For _index.md, nav_path is its parent directories
                dir_path = relative.rsplit("/", 1)[0] if "/" in relative else ""
                nav_path = self._build_nav_path(dir_path, section_titles, parent_only=True)
            else:
                # For regular files, nav_path includes the directory hierarchy
                dir_path = relative.rsplit("/", 1)[0] if "/" in relative else ""
                nav_path = self._build_nav_path(dir_path, section_titles, parent_only=False)

            nav_map[fp] = nav_path

        logger.info(f"Hugo resolver mapped {len(nav_map)}/{len(file_paths)} files")
        return nav_map

    def _build_nav_path(
        self,
        dir_path: str,
        section_titles: Dict[str, str],
        parent_only: bool = False,
    ) -> List[str]:
        """Build nav_path from directory path using section titles.

        For a file at ``getting-started/quick-start.md``, the dir_path is
        ``getting-started``. We look up section titles for each ancestor directory.

        Args:
            dir_path: Directory path relative to content root
            section_titles: Map of dir_path -> section title from _index.md
            parent_only: If True, exclude the dir_path itself (for _index.md files)
        """
        if not dir_path:
            return []

        parts = dir_path.split("/")
        nav_path = []

        # Build cumulative path and look up title for each level
        for i in range(len(parts)):
            if parent_only and i == len(parts) - 1:
                break
            cumulative = "/".join(parts[: i + 1])
            title = section_titles.get(cumulative)
            if title:
                nav_path.append(title)
            else:
                # No _index.md for this directory, use cleaned dir name
                nav_path.append(self._clean_segment(parts[i]))

        return nav_path

    @staticmethod
    def _clean_segment(segment: str) -> str:
        """Clean a directory name into a human-readable label."""
        cleaned = _SEPARATOR_RE.sub(" ", segment)
        words = cleaned.split()
        result = []
        for word in words:
            if word.isupper() and len(word) > 1:
                result.append(word)
            else:
                result.append(word.capitalize())
        return " ".join(result)
