# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
MkDocs nav resolver.

Parses ``mkdocs.yml`` to build a file-path → nav_path mapping.
MkDocs nav format is a nested YAML list::

    nav:
      - Home: index.md
      - User Guide:
        - Getting Started: user-guide/getting-started.md
        - Configuration: user-guide/configuration.md
      - API Reference:
        - Overview: api/overview.md
"""

from typing import Any, Dict, List, Optional

from datus.storage.document.nav_resolver.base_resolver import BaseNavResolver
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class MkDocsResolver(BaseNavResolver):
    """Resolves nav_path from MkDocs mkdocs.yml configuration."""

    def resolve(
        self,
        config_content: str,
        file_paths: List[str],
        content_root: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        if not config_content:
            return {}

        try:
            import yaml

            config = yaml.safe_load(config_content)
        except Exception as e:
            logger.warning(f"Failed to parse mkdocs.yml: {e}")
            return {}

        if not isinstance(config, dict):
            return {}

        # Read docs_dir to refine content_root
        docs_dir = config.get("docs_dir", "docs")
        if docs_dir and not content_root:
            content_root = docs_dir.rstrip("/") + "/"

        # Parse nav section
        nav = config.get("nav")
        if not nav or not isinstance(nav, list):
            logger.info("No 'nav' section found in mkdocs.yml")
            return {}

        # Build mapping: relative_path -> nav_path
        relative_map: Dict[str, List[str]] = {}
        self._walk_nav(nav, [], relative_map)

        # Match to actual file_paths
        nav_map: Dict[str, List[str]] = {}
        path_index: Dict[str, str] = {}
        for fp in file_paths:
            relative = self._strip_content_root(fp, content_root)
            path_index[relative] = fp

        for rel_path, nav_path in relative_map.items():
            if rel_path in path_index:
                nav_map[path_index[rel_path]] = nav_path
            else:
                # Try matching with different content_root interpretations
                for indexed_rel, fp in path_index.items():
                    if indexed_rel.endswith(rel_path) or rel_path.endswith(indexed_rel):
                        nav_map[fp] = nav_path
                        break

        logger.info(f"MkDocs resolver mapped {len(nav_map)}/{len(file_paths)} files")
        return nav_map

    def _walk_nav(
        self,
        nav_items: List[Any],
        parent_path: List[str],
        result: Dict[str, List[str]],
    ) -> None:
        """Recursively walk MkDocs nav items.

        MkDocs nav format:
        - Each item is a dict with one key (label) mapping to either:
          - a string (file path): ``{"Getting Started": "getting-started.md"}``
          - a list (children): ``{"User Guide": [{...}, ...]}``
        """
        for item in nav_items:
            if isinstance(item, dict):
                for label, value in item.items():
                    if isinstance(value, str):
                        # Leaf node: label -> file path
                        result[value] = list(parent_path)
                    elif isinstance(value, list):
                        # Category node: label -> children
                        child_path = parent_path + [label]
                        self._walk_nav(value, child_path, result)
            elif isinstance(item, str):
                # Bare string (no label), uncommon but possible
                result[item] = list(parent_path)
