# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Documentation Navigation Resolver

Auto-detects documentation frameworks (Docusaurus, Hugo, MkDocs) in GitHub
repositories and resolves nav_path for each document file.
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from datus.storage.document.nav_resolver.base_resolver import BaseNavResolver
from datus.storage.document.nav_resolver.detector import (
    FRAMEWORK_DOCUSAURUS,
    FRAMEWORK_HUGO,
    FRAMEWORK_MKDOCS,
    DocFrameworkDetector,
    FrameworkInfo,
)
from datus.storage.document.nav_resolver.docusaurus_resolver import DocusaurusResolver
from datus.storage.document.nav_resolver.fallback_resolver import FallbackResolver
from datus.storage.document.nav_resolver.hugo_resolver import HugoResolver
from datus.storage.document.nav_resolver.mkdocs_resolver import MkDocsResolver
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from github.Repository import Repository

    from datus.storage.document.fetcher.rate_limiter import RateLimiter
    from datus.storage.document.schemas import FetchedDocument

logger = get_logger(__name__)

RESOLVER_MAP: Dict[str, type] = {
    FRAMEWORK_DOCUSAURUS: DocusaurusResolver,
    FRAMEWORK_HUGO: HugoResolver,
    FRAMEWORK_MKDOCS: MkDocsResolver,
}

# Simple frontmatter extraction pattern for Hugo context building
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


class NavResolverPipeline:
    """Orchestrates framework detection and nav_path resolution.

    Usage inside GitHubFetcher::

        pipeline = NavResolverPipeline()
        nav_map = pipeline.resolve(repo, branch, file_paths, rate_limiter, documents)
        for doc in documents:
            doc.metadata["nav_path"] = nav_map.get(doc.doc_path, [])
    """

    def resolve(
        self,
        repo: "Repository",
        branch: str,
        file_paths: List[str],
        rate_limiter: Optional["RateLimiter"] = None,
        fetched_docs: Optional[List["FetchedDocument"]] = None,
    ) -> Dict[str, List[str]]:
        """Detect framework and resolve nav_path for all files.

        Args:
            repo: PyGithub Repository object
            branch: Branch name
            file_paths: List of document file paths
            rate_limiter: Optional rate limiter
            fetched_docs: Already-fetched documents (used for Hugo frontmatter)

        Returns:
            Dict mapping file_path -> nav_path list
        """
        # 1. Detect framework
        detector = DocFrameworkDetector()
        info = detector.detect(repo, branch, rate_limiter)

        # 2. Pick resolver
        resolver_cls = RESOLVER_MAP.get(info.framework, FallbackResolver)
        resolver: BaseNavResolver = resolver_cls()

        # 3. Build extra context for Hugo (frontmatter from fetched docs)
        extra_context = None
        if info.framework == FRAMEWORK_HUGO and fetched_docs:
            extra_context = self._extract_frontmatter_context(fetched_docs)

        # 4. Resolve nav paths
        nav_map = resolver.resolve(
            config_content=info.config_content,
            file_paths=file_paths,
            content_root=info.content_root,
            extra_context=extra_context,
        )

        # 5. Fill gaps with fallback for unmapped files
        unmapped = [p for p in file_paths if p not in nav_map]
        if unmapped:
            fallback = FallbackResolver()
            fallback_map = fallback.resolve("", unmapped, info.content_root)
            nav_map.update(fallback_map)

        logger.info(f"Nav resolution complete: framework={info.framework}, " f"mapped={len(nav_map)}/{len(file_paths)}")
        return nav_map

    @staticmethod
    def _extract_frontmatter_context(
        docs: List["FetchedDocument"],
    ) -> Dict[str, Dict[str, Any]]:
        """Extract frontmatter metadata from fetched documents.

        Used by Hugo resolver to read titles and weights from _index.md files.

        Returns:
            Dict mapping doc_path -> frontmatter dict
        """
        context: Dict[str, Dict[str, Any]] = {}
        for doc in docs:
            if doc.content_type not in ("markdown", "rst"):
                continue
            fm = _parse_simple_frontmatter(doc.raw_content)
            if fm:
                context[doc.doc_path] = fm
        return context


def _parse_simple_frontmatter(content: str) -> Dict[str, str]:
    """Quick frontmatter extraction without full YAML parsing.

    Handles simple ``key: value`` pairs in YAML frontmatter.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}

    metadata: Dict[str, str] = {}
    for line in match.group(1).split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                metadata[key] = value

    return metadata


__all__ = [
    "NavResolverPipeline",
    "DocFrameworkDetector",
    "FrameworkInfo",
    "BaseNavResolver",
    "DocusaurusResolver",
    "HugoResolver",
    "MkDocsResolver",
    "FallbackResolver",
]
