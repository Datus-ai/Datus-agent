# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Documentation framework detector.

Probes a GitHub repository to identify which documentation framework
(Docusaurus, Hugo, MkDocs, etc.) is used and locate its config file.
"""

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from github.Repository import Repository

    from datus.storage.document.fetcher.rate_limiter import RateLimiter

logger = get_logger(__name__)

FRAMEWORK_DOCUSAURUS = "docusaurus"
FRAMEWORK_HUGO = "hugo"
FRAMEWORK_MKDOCS = "mkdocs"
FRAMEWORK_UNKNOWN = "unknown"


@dataclass
class FrameworkInfo:
    """Result of documentation framework detection."""

    framework: str = FRAMEWORK_UNKNOWN
    config_path: str = ""
    content_root: str = ""
    config_content: str = ""


@dataclass
class _ProbeSpec:
    """Specification for probing a framework."""

    framework: str
    config_candidates: List[str]
    content_root_candidates: List[str] = field(default_factory=list)


# Probe order matters: MkDocs first (root-level config), then Hugo, then Docusaurus
_PROBE_SPECS = [
    _ProbeSpec(
        framework=FRAMEWORK_MKDOCS,
        config_candidates=["mkdocs.yml", "mkdocs.yaml"],
        content_root_candidates=["docs/"],
    ),
    _ProbeSpec(
        framework=FRAMEWORK_HUGO,
        config_candidates=["hugo.yaml", "hugo.toml", "hugo.json", "site/hugo.yaml", "site/hugo.toml"],
        content_root_candidates=["site/content/", "content/"],
    ),
    _ProbeSpec(
        framework=FRAMEWORK_DOCUSAURUS,
        config_candidates=[
            "sidebars.json",
            "sidebars.js",
            "docs/docusaurus/sidebars.json",
            "website/sidebars.json",
            "website/sidebars.js",
        ],
        content_root_candidates=["docs/en/", "docs/", "website/docs/"],
    ),
]


class DocFrameworkDetector:
    """Detects which documentation framework a GitHub repository uses.

    Strategy:
    1. Fetch root directory listing (1 API call) to get visible file names
    2. For each framework probe spec, check if config candidates exist
    3. On match, fetch config content and detect content root
    4. Returns FrameworkInfo with framework type, config path, content root, and config content

    Typical API cost: 1-3 calls.
    """

    def detect(
        self,
        repo: "Repository",
        branch: str,
        rate_limiter: Optional["RateLimiter"] = None,
    ) -> FrameworkInfo:
        """Detect documentation framework in the repository.

        Args:
            repo: PyGithub Repository object
            branch: Branch name to inspect
            rate_limiter: Optional rate limiter for API calls

        Returns:
            FrameworkInfo with detected framework details
        """
        # Fetch root listing to know what exists
        root_names = self._get_dir_names(repo, "", branch, rate_limiter)
        if root_names is None:
            return FrameworkInfo()

        for spec in _PROBE_SPECS:
            result = self._try_probe(repo, branch, rate_limiter, spec, root_names)
            if result:
                return result

        logger.info("No documentation framework detected, will use fallback resolver")
        return FrameworkInfo()

    def _try_probe(
        self,
        repo: "Repository",
        branch: str,
        rate_limiter: Optional["RateLimiter"],
        spec: _ProbeSpec,
        root_names: set,
    ) -> Optional[FrameworkInfo]:
        """Try to detect a specific framework."""
        for candidate in spec.config_candidates:
            # Quick check: if the candidate is at root level, verify it exists
            parts = candidate.split("/")
            if parts[0] not in root_names and len(parts) == 1:
                continue
            if len(parts) > 1 and parts[0] not in root_names:
                continue

            # Try to fetch the config file
            content = self._fetch_file_content(repo, candidate, branch, rate_limiter)
            if content is not None:
                # Detect content root
                content_root = self._detect_content_root(repo, branch, rate_limiter, spec, root_names)

                logger.info(
                    f"Detected framework: {spec.framework}, " f"config: {candidate}, content_root: {content_root}"
                )
                return FrameworkInfo(
                    framework=spec.framework,
                    config_path=candidate,
                    content_root=content_root,
                    config_content=content,
                )

        return None

    def _detect_content_root(
        self,
        repo: "Repository",
        branch: str,
        rate_limiter: Optional["RateLimiter"],
        spec: _ProbeSpec,
        root_names: set,
    ) -> str:
        """Detect the content root directory for the framework."""
        for candidate in spec.content_root_candidates:
            # Check if the first directory segment exists in root listing
            first_dir = candidate.split("/")[0]
            if first_dir not in root_names:
                continue

            # For nested paths (e.g., "docs/en/"), verify the full path exists
            candidate_path = candidate.rstrip("/")
            if "/" in candidate_path:
                # Need to verify the nested directory actually exists
                dir_names = self._get_dir_names(repo, candidate_path, branch, rate_limiter)
                if dir_names is None:
                    continue  # Directory doesn't exist, try next candidate

            return candidate
        return ""

    def _get_dir_names(
        self,
        repo: "Repository",
        path: str,
        branch: str,
        rate_limiter: Optional["RateLimiter"],
    ) -> Optional[set]:
        """Get names of files/dirs at the given path."""
        try:
            if rate_limiter:
                rate_limiter.wait("api.github.com")
            contents = repo.get_contents(path, ref=branch)
            if isinstance(contents, list):
                return {item.name for item in contents}
            return {contents.name}
        except Exception as e:
            logger.warning(f"Failed to list directory '{path}': {e}")
            return None

    def _fetch_file_content(
        self,
        repo: "Repository",
        path: str,
        branch: str,
        rate_limiter: Optional["RateLimiter"],
    ) -> Optional[str]:
        """Fetch a single file's content from the repo."""
        try:
            if rate_limiter:
                rate_limiter.wait("api.github.com")
            content_file = repo.get_contents(path, ref=branch)
            if hasattr(content_file, "content") and content_file.content:
                return base64.b64decode(content_file.content).decode("utf-8")
            # For large files, use the download URL
            if hasattr(content_file, "download_url") and content_file.download_url:
                import urllib.request
                from urllib.parse import urlparse

                parsed = urlparse(content_file.download_url)
                if parsed.scheme not in ("http", "https"):
                    return None
                with urllib.request.urlopen(content_file.download_url, timeout=30) as resp:
                    return resp.read().decode("utf-8")
            return None
        except Exception:
            return None
