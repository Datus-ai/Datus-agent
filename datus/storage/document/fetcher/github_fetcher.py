# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GitHub Repository Document Fetcher

Fetches documentation from GitHub repositories using the GitHub API.
Supports recursive directory traversal, version detection from releases/tags,
and rate limit handling.
"""

import base64
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from datus.storage.document.fetcher.base_fetcher import BaseFetcher
from datus.storage.document.fetcher.rate_limiter import RateLimiter, get_rate_limiter
from datus.storage.document.schemas import SOURCE_TYPE_GITHUB, FetchedDocument
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Check if PyGithub is available
try:
    from github import Github, GithubException, RateLimitExceededException
    from github.Repository import Repository

    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    Github = None
    GithubException = Exception
    RateLimitExceededException = Exception


@dataclass
class GitHubFetchMetadata:
    """Lightweight metadata collected in Phase 1 of batch fetching.

    Contains everything needed to fetch content in batches without
    re-doing repo detection, version detection, or nav resolution.
    """

    repo: Any  # PyGithub Repository object
    branch: str
    version: str
    source: str
    file_paths: List[str]
    nav_map: Dict[str, List[str]] = field(default_factory=dict)


class GitHubFetcher(BaseFetcher):
    """Fetcher for GitHub repository documentation.

    Uses the GitHub API (via PyGithub) to fetch documentation files
    from repositories. Supports:
    - Recursive directory traversal
    - Version detection from releases/tags
    - Rate limit handling with backoff
    - Parallel fetching with thread pool

    Example:
        >>> fetcher = GitHubFetcher(platform="snowflake")
        >>> docs = fetcher.fetch("snowflakedb/snowflake-connector-python")
    """

    DEFAULT_PATHS = ["docs", "README.md", "CHANGELOG.md"]
    DEFAULT_BRANCH = "main"

    def __init__(
        self,
        platform: str,
        version: Optional[str] = None,
        github_ref: Optional[str] = None,
        token: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        pool_size: int = 4,
    ):
        """Initialize the GitHub fetcher.

        Args:
            platform: Target platform name
            version: Target version label (optional, auto-detected if not provided)
            github_ref: Explicit git ref (branch or tag) to fetch from.
                When set, content is fetched from this ref instead of the default branch.
                Examples: "v3.4.0" (tag), "versioned-docs" (branch).
                When None, fetches from the default branch.
            token: GitHub API token (falls back to GITHUB_TOKEN env var)
            rate_limiter: Rate limiter instance (uses global if not provided)
            pool_size: Thread pool size for parallel fetching
        """
        super().__init__(platform=platform, version=version)
        self.github_ref = github_ref

        if not GITHUB_AVAILABLE:
            raise ImportError("PyGithub is required for GitHub fetching. " "Install it with: pip install PyGithub")

        self.token = token or os.getenv("GITHUB_TOKEN")
        self.rate_limiter = rate_limiter or get_rate_limiter()
        self.pool_size = pool_size

        # Configure rate limiter for authenticated access
        if self.token:
            self.rate_limiter.configure_github_authenticated()
            self._github = Github(self.token)
            logger.info("GitHub fetcher initialized with authentication")
        else:
            self._github = Github()
            logger.warning(
                "GitHub fetcher initialized without authentication. "
                "Rate limit: 60 requests/hour. Set GITHUB_TOKEN for 5000/hour."
            )

    def fetch(
        self,
        source: str,
        branch: Optional[str] = None,
        paths: Optional[List[str]] = None,
        **kwargs,
    ) -> List[FetchedDocument]:
        """Fetch documentation from a GitHub repository.

        Uses collect_metadata() + fetch_batch() internally for a clean
        two-phase approach: metadata first, then content.

        Args:
            source: Repository in "owner/repo" format
            branch: Branch to fetch from (default: main or master)
            paths: Paths to fetch (default: docs, README.md, CHANGELOG.md)
            **kwargs: Additional parameters

        Returns:
            List of fetched documents
        """
        metadata = self.collect_metadata(source=source, branch=branch, paths=paths)
        if not metadata.file_paths:
            return []

        documents = self.fetch_batch(metadata, metadata.file_paths)
        logger.info(f"Successfully fetched {len(documents)} documents from {source}")
        return documents

    def collect_metadata(
        self,
        source: str,
        branch: Optional[str] = None,
        paths: Optional[List[str]] = None,
    ) -> GitHubFetchMetadata:
        """Phase 1: Collect file paths and resolve nav_map without fetching content.

        Lightweight operation that only collects file path strings and resolves
        navigation paths. No document content is fetched (except Hugo _index.md
        files needed for section title extraction).

        Args:
            source: Repository in "owner/repo" format
            branch: Branch to fetch from (default: auto-detect)
            paths: Paths to search (default: docs, README.md, CHANGELOG.md)

        Returns:
            GitHubFetchMetadata with repo, branch, version, file_paths, nav_map
        """
        repo = self._get_repo(source)
        paths = paths or self.DEFAULT_PATHS

        if self.github_ref:
            # Explicit git ref provided — use it as branch for fetching,
            # and as version label if not explicitly set
            branch = branch or self.github_ref
            version = self.version or self.github_ref
        else:
            branch = branch or self._detect_default_branch(repo)
            version = self.version or self._detect_version(repo, branch)

        logger.info(f"Collecting metadata from {source}@{branch}, version: {version}")

        # Collect all file paths (lightweight, no content)
        file_paths = []
        for path in paths:
            try:
                found_paths = self._collect_file_paths(repo, path, branch)
                file_paths.extend(found_paths)
            except GithubException as e:
                if e.status == 404:
                    logger.debug(f"Path not found: {path}")
                else:
                    logger.warning(f"Error accessing {path}: {e}")

        if not file_paths:
            logger.warning(f"No documentation files found in {source}")
            return GitHubFetchMetadata(
                repo=repo,
                branch=branch,
                version=version,
                source=source,
                file_paths=[],
                nav_map={},
            )

        logger.info(f"Found {len(file_paths)} documentation files")

        # Resolve nav_map using file_paths only (no full content needed)
        nav_map = self._resolve_nav_map(repo, branch, source, version, file_paths)

        return GitHubFetchMetadata(
            repo=repo,
            branch=branch,
            version=version,
            source=source,
            file_paths=file_paths,
            nav_map=nav_map,
        )

    # Pattern to detect version from path segments like "releases/0.2.0/" or "v3.4.0/"
    _VERSION_PATH_RE = re.compile(r"(?:^|/)v?(\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?)/")

    @staticmethod
    def _detect_version_from_path(file_path: str) -> Optional[str]:
        """Try to extract a version string from the file path.

        Matches patterns like:
        - releases/0.2.0/foo.md → 0.2.0
        - content/releases/v3.4.0/bar.md → 3.4.0
        - versioned_docs/1.2.3-beta/baz.md → 1.2.3-beta

        Args:
            file_path: Document file path

        Returns:
            Version string if found, None otherwise
        """
        match = GitHubFetcher._VERSION_PATH_RE.search(file_path)
        return match.group(1) if match else None

    def fetch_batch(
        self,
        metadata: GitHubFetchMetadata,
        file_paths_batch: List[str],
    ) -> List[FetchedDocument]:
        """Phase 2: Fetch content for a subset of file paths.

        Uses pre-resolved nav_map from metadata to set nav_path on each document.

        Args:
            metadata: Metadata from collect_metadata()
            file_paths_batch: Subset of file paths to fetch content for

        Returns:
            List of fetched documents with nav_path in metadata
        """
        if not file_paths_batch:
            return []

        documents = []
        with ThreadPoolExecutor(max_workers=self.pool_size) as executor:
            futures = {
                executor.submit(
                    self._fetch_file,
                    metadata.repo,
                    file_path,
                    metadata.branch,
                    metadata.version,
                    metadata.source,
                ): file_path
                for file_path in file_paths_batch
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    doc = future.result()
                    if doc:
                        # Override version from path when user didn't specify one
                        # (e.g., releases/0.2.0/foo.md → version=0.2.0)
                        if not self.version:
                            path_version = self._detect_version_from_path(doc.doc_path)
                            if path_version:
                                doc.version = path_version

                        # Apply pre-resolved nav_path
                        nav_path = metadata.nav_map.get(doc.doc_path, [])
                        doc.metadata["nav_path"] = nav_path
                        if nav_path:
                            doc.metadata["group_name"] = nav_path[0]
                        documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to fetch {file_path}: {e}")

        return documents

    def _resolve_nav_map(
        self,
        repo: "Repository",
        branch: str,
        source: str,
        version: str,
        file_paths: List[str],
    ) -> Dict[str, List[str]]:
        """Resolve nav_map for all file paths without requiring full document content.

        For Hugo repos, eagerly fetches only _index.md files (small subset)
        to extract section titles. Other frameworks only need file path strings.

        Args:
            repo: PyGithub Repository object
            branch: Branch name
            source: Repository name
            version: Version string
            file_paths: All file paths to resolve

        Returns:
            Dict mapping file_path -> nav_path list
        """
        try:
            from datus.storage.document.nav_resolver import NavResolverPipeline

            # Pre-fetch _index.md files for potential Hugo frontmatter extraction.
            # This is cheap (typically <20 files) and harmless for non-Hugo frameworks
            # since the pipeline ignores fetched_docs when framework != Hugo.
            index_paths = [p for p in file_paths if p.endswith("_index.md")]
            index_docs = None
            if index_paths:
                index_docs = []
                for ip in index_paths:
                    doc = self._fetch_file(repo, ip, branch, version, source)
                    if doc:
                        index_docs.append(doc)
                logger.info(f"Pre-fetched {len(index_docs)} _index.md files for nav resolution")

            pipeline = NavResolverPipeline()
            nav_map = pipeline.resolve(
                repo=repo,
                branch=branch,
                file_paths=file_paths,
                rate_limiter=self.rate_limiter,
                fetched_docs=index_docs,
            )
            return nav_map

        except Exception as e:
            logger.warning(f"Nav path resolution failed: {e}")
            return {}

    def fetch_single(
        self,
        path: str,
        repo_name: Optional[str] = None,
        branch: Optional[str] = None,
        **kwargs,
    ) -> Optional[FetchedDocument]:
        """Fetch a single file from a repository.

        Args:
            path: File path within the repository
            repo_name: Repository in "owner/repo" format
            branch: Branch to fetch from
            **kwargs: Additional parameters

        Returns:
            Fetched document or None if not found
        """
        if not repo_name:
            raise ValueError("repo_name is required for fetch_single")

        repo = self._get_repo(repo_name)

        if self.github_ref:
            branch = branch or self.github_ref
            version = self.version or self.github_ref
        else:
            branch = branch or self._detect_default_branch(repo)
            version = self.version or self._detect_version(repo, branch)

        return self._fetch_file(repo, path, branch, version, repo_name)

    def _get_repo(self, repo_name: str) -> "Repository":
        """Get repository object.

        Args:
            repo_name: Repository in "owner/repo" format

        Returns:
            Repository object
        """
        self.rate_limiter.wait("api.github.com")
        return self._github.get_repo(repo_name)

    def _detect_default_branch(self, repo: "Repository") -> str:
        """Detect the default branch of a repository.

        Args:
            repo: Repository object

        Returns:
            Default branch name
        """
        try:
            return repo.default_branch
        except Exception:
            # Try common branch names
            for branch in ["main", "master"]:
                try:
                    repo.get_branch(branch)
                    return branch
                except GithubException:
                    continue
            return "main"

    def _detect_version(self, repo: "Repository", branch: str) -> str:
        """Detect version from releases, tags, or branch.

        Priority:
        1. Latest release tag
        2. Latest tag
        3. Branch name (if not main/master)
        4. Current date

        Args:
            repo: Repository object
            branch: Branch name

        Returns:
            Version string
        """
        # Try releases first
        try:
            self.rate_limiter.wait("api.github.com")
            releases = list(repo.get_releases()[:5])
            if releases:
                return releases[0].tag_name
        except GithubException:
            pass

        # Try tags
        try:
            self.rate_limiter.wait("api.github.com")
            tags = list(repo.get_tags()[:5])
            if tags:
                return tags[0].name
        except GithubException:
            pass

        # Use branch name if not default
        if branch not in ("main", "master"):
            return branch

        # Fall back to date
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _collect_file_paths(
        self,
        repo: "Repository",
        path: str,
        branch: str,
    ) -> List[str]:
        """Recursively collect file paths from a directory.

        Args:
            repo: Repository object
            path: Path to explore
            branch: Branch name

        Returns:
            List of file paths
        """
        self.rate_limiter.wait("api.github.com")
        contents = repo.get_contents(path, ref=branch)

        file_paths = []

        if not isinstance(contents, list):
            contents = [contents]

        for content in contents:
            if content.type == "dir":
                # Recurse into directory
                try:
                    sub_paths = self._collect_file_paths(repo, content.path, branch)
                    file_paths.extend(sub_paths)
                except GithubException as e:
                    logger.debug(f"Could not access directory {content.path}: {e}")
            elif content.type == "file" and self._is_doc_file(content.name):
                file_paths.append(content.path)

        return file_paths

    def _fetch_file(
        self,
        repo: "Repository",
        file_path: str,
        branch: str,
        version: str,
        repo_name: str,
    ) -> Optional[FetchedDocument]:
        """Fetch a single file's content.

        Args:
            repo: Repository object
            file_path: Path to the file
            branch: Branch name
            version: Version string
            repo_name: Repository name for URL construction

        Returns:
            FetchedDocument or None if failed
        """
        try:
            self.rate_limiter.wait("api.github.com")
            content = repo.get_contents(file_path, ref=branch)

            if isinstance(content, list):
                logger.warning(f"Expected file but got directory: {file_path}")
                return None

            # Decode content
            if content.encoding == "base64":
                raw_content = base64.b64decode(content.content).decode("utf-8")
            else:
                raw_content = content.content

            # Build source URL
            source_url = f"https://github.com/{repo_name}/blob/{branch}/{file_path}"

            # Detect content type
            content_type = self._detect_content_type(content.name, raw_content)

            # Build metadata
            metadata = {
                "sha": content.sha,
                "size": content.size,
                "branch": branch,
                "repo": repo_name,
            }

            return FetchedDocument(
                platform=self.platform,
                version=version,
                source_url=source_url,
                source_type=SOURCE_TYPE_GITHUB,
                doc_path=file_path,
                raw_content=raw_content,
                content_type=content_type,
                metadata=metadata,
            )

        except GithubException as e:
            if e.status == 404:
                logger.debug(f"File not found: {file_path}")
            else:
                logger.warning(f"GitHub API error for {file_path}: {e}")
            return None
        except UnicodeDecodeError as e:
            logger.warning(f"Could not decode {file_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error fetching {file_path}: {e}")
            return None

    def list_doc_files(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        paths: Optional[List[str]] = None,
    ) -> List[str]:
        """List documentation files in a repository without fetching content.

        Args:
            repo_name: Repository in "owner/repo" format
            branch: Branch name
            paths: Paths to search

        Returns:
            List of file paths
        """
        repo = self._get_repo(repo_name)
        branch = branch or self._detect_default_branch(repo)
        paths = paths or self.DEFAULT_PATHS

        file_paths = []
        for path in paths:
            try:
                found = self._collect_file_paths(repo, path, branch)
                file_paths.extend(found)
            except GithubException:
                continue

        return file_paths

    def get_repo_info(self, repo_name: str) -> Dict[str, Any]:
        """Get repository information.

        Args:
            repo_name: Repository in "owner/repo" format

        Returns:
            Dict with repo info (name, description, stars, etc.)
        """
        repo = self._get_repo(repo_name)

        return {
            "full_name": repo.full_name,
            "description": repo.description,
            "default_branch": repo.default_branch,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
            "language": repo.language,
            "topics": repo.get_topics() if hasattr(repo, "get_topics") else [],
        }
