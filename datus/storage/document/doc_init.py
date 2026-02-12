# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Document Initialization Module

Provides functions for importing and initializing documentation:
- import_documents: Import local documents into DocumentStore
- init_platform_docs: Full pipeline for platform documentation

Uses streaming processing where each document is fully processed
(fetch → chunk → store) in a single thread for maximum efficiency.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from datus.storage.document.fetcher import GitHubFetcher, LocalFetcher, RateLimiter, WebFetcher
from datus.storage.document.schemas import SOURCE_TYPE_GITHUB, SOURCE_TYPE_LOCAL
from datus.storage.document.store import document_store
from datus.storage.document.streaming_processor import StreamingDocProcessor
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import DocumentConfig

logger = get_logger(__name__)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class VersionStats:
    """Per-version statistics for platform documentation."""

    version: str
    doc_count: int
    chunk_count: int


@dataclass
class InitResult:
    """Result of platform documentation initialization.

    Attributes:
        platform: Platform name
        version: Documentation version (comma-separated if multiple)
        source: Source location
        total_docs: Number of documents processed
        total_chunks: Number of chunks created
        success: Whether initialization succeeded
        errors: List of error messages
        duration_seconds: Time taken in seconds
        version_details: Per-version breakdown (populated in check mode)
    """

    platform: str
    version: str
    source: str
    total_docs: int
    total_chunks: int
    success: bool
    errors: List[str]
    duration_seconds: float
    version_details: Optional[List[VersionStats]] = None


# =============================================================================
# Version Detection Helpers
# =============================================================================

# Pattern to detect version strings like "1.3.0", "v2.0.0", "1.2.3-beta"
_VERSION_PATH_RE = re.compile(r"^v?(\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?)$")


def _detect_versions_from_paths(paths: List[str]) -> Set[str]:
    """Detect version strings from path list.

    Used to identify when GitHub paths represent version directories
    (e.g., ["1.3.0", "1.2.0"]) rather than regular paths (e.g., ["docs", "README.md"]).

    Args:
        paths: List of path strings to check

    Returns:
        Set of detected version strings (empty if paths aren't version directories)
    """
    if not paths:
        return set()

    versions = set()
    for path in paths:
        # Get the first path component (e.g., "1.3.0" from "1.3.0/docs/intro.md")
        first_segment = path.split("/")[0]
        match = _VERSION_PATH_RE.match(first_segment)
        if match:
            versions.add(match.group(1))

    # Only return versions if ALL paths start with a version pattern
    # (to avoid false positives from paths like "v1-api/docs")
    version_count = 0
    for path in paths:
        first_segment = path.split("/")[0]
        if _VERSION_PATH_RE.match(first_segment):
            version_count += 1

    if version_count == len(paths):
        return versions

    return set()


def _detect_versions_from_file_paths(file_paths: List[str]) -> Set[str]:
    """Detect version strings from the first path segment of file paths.

    Used after auto-discovery to detect version directories from actual file paths
    like ["1.2.0/docs/intro.md", "1.3.0/guides/setup.md"].

    Args:
        file_paths: List of file path strings (e.g., from GitHub metadata)

    Returns:
        Set of detected version strings, or empty set if no consistent pattern
    """
    if not file_paths:
        return set()

    versions = set()
    for fp in file_paths:
        first_segment = fp.split("/")[0]
        match = _VERSION_PATH_RE.match(first_segment)
        if match:
            versions.add(match.group(1))

    # Only return if a meaningful proportion of files are under version dirs
    # (avoids false positives from a single versioned path in a mixed repo)
    if not versions:
        return set()

    versioned_count = sum(1 for fp in file_paths if _VERSION_PATH_RE.match(fp.split("/")[0]))
    if versioned_count >= len(file_paths) * 0.5:
        return versions

    return set()


def _build_version_details(store, all_versions: List[str], target_versions: Set[str]) -> List[VersionStats]:
    """Build per-version statistics, optionally filtered to target versions.

    Args:
        store: DocumentStore instance
        all_versions: All versions found in the store
        target_versions: If non-empty, only include these versions

    Returns:
        List of VersionStats sorted by version string
    """
    versions_to_check = sorted(target_versions) if target_versions else sorted(all_versions)
    details = []
    for ver in versions_to_check:
        ver_stats = store.get_stats_by_version(ver)
        details.append(
            VersionStats(
                version=ver,
                doc_count=ver_stats.get("doc_count", 0),
                chunk_count=ver_stats.get("total_chunks", 0),
            )
        )
    return details


# =============================================================================
# Platform Documentation Functions
# =============================================================================


def _make_empty_result(platform: str, version: str, source: str, start_time, errors=None) -> InitResult:
    """Create an InitResult for empty/no-op cases."""
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    return InitResult(
        platform=platform,
        version=version or "unknown",
        source=source,
        total_docs=0,
        total_chunks=0,
        success=True,
        errors=errors or [],
        duration_seconds=duration,
    )


def init_platform_docs(
    db_path: str,
    platform: str,
    cfg: "DocumentConfig",
    build_mode: str = "overwrite",
    pool_size: int = 4,
) -> InitResult:
    """Initialize platform documentation knowledge base.

    Uses streaming processing where each document is fully processed
    (fetch → chunk → store) in a single thread for maximum efficiency.

    Pipeline:
      1. check mode: return existing store stats without any fetching
      2. overwrite mode:
         a. Resolve version (from source metadata or user input)
         b. Delete existing data for the resolved version
         c. Streaming process: fetch → chunk → store per document
         d. Create indices

    Args:
        db_path: Path to LanceDB database
        platform: Target platform (snowflake, duckdb, postgresql, etc.)
        cfg: DocumentConfig with source, version, paths, chunk_size,
            include_patterns, exclude_patterns, etc.
        build_mode: Build mode ("check" or "overwrite")
        pool_size: Thread pool size for parallel processing

    Returns:
        InitResult with statistics and status
    """
    source = cfg.source or ""
    source_type = cfg.type
    version = cfg.version

    start_time = datetime.now(timezone.utc)

    logger.info(f"Initializing {platform} documentation from {source} ({source_type})")

    store = document_store(db_path)

    # ==================================================================
    # Check mode: return existing store stats without any I/O or fetching
    # ==================================================================
    if build_mode == "check":
        stats = store.get_stats()
        all_versions = stats.get("versions", [])
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Resolve target versions: explicit --version, or detected from paths
        target_versions: Set[str] = set()
        if version:
            target_versions.add(version)
        else:
            target_versions = _detect_versions_from_paths(cfg.paths or [])

        # Build per-version breakdown from store data
        version_details = _build_version_details(store, all_versions, target_versions)

        # Compute totals from the (possibly filtered) version details
        shown_versions = [vd.version for vd in version_details]
        total_docs = sum(vd.doc_count for vd in version_details)
        total_chunks = sum(vd.chunk_count for vd in version_details)
        version_str = ", ".join(shown_versions) if shown_versions else "unknown"

        return InitResult(
            platform=platform,
            version=version_str,
            source=source,
            total_docs=total_docs,
            total_chunks=total_chunks,
            success=True,
            errors=[],
            duration_seconds=duration,
            version_details=version_details,
        )

    # Initialize components
    rate_limiter = RateLimiter()
    processor = StreamingDocProcessor(
        store=store,
        chunk_size=cfg.chunk_size,
        pool_size=pool_size,
    )

    # Track versions from path-based directories (e.g., paths=["1.3.0", "1.2.0"])
    path_versions: Set[str] = set()

    try:
        # ==================================================================
        # Phase 1: Resolve version + prepare for processing
        # ==================================================================
        if source_type == SOURCE_TYPE_GITHUB:
            # Detect if paths represent version directories
            path_versions = _detect_versions_from_paths(cfg.paths or [])
            if path_versions:
                logger.info(f"Detected versioned paths: {sorted(path_versions)}")

            fetcher = GitHubFetcher(
                platform=platform,
                version=version,
                github_ref=cfg.github_ref,
                token=cfg.github_token,
                rate_limiter=rate_limiter,
                pool_size=pool_size,
            )
            github_metadata = fetcher.collect_metadata(source=source, paths=cfg.paths)

            if not github_metadata.file_paths:
                logger.warning(f"No documentation files found in {source}")
                return _make_empty_result(platform, version, source, start_time, ["No documents found"])

            # Re-detect path_versions from actual file paths — this covers
            # auto-discovered version directories when --paths was not specified
            if not path_versions:
                path_versions = _detect_versions_from_file_paths(github_metadata.file_paths)
                if path_versions:
                    logger.info(f"Detected versioned dirs from files: {sorted(path_versions)}")

            if not version:
                if path_versions:
                    # Use first detected path version as global fallback;
                    # per-doc version is overridden from doc_path anyway
                    version = sorted(path_versions)[0]
                else:
                    version = github_metadata.version

            logger.info(f"Found {len(github_metadata.file_paths)} files, version='{version}'")

            # Phase 2: Delete existing data
            _delete_existing_versions(store, version, path_versions)

            # Phase 3: Streaming process
            stats = processor.process_github(
                fetcher=fetcher,
                metadata=github_metadata,
                version=version,
                platform=platform,
            )

        elif source_type == SOURCE_TYPE_LOCAL:
            fetcher = LocalFetcher(platform=platform, version=version)
            documents = fetcher.fetch(
                source=source,
                recursive=True,
                include_patterns=cfg.include_patterns,
                exclude_patterns=cfg.exclude_patterns,
            )

            if not documents:
                logger.warning(f"No documents fetched from {source}")
                return _make_empty_result(platform, version, source, start_time, ["No documents found"])

            if not version:
                version = documents[0].version

            logger.info(f"Found {len(documents)} documents, version='{version}'")

            # Phase 2: Delete existing data
            _delete_existing_versions(store, version, path_versions)

            # Phase 3: Streaming process
            stats = processor.process_local(
                fetcher=fetcher,
                documents=documents,
                version=version,
                platform=platform,
            )

        else:
            # Website - true streaming with URL discovery
            fetcher = WebFetcher(
                platform=platform,
                version=version,
                rate_limiter=rate_limiter,
                pool_size=pool_size,
            )

            # Detect version from URL
            if not version:
                version = fetcher._detect_version_from_url(source)

            logger.info(f"Starting website crawl from {source}, version='{version}'")

            # Phase 2: Delete existing data
            _delete_existing_versions(store, version, path_versions)

            # Phase 3: Streaming process with URL discovery
            stats = processor.process_website(
                fetcher=fetcher,
                base_url=source,
                version=version,
                platform=platform,
                max_depth=cfg.max_depth,
                include_patterns=cfg.include_patterns,
                exclude_patterns=cfg.exclude_patterns,
            )

    except Exception as e:
        logger.error(f"Failed to process documents: {e}")
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        error_version = ", ".join(sorted(path_versions)) if path_versions else (version or "unknown")
        return InitResult(
            platform=platform,
            version=error_version,
            source=source,
            total_docs=0,
            total_chunks=0,
            success=False,
            errors=[f"Processing error: {str(e)}"],
            duration_seconds=duration,
        )

    # Create indices once after all processing is complete
    if stats.total_chunks > 0:
        try:
            store.create_indices()
        except Exception as e:
            logger.error(f"Failed to create indices: {e}")
            stats.add_error(f"Index error: {str(e)}")

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()

    # For multi-version mode, report versions as comma-separated string
    result_version = version
    if path_versions:
        result_version = ", ".join(sorted(path_versions))

    logger.info(
        f"Platform documentation initialized: {stats.total_docs} docs, {stats.total_chunks} chunks, {duration:.1f}s"
    )

    return InitResult(
        platform=platform,
        version=result_version,
        source=source,
        total_docs=stats.total_docs,
        total_chunks=stats.total_chunks,
        success=len(stats.errors) == 0 or stats.total_chunks > 0,
        errors=stats.errors,
        duration_seconds=duration,
    )


def _delete_existing_versions(store, version: str, path_versions: Set[str]) -> None:
    """Delete existing data for the resolved version(s).

    Args:
        store: DocumentStore instance
        version: Single version string
        path_versions: Set of versions from path-based directories
    """
    if path_versions:
        # Multi-version mode: delete each version from path directories
        total_deleted = 0
        for ver in sorted(path_versions):
            deleted = store.delete_docs(version=ver)
            if deleted:
                logger.info(f"Overwrite: deleted {deleted} existing chunks for version '{ver}'")
                total_deleted += deleted
        if total_deleted:
            logger.info(f"Overwrite: deleted {total_deleted} total chunks across {len(path_versions)} versions")
    else:
        # Single-version mode: delete for the resolved version
        deleted = store.delete_docs(version=version)
        if deleted:
            logger.info(f"Overwrite: deleted {deleted} existing chunks for version '{version}'")


# =============================================================================
# Platform Inference
# =============================================================================


def infer_platform_from_source(source: str) -> Optional[str]:
    """Infer platform name from a document source string.

    Handles three source formats:
      - GitHub repo:  "owner/repo" or "https://github.com/owner/repo/..."
      - Website URL:  "https://docs.snowflake.com/..."
      - Local path:   "/path/to/starrocks-docs"

    Returns:
        Lowercase platform name, or None if unable to infer.
    """
    from urllib.parse import urlparse

    source = source.strip().rstrip("/")
    if not source:
        return None

    # --- GitHub URL: https://github.com/owner/repo/... ---
    gh_url_match = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/|$)", source)
    if gh_url_match:
        repo_name = gh_url_match.group(2).lower()
        # Strip common suffixes: "-docs", "-documentation", etc.
        repo_name = re.sub(r"[_-]?(docs?|documentation|website)$", "", repo_name)
        return repo_name or None

    # --- GitHub shorthand: "owner/repo" (no scheme, exactly one slash) ---
    if "/" in source and not source.startswith(("http://", "https://", "/")):
        parts = source.split("/")
        if len(parts) == 2 and parts[0] and parts[1]:
            repo_name = parts[1].lower()
            repo_name = re.sub(r"[_-]?(docs?|documentation|website)$", "", repo_name)
            return repo_name or None

    # --- Website URL: extract from domain ---
    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        domain = parsed.netloc.lower()
        # Remove port, "www.", and TLD suffixes
        domain = re.sub(r":\d+$", "", domain)
        domain = re.sub(r"^www\.", "", domain)
        # "docs.snowflake.com" -> "snowflake"
        # "snowflake.com" -> "snowflake"
        domain_parts = domain.split(".")
        if len(domain_parts) >= 2:
            # Pick the second-level domain (e.g., "snowflake" from "docs.snowflake.com")
            return domain_parts[-2] or None
        return None

    # --- Local path: use the last directory component ---
    name = Path(source).name.lower()
    if name:
        # Strip common suffixes
        name = re.sub(r"[_-]?(docs?|documentation|website)$", "", name)
        return name or None

    return None


# =============================================================================
# Simple Document Import Functions
# =============================================================================


def import_documents(
    store,  # DocumentStore
    directory_path: str,
    recursive: bool = False,
    chunk_size: int = 1024,
    pool_size: int = 4,
    platform: str = "local",
    version: str = "local",
) -> Tuple[int, List[str]]:
    """Import documents from a directory into the document store.

    Uses streaming processing where each document is fully processed
    (fetch → chunk → store) in a single thread.

    Args:
        store: DocumentStore instance
        directory_path: Path to the directory containing documents
        recursive: Whether to scan subdirectories recursively
        chunk_size: Target chunk size in characters
        pool_size: Number of worker threads
        platform: Platform name to tag imported documents with
        version: Version string to tag imported documents with

    Returns:
        Tuple containing (number of chunks imported, list of document titles)
    """
    try:
        document_path = Path(directory_path)
        if not document_path.exists() or not document_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return 0, []

        # Fetch documents
        fetcher = LocalFetcher(platform=platform, version=version)
        documents = fetcher.fetch(
            source=directory_path,
            recursive=recursive,
        )

        if not documents:
            logger.warning(f"No documents found in {directory_path}")
            return 0, []

        logger.info(f"Found {len(documents)} documents in {directory_path}")

        # Extract titles before processing
        imported_titles = []
        for doc in documents:
            doc_metadata = doc.metadata or {}
            title = doc_metadata.get("title", doc.doc_path)
            imported_titles.append(title)

        # Streaming process
        processor = StreamingDocProcessor(
            store=store,
            chunk_size=chunk_size,
            pool_size=pool_size,
        )

        stats = processor.process_local(
            fetcher=fetcher,
            documents=documents,
            version=version,
            platform=platform,
        )

        # Create indices once after all processing
        if stats.total_chunks > 0:
            store.create_indices()

        logger.info(f"Imported {stats.total_chunks} chunks from {len(documents)} documents")
        return stats.total_chunks, imported_titles

    except Exception as e:
        logger.error(f"Document import failed: {str(e)}")
        return 0, []
