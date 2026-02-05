# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Document Initialization Module

Provides functions for importing and initializing documentation:
- import_documents: Import local documents into DocumentStore
- init_platform_docs: Full pipeline for platform documentation
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

from datus.storage.document.chunker import SemanticChunker
from datus.storage.document.chunker.semantic_chunker import ChunkingConfig
from datus.storage.document.cleaner import DocumentCleaner
from datus.storage.document.fetcher import GitHubFetcher, LocalFetcher, RateLimiter, WebFetcher
from datus.storage.document.parser import HTMLParser, MarkdownParser
from datus.storage.document.schemas import (
    CONTENT_TYPE_MARKDOWN,
    SOURCE_TYPE_GITHUB,
    SOURCE_TYPE_LOCAL,
    FetchedDocument,
    PlatformDocChunk,
)
from datus.storage.document.store import document_store
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import DocumentConfig

logger = get_logger(__name__)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class InitResult:
    """Result of platform documentation initialization.

    Attributes:
        platform: Platform name
        version: Documentation version
        source: Source location
        total_docs: Number of documents processed
        total_chunks: Number of chunks created
        success: Whether initialization succeeded
        errors: List of error messages
        duration_seconds: Time taken in seconds
    """

    platform: str
    version: str
    source: str
    total_docs: int
    total_chunks: int
    success: bool
    errors: List[str]
    duration_seconds: float


# =============================================================================
# Batch Processing Helpers
# =============================================================================

DEFAULT_BATCH_SIZE = 50


def _process_batch(
    documents: List[FetchedDocument],
    cleaner: "DocumentCleaner",
    markdown_parser: "MarkdownParser",
    html_parser: "HTMLParser",
    chunker: "SemanticChunker",
    pool_size: int,
    errors: List[str],
) -> List[PlatformDocChunk]:
    """Process a batch of documents into chunks using a thread pool.

    Args:
        documents: Batch of fetched documents to process
        cleaner: Document cleaner instance
        markdown_parser: Markdown parser instance
        html_parser: HTML parser instance
        chunker: Semantic chunker instance
        pool_size: Thread pool size for parallel processing
        errors: Shared error list (appended to on failure)

    Returns:
        List of chunks produced from the batch
    """
    if not documents:
        return []

    errors_lock = threading.Lock()

    def process_one(doc: FetchedDocument) -> List[PlatformDocChunk]:
        try:
            cleaned_doc = cleaner.clean(doc)

            if cleaned_doc.content_type == CONTENT_TYPE_MARKDOWN:
                parsed = markdown_parser.parse(cleaned_doc)
            else:
                parsed = html_parser.parse(cleaned_doc)

            # Merge nav_path from fetcher metadata (set by NavResolverPipeline)
            if doc.metadata.get("nav_path"):
                parsed.metadata["nav_path"] = doc.metadata["nav_path"]
            if doc.metadata.get("group_name"):
                parsed.metadata["group_name"] = doc.metadata["group_name"]

            base_metadata = {
                "platform": doc.platform,  # kept for keyword extraction only
                "version": doc.version,
                "source_type": doc.source_type,
                "source_url": doc.source_url,
                "doc_path": doc.doc_path,
                "content_hash": doc.metadata.get("content_hash", ""),
            }

            return chunker.chunk(parsed, base_metadata)

        except Exception as e:
            logger.warning(f"Failed to process {doc.doc_path}: {e}")
            with errors_lock:
                errors.append(f"Process error ({doc.doc_path}): {str(e)}")
            return []

    batch_chunks: List[PlatformDocChunk] = []

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = {executor.submit(process_one, doc): doc for doc in documents}

        for future in as_completed(futures):
            doc = futures[future]
            try:
                chunks = future.result()
                batch_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Processing failed for {doc.doc_path}: {e}")
                with errors_lock:
                    errors.append(f"Error ({doc.doc_path}): {str(e)}")

    return batch_chunks


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
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> InitResult:
    """Initialize platform documentation knowledge base.

    Pipeline:
      1. check mode: return existing store stats without any fetching
      2. overwrite mode:
         a. Resolve version (from source metadata or user input)
         b. Delete existing data for the resolved version
         c. Fetch → Parse → Clean → Chunk → Store (micro-batch)
         d. Create indices

    Args:
        db_path: Path to LanceDB database
        platform: Target platform (snowflake, duckdb, postgresql, etc.)
        cfg: DocumentConfig with source, version, paths, chunk_size,
            include_patterns, exclude_patterns, etc.
        build_mode: Build mode ("check" or "overwrite")
        pool_size: Thread pool size for parallel processing
        batch_size: Number of documents to process per micro-batch

    Returns:
        InitResult with statistics and status
    """
    source = cfg.source or ""
    source_type = cfg.type
    version = cfg.version

    start_time = datetime.now(timezone.utc)
    errors: List[str] = []

    logger.info(f"Initializing {platform} documentation from {source} ({source_type})")

    store = document_store(db_path)

    # ==================================================================
    # Check mode: return existing store stats without any I/O or fetching
    # ==================================================================
    if build_mode == "check":
        stats = store.get_stats()
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        return InitResult(
            platform=platform,
            version=stats.get("versions", ["unknown"])[-1] if stats.get("versions") else "unknown",
            source=source,
            total_docs=stats.get("doc_count", 0),
            total_chunks=stats.get("total_chunks", 0),
            success=True,
            errors=[],
            duration_seconds=duration,
        )

    # Initialize pipeline components
    rate_limiter = RateLimiter()
    cleaner = DocumentCleaner()
    markdown_parser = MarkdownParser()
    html_parser = HTMLParser()
    chunker = SemanticChunker(config=ChunkingConfig(chunk_size=cfg.chunk_size))

    total_docs = 0
    total_chunks = 0

    try:
        # ==================================================================
        # Phase 1: Resolve version + fetch documents
        # ==================================================================
        github_metadata = None  # only for GitHub source type

        if source_type == SOURCE_TYPE_GITHUB:
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

            if not version:
                version = github_metadata.version

            logger.info(
                f"Phase 1 complete: version='{version}', "
                f"{len(github_metadata.file_paths)} files, batch_size={batch_size}"
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

        else:
            # Website
            fetcher = WebFetcher(
                platform=platform,
                version=version,
                rate_limiter=rate_limiter,
                pool_size=pool_size,
            )
            documents = fetcher.fetch(
                source=source,
                max_depth=cfg.max_depth,
                include_patterns=cfg.include_patterns,
                exclude_patterns=cfg.exclude_patterns,
            )
            if not documents:
                logger.warning(f"No documents fetched from {source}")
                return _make_empty_result(platform, version, source, start_time, ["No documents found"])
            if not version:
                version = documents[0].version

        # ==================================================================
        # Phase 2: Overwrite — delete existing data for the resolved version
        # ==================================================================
        deleted = store.delete_docs(version=version)
        if deleted:
            logger.info(f"Overwrite: deleted {deleted} existing chunks for version '{version}'")

        # ==================================================================
        # Phase 3: Process + Store (micro-batch)
        # ==================================================================
        if source_type == SOURCE_TYPE_GITHUB:
            for i in range(0, len(github_metadata.file_paths), batch_size):
                batch_paths = github_metadata.file_paths[i : i + batch_size]
                batch_num = i // batch_size + 1

                batch_docs = fetcher.fetch_batch(github_metadata, batch_paths)
                total_docs += len(batch_docs)

                batch_chunks = _process_batch(
                    batch_docs, cleaner, markdown_parser, html_parser, chunker, pool_size, errors
                )
                total_chunks += len(batch_chunks)

                if batch_chunks:
                    try:
                        store.store_chunks(batch_chunks)
                    except Exception as e:
                        logger.error(f"Failed to store batch {batch_num}: {e}")
                        errors.append(f"Storage error (batch {batch_num}): {str(e)}")

                logger.info(f"Batch {batch_num}: {len(batch_docs)} docs -> {len(batch_chunks)} chunks")

        else:
            # Local/Web: documents already fetched in Phase 1
            logger.info(f"Processing {len(documents)} documents in batches of {batch_size}")

            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                batch_num = i // batch_size + 1
                total_docs += len(batch_docs)

                batch_chunks = _process_batch(
                    batch_docs, cleaner, markdown_parser, html_parser, chunker, pool_size, errors
                )
                total_chunks += len(batch_chunks)

                if batch_chunks:
                    try:
                        store.store_chunks(batch_chunks)
                    except Exception as e:
                        logger.error(f"Failed to store batch {batch_num}: {e}")
                        errors.append(f"Storage error (batch {batch_num}): {str(e)}")

                logger.info(f"Batch {batch_num}: {len(batch_docs)} docs -> {len(batch_chunks)} chunks")

    except Exception as e:
        logger.error(f"Failed to process documents: {e}")
        errors.append(f"Fetch error: {str(e)}")
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        return InitResult(
            platform=platform,
            version=version or "unknown",
            source=source,
            total_docs=total_docs,
            total_chunks=total_chunks,
            success=False,
            errors=errors,
            duration_seconds=duration,
        )

    # Create indices once after all batches are stored
    if total_chunks > 0:
        try:
            store.create_indices()
        except Exception as e:
            logger.error(f"Failed to create indices: {e}")
            errors.append(f"Index error: {str(e)}")

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()

    logger.info(f"Platform documentation initialized: {total_docs} docs, {total_chunks} chunks, {duration:.1f}s")

    return InitResult(
        platform=platform,
        version=version,
        source=source,
        total_docs=total_docs,
        total_chunks=total_chunks,
        success=len(errors) == 0 or total_chunks > 0,
        errors=errors,
        duration_seconds=duration,
    )


# =============================================================================
# Simple Document Import Functions
# =============================================================================


def import_documents(
    store,  # DocumentStore
    directory_path: str,
    recursive: bool = False,
    chunk_size: int = 1024,
    batch_size: int = DEFAULT_BATCH_SIZE,
    platform: str = "local",
    version: str = "local",
) -> Tuple[int, List[str]]:
    """Import documents from a directory into the document store.

    Uses the full pipeline with micro-batch processing:
    - Fetches documents from local directory
    - Processes and stores in batches to limit memory usage

    Args:
        store: DocumentStore instance
        directory_path: Path to the directory containing documents
        recursive: Whether to scan subdirectories recursively
        chunk_size: Target chunk size in characters
        batch_size: Number of documents to process per micro-batch
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

        # Initialize pipeline components
        fetcher = LocalFetcher(platform=platform, version=version)
        cleaner = DocumentCleaner()
        markdown_parser = MarkdownParser()
        html_parser = HTMLParser()
        chunker = SemanticChunker(config=ChunkingConfig(chunk_size=chunk_size))

        # Fetch documents
        documents = fetcher.fetch(
            source=directory_path,
            recursive=recursive,
        )

        if not documents:
            logger.warning(f"No documents found in {directory_path}")
            return 0, []

        logger.info(f"Found {len(documents)} documents in {directory_path}")

        total_chunks = 0
        imported_titles = []
        errors = []

        # Process and store in micro-batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]

            # Process batch into chunks
            batch_chunks = _process_batch(
                batch_docs,
                cleaner,
                markdown_parser,
                html_parser,
                chunker,
                1,
                errors,  # pool_size=1 for local imports
            )
            total_chunks += len(batch_chunks)

            # Extract titles from batch docs
            for doc in batch_docs:
                title = doc.metadata.get("title", doc.doc_path)
                imported_titles.append(title)

            if batch_chunks:
                store.store_chunks(batch_chunks)

        # Create indices once after all batches
        if total_chunks > 0:
            store.create_indices()

        logger.info(f"Imported {total_chunks} chunks from {len(documents)} documents")
        return total_chunks, imported_titles

    except Exception as e:
        logger.error(f"Document import failed: {str(e)}")
        return 0, []
