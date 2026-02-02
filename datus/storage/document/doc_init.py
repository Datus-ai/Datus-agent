# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Document Initialization Module

Provides functions for importing and initializing documentation:
- import_documents: Import local documents into DocumentStore
- init_platform_docs: Full pipeline for platform documentation
- search_platform_docs: Search indexed documentation
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from datus.storage.embedding_models import EmbeddingModel
from datus.utils.loggings import get_logger

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
                "platform": doc.platform,
                "version": doc.version,
                "source_type": doc.source_type,
                "source_url": doc.source_url,
                "doc_path": doc.doc_path,
            }

            return chunker.chunk(parsed, base_metadata)

        except Exception as e:
            logger.warning(f"Failed to process {doc.doc_path}: {e}")
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
                errors.append(f"Error ({doc.doc_path}): {str(e)}")

    return batch_chunks


# =============================================================================
# Platform Documentation Functions
# =============================================================================


def init_platform_docs(
    db_path: str,
    platform: str,
    source: str,
    source_type: str = SOURCE_TYPE_GITHUB,
    version: Optional[str] = None,
    paths: Optional[List[str]] = None,
    build_mode: str = "incremental",
    chunk_size: int = 1024,
    pool_size: int = 4,
    github_token: Optional[str] = None,
    github_ref: Optional[str] = None,
    max_depth: int = 2,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> InitResult:
    """Initialize platform documentation knowledge base.

    Fetches documentation from the specified source, parses it, splits into
    chunks, and stores in the vector database. Uses micro-batch processing
    to limit peak memory usage.

    Args:
        db_path: Path to LanceDB database
        platform: Target platform (snowflake, duckdb, postgresql, etc.)
        source: Source location (GitHub repo "owner/repo", website URL, or local path)
        source_type: Source type ("github", "website", or "local")
        version: Specific version (auto-detected if not provided)
        paths: Paths to fetch for GitHub (default: docs, README.md)
        build_mode: Build mode ("check", "overwrite", "incremental")
        chunk_size: Target chunk size in characters
        pool_size: Thread pool size for parallel processing
        github_token: GitHub API token (falls back to GITHUB_TOKEN env var)
        github_ref: Explicit git ref (branch or tag) to fetch from
        max_depth: Maximum crawl depth for websites
        include_patterns: URL/file patterns to include
        exclude_patterns: URL/file patterns to exclude
        batch_size: Number of documents to process per micro-batch

    Returns:
        InitResult with statistics and status
    """
    start_time = datetime.now(timezone.utc)
    errors = []

    logger.info(f"Initializing {platform} documentation from {source} ({source_type})")

    store = document_store(db_path)

    # Check existing data for build mode
    if build_mode == "check":
        stats = store.get_platform_stats(platform)
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        return InitResult(
            platform=platform,
            version=stats.get("versions", ["unknown"])[0] if stats["versions"] else "unknown",
            source=source,
            total_docs=stats.get("doc_count", 0),
            total_chunks=stats.get("total_chunks", 0),
            success=True,
            errors=[],
            duration_seconds=duration,
        )

    if build_mode == "overwrite":
        deleted = store.delete_by_platform(platform)
        logger.info(f"Deleted {deleted} existing chunks for {platform}")

    # Initialize pipeline components
    rate_limiter = RateLimiter()
    cleaner = DocumentCleaner()
    markdown_parser = MarkdownParser()
    html_parser = HTMLParser()
    chunker = SemanticChunker(config=ChunkingConfig(chunk_size=chunk_size))

    total_docs = 0
    total_chunks = 0

    try:
        if source_type == SOURCE_TYPE_GITHUB:
            # GitHub: two-phase micro-batch processing
            # Phase 1: Collect file paths + resolve nav_map (lightweight)
            fetcher = GitHubFetcher(
                platform=platform,
                version=version,
                github_ref=github_ref,
                token=github_token,
                rate_limiter=rate_limiter,
                pool_size=pool_size,
            )
            metadata = fetcher.collect_metadata(source=source, paths=paths)

            if not metadata.file_paths:
                logger.warning(f"No documentation files found in {source}")
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                return InitResult(
                    platform=platform,
                    version=version or "unknown",
                    source=source,
                    total_docs=0,
                    total_chunks=0,
                    success=True,
                    errors=["No documents found"],
                    duration_seconds=duration,
                )

            if not version:
                version = metadata.version

            logger.info(
                f"GitHub Phase 1 complete: {len(metadata.file_paths)} files, " f"processing in batches of {batch_size}"
            )

            # Phase 2: Batch fetch + process + store
            for i in range(0, len(metadata.file_paths), batch_size):
                batch_paths = metadata.file_paths[i : i + batch_size]
                batch_num = i // batch_size + 1

                batch_docs = fetcher.fetch_batch(metadata, batch_paths)
                total_docs += len(batch_docs)

                batch_chunks = _process_batch(
                    batch_docs,
                    cleaner,
                    markdown_parser,
                    html_parser,
                    chunker,
                    pool_size,
                    errors,
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
            # Local/Web: fetch all, then process+store in micro-batches
            if source_type == SOURCE_TYPE_LOCAL:
                fetcher = LocalFetcher(platform=platform, version=version)
                documents = fetcher.fetch(
                    source=source,
                    recursive=True,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                )
            else:
                fetcher = WebFetcher(
                    platform=platform,
                    version=version,
                    rate_limiter=rate_limiter,
                    pool_size=pool_size,
                )
                documents = fetcher.fetch(
                    source=source,
                    max_depth=max_depth,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                )

            if not documents:
                logger.warning(f"No documents fetched from {source}")
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                return InitResult(
                    platform=platform,
                    version=version or "unknown",
                    source=source,
                    total_docs=0,
                    total_chunks=0,
                    success=True,
                    errors=["No documents found"],
                    duration_seconds=duration,
                )

            if not version:
                version = documents[0].version

            logger.info(f"Fetched {len(documents)} documents, " f"processing in batches of {batch_size}")

            # Process + store in micro-batches
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                batch_num = i // batch_size + 1
                total_docs += len(batch_docs)

                batch_chunks = _process_batch(
                    batch_docs,
                    cleaner,
                    markdown_parser,
                    html_parser,
                    chunker,
                    pool_size,
                    errors,
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

    logger.info(f"Platform documentation initialized: " f"{total_docs} docs, {total_chunks} chunks, {duration:.1f}s")

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


def search_platform_docs(
    db_path: str,
    embedding_model: EmbeddingModel,
    query: str,
    platform: Optional[str] = None,
    version: Optional[str] = None,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Search platform documentation.

    Args:
        db_path: Path to LanceDB database
        embedding_model: Embedding model for vectorization
        query: Search query
        platform: Filter by platform
        version: Filter by version
        top_n: Maximum results to return

    Returns:
        List of matching chunks as dictionaries
    """
    from datus.storage.document.store import DocumentStore

    store = DocumentStore(db_path=db_path, embedding_model=embedding_model)
    return store.search_docs(
        query=query,
        platform=platform,
        version=version,
        top_n=top_n,
    )


def list_platforms(
    db_path: str,
    embedding_model: EmbeddingModel,
) -> List[Dict[str, Any]]:
    """List all indexed platforms.

    Args:
        db_path: Path to LanceDB database
        embedding_model: Embedding model for vectorization

    Returns:
        List of platform info dictionaries
    """
    from datus.storage.document.store import DocumentStore

    store = DocumentStore(db_path=db_path, embedding_model=embedding_model)
    return store.list_platforms()


def delete_platform_docs(
    db_path: str,
    embedding_model: EmbeddingModel,
    platform: str,
    version: Optional[str] = None,
) -> int:
    """Delete platform documentation.

    Args:
        db_path: Path to LanceDB database
        embedding_model: Embedding model for vectorization
        platform: Platform to delete
        version: Specific version to delete (all if not specified)

    Returns:
        Number of chunks deleted
    """
    from datus.storage.document.store import DocumentStore

    store = DocumentStore(db_path=db_path, embedding_model=embedding_model)
    return store.delete_by_platform(platform, version)


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
