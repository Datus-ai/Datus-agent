# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Streaming Document Processor

Provides unified streaming processing for documents from any source (Website, GitHub, Local).
Each document is fully processed (fetch → chunk → store) in a single thread,
with new URLs/files discovered during processing added to the work queue.
"""

import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional, Set
from urllib.parse import urlparse

from datus.storage.document.chunker import SemanticChunker
from datus.storage.document.chunker.semantic_chunker import ChunkingConfig
from datus.storage.document.cleaner import DocumentCleaner
from datus.storage.document.parser import HTMLParser, MarkdownParser
from datus.storage.document.schemas import (
    CONTENT_TYPE_HTML,
    CONTENT_TYPE_MARKDOWN,
    CONTENT_TYPE_RST,
    FetchedDocument,
    PlatformDocChunk,
)
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.storage.document.store import DocumentStore

logger = get_logger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for streaming processing."""

    total_docs: int = 0
    total_chunks: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Thread-safe locks
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def increment(self, docs: int = 0, chunks: int = 0):
        """Thread-safe increment of counters."""
        with self._lock:
            self.total_docs += docs
            self.total_chunks += chunks

    def add_error(self, error: str):
        """Thread-safe error logging."""
        with self._lock:
            self.errors.append(error)

    @property
    def duration_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()


class StreamingDocProcessor:
    """Streaming document processor using producer-consumer pattern.

    Each document is fully processed (fetch → clean → parse → chunk → store)
    in a single thread. For website sources, new URLs discovered during
    processing are added to the work queue.

    Example:
        >>> processor = StreamingDocProcessor(store, chunk_size=1024)
        >>> stats = processor.process_website(
        ...     fetcher=web_fetcher,
        ...     base_url="https://docs.example.com/",
        ...     max_depth=2,
        ... )
        >>> print(f"Processed {stats.total_docs} docs, {stats.total_chunks} chunks")
    """

    def __init__(
        self,
        store: "DocumentStore",
        chunk_size: int = 1024,
        pool_size: int = 4,
    ):
        """Initialize the streaming processor.

        Args:
            store: DocumentStore for storing chunks
            chunk_size: Target chunk size in characters
            pool_size: Number of worker threads
        """
        self.store = store
        self.pool_size = pool_size

        # Processing components
        self.cleaner = DocumentCleaner()
        self.markdown_parser = MarkdownParser()
        self.html_parser = HTMLParser()
        self.chunker = SemanticChunker(config=ChunkingConfig(chunk_size=chunk_size))

        # Thread-safe state
        self._visited: Set[str] = set()
        self._visited_lock = threading.Lock()
        self._pending_futures: Set[Future] = set()
        self._futures_lock = threading.Lock()

    def _process_single_document(
        self,
        doc: FetchedDocument,
        base_metadata: dict,
        stats: ProcessingStats,
    ) -> List[PlatformDocChunk]:
        """Process a single document through the full pipeline.

        Args:
            doc: Fetched document to process
            base_metadata: Base metadata to include in chunks
            stats: Stats object to update

        Returns:
            List of chunks created
        """
        try:
            # Clean
            cleaned_doc = self.cleaner.clean(doc)

            # Parse based on content type
            if cleaned_doc.content_type in (CONTENT_TYPE_MARKDOWN, CONTENT_TYPE_RST):
                parsed = self.markdown_parser.parse(cleaned_doc)
            elif cleaned_doc.content_type == CONTENT_TYPE_HTML:
                parsed = self.html_parser.parse(cleaned_doc)
            else:
                logger.debug(f"Unknown content type '{cleaned_doc.content_type}', using Markdown parser")
                parsed = self.markdown_parser.parse(cleaned_doc)

            # Merge nav_path from fetcher metadata
            doc_metadata = doc.metadata or {}
            parsed_metadata = parsed.metadata if parsed.metadata is not None else {}
            if doc_metadata.get("nav_path"):
                parsed_metadata["nav_path"] = doc_metadata["nav_path"]
            if doc_metadata.get("group_name"):
                parsed_metadata["group_name"] = doc_metadata["group_name"]
            if parsed.metadata is None:
                parsed.metadata = parsed_metadata

            # Build chunk metadata
            chunk_metadata = {
                **base_metadata,
                "source_url": doc.source_url,
                "doc_path": doc.doc_path,
                "content_hash": doc_metadata.get("content_hash", ""),
            }
            # Use per-doc version if available (overridden from path in multi-version mode,
            # e.g., "1.3.0/intro.md" → version="1.3.0")
            if doc.version:
                chunk_metadata["version"] = doc.version

            # Chunk
            chunks = self.chunker.chunk(parsed, chunk_metadata)

            # Store
            if chunks:
                self.store.store_chunks(chunks)

            # Update stats
            stats.increment(docs=1, chunks=len(chunks))

            logger.debug(f"Processed: {doc.doc_path} -> {len(chunks)} chunks")

            return chunks

        except Exception as e:
            logger.warning(f"Failed to process {doc.doc_path}: {e}")
            stats.add_error(f"Process error ({doc.doc_path}): {str(e)}")
            return []

    def _mark_visited(self, url: str) -> bool:
        """Mark URL as visited, return True if it was new.

        Args:
            url: URL to mark as visited

        Returns:
            True if URL was not previously visited, False otherwise
        """
        with self._visited_lock:
            if url in self._visited:
                return False
            self._visited.add(url)
            return True

    def _is_visited(self, url: str) -> bool:
        """Check if URL has been visited."""
        with self._visited_lock:
            return url in self._visited

    # =========================================================================
    # Website Processing
    # =========================================================================

    def process_website(
        self,
        fetcher,  # WebFetcher
        base_url: str,
        version: str,
        platform: str,
        max_depth: int = 2,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        enforce_source_prefix: bool = True,
    ) -> ProcessingStats:
        """Process website documentation with streaming.

        Each URL is processed completely (fetch → chunk → store) in a thread.
        New URLs discovered during processing are added to the work queue.

        Args:
            fetcher: WebFetcher instance
            base_url: Starting URL
            version: Document version
            platform: Platform name
            max_depth: Maximum crawl depth
            include_patterns: URL patterns to include
            exclude_patterns: URL patterns to exclude
            enforce_source_prefix: Enforce source path prefix matching

        Returns:
            ProcessingStats with results
        """
        stats = ProcessingStats()

        # Normalize URL
        if not base_url.startswith(("http://", "https://")):
            base_url = "https://" + base_url

        parsed = urlparse(base_url)
        base_domain = parsed.netloc
        source_path_prefix = fetcher._extract_path_prefix(parsed.path) if enforce_source_prefix else None

        # Compile patterns
        include_re = [re.compile(p) for p in (include_patterns or [])]
        exclude_re = [re.compile(p) for p in (exclude_patterns or [])]

        base_metadata = {
            "platform": platform,
            "version": version,
            "source_type": "website",
        }

        logger.info(f"Starting streaming processing from {base_url} (max_depth={max_depth})")

        def process_url(url: str, depth: int):
            """Process a single URL and discover new links."""
            try:
                # Fetch page
                result = fetcher._fetch_page(url, depth, base_domain, version)
                if result is None:
                    return

                doc, links = result

                # Process document
                self._process_single_document(doc, base_metadata, stats)

                # Discover and submit new URLs
                if depth < max_depth:
                    for link in links:
                        # Check if should include
                        link_parsed = urlparse(link)
                        if link_parsed.netloc != base_domain:
                            continue
                        if source_path_prefix and not link_parsed.path.startswith(source_path_prefix):
                            continue
                        if not fetcher._should_include(link, include_re, exclude_re):
                            continue

                        # Submit if not visited
                        if self._mark_visited(link):
                            future = executor.submit(process_url, link, depth + 1)
                            with self._futures_lock:
                                self._pending_futures.add(future)
                                future.add_done_callback(lambda f: self._remove_future(f))

            except Exception as e:
                logger.warning(f"Error processing {url}: {e}")
                stats.add_error(f"Error ({url}): {str(e)}")

        def _wait_for_completion():
            """Wait for all pending futures to complete."""
            while True:
                with self._futures_lock:
                    if not self._pending_futures:
                        break
                    futures_copy = list(self._pending_futures)

                # Wait for at least one to complete
                for future in futures_copy:
                    try:
                        future.result(timeout=0.1)
                    except TimeoutError:
                        pass  # Expected - just polling
                    except Exception as e:
                        logger.debug(f"Future completed with error: {e}")

        # Reset state
        self._visited.clear()
        self._pending_futures.clear()

        # Mark base URL as visited and start processing
        self._mark_visited(base_url)

        with ThreadPoolExecutor(max_workers=self.pool_size) as executor:
            # Submit initial URL
            initial_future = executor.submit(process_url, base_url, 0)
            with self._futures_lock:
                self._pending_futures.add(initial_future)
                initial_future.add_done_callback(lambda f: self._remove_future(f))

            # Wait for all work to complete
            _wait_for_completion()

        logger.info(
            f"Website processing complete: {stats.total_docs} docs, "
            f"{stats.total_chunks} chunks in {stats.duration_seconds:.1f}s"
        )

        return stats

    def _remove_future(self, future: Future):
        """Remove completed future from pending set."""
        with self._futures_lock:
            self._pending_futures.discard(future)

    # =========================================================================
    # GitHub Processing
    # =========================================================================

    def process_github(
        self,
        fetcher,  # GitHubFetcher
        metadata,  # GitHubRepoMetadata
        version: str,
        platform: str,
    ) -> ProcessingStats:
        """Process GitHub repository documentation with streaming.

        Each file is processed completely (fetch → chunk → store) in a thread.

        Args:
            fetcher: GitHubFetcher instance
            metadata: GitHubRepoMetadata from collect_metadata()
            version: Document version
            platform: Platform name

        Returns:
            ProcessingStats with results
        """
        stats = ProcessingStats()

        if not metadata.file_paths:
            logger.warning("No files to process")
            return stats

        base_metadata = {
            "platform": platform,
            "version": version,
            "source_type": "github",
        }

        logger.info(f"Starting streaming processing for {len(metadata.file_paths)} GitHub files")

        def process_file(file_path: str):
            """Process a single file."""
            try:
                # Fetch single file
                docs = fetcher.fetch_batch(metadata, [file_path])
                if not docs:
                    return

                doc = docs[0]

                # Process document
                self._process_single_document(doc, base_metadata, stats)

            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                stats.add_error(f"Error ({file_path}): {str(e)}")

        with ThreadPoolExecutor(max_workers=self.pool_size) as executor:
            # Submit all files
            futures = [executor.submit(process_file, path) for path in metadata.file_paths]

            # Wait for completion and log progress
            completed = 0
            total = len(futures)
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Future failed: {e}")

                completed += 1
                if completed % 50 == 0 or completed == total:
                    logger.info(f"Progress: {completed}/{total} files processed")

        logger.info(
            f"GitHub processing complete: {stats.total_docs} docs, "
            f"{stats.total_chunks} chunks in {stats.duration_seconds:.1f}s"
        )

        return stats

    # =========================================================================
    # Local Processing
    # =========================================================================

    def process_local(
        self,
        fetcher,  # LocalFetcher
        documents: List[FetchedDocument],
        version: str,
        platform: str,
    ) -> ProcessingStats:
        """Process local documents with streaming.

        Each document is processed completely (chunk → store) in a thread.

        Args:
            fetcher: LocalFetcher instance (unused but kept for consistency)
            documents: List of already-fetched documents
            version: Document version
            platform: Platform name

        Returns:
            ProcessingStats with results
        """
        stats = ProcessingStats()

        if not documents:
            logger.warning("No documents to process")
            return stats

        base_metadata = {
            "platform": platform,
            "version": version,
            "source_type": "local",
        }

        logger.info(f"Starting streaming processing for {len(documents)} local documents")

        def process_doc(doc: FetchedDocument):
            """Process a single document."""
            try:
                self._process_single_document(doc, base_metadata, stats)
            except Exception as e:
                logger.warning(f"Error processing {doc.doc_path}: {e}")
                stats.add_error(f"Error ({doc.doc_path}): {str(e)}")

        with ThreadPoolExecutor(max_workers=self.pool_size) as executor:
            # Submit all documents
            futures = [executor.submit(process_doc, doc) for doc in documents]

            # Wait for completion and log progress
            completed = 0
            total = len(futures)
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Future failed: {e}")

                completed += 1
                if completed % 50 == 0 or completed == total:
                    logger.info(f"Progress: {completed}/{total} documents processed")

        logger.info(
            f"Local processing complete: {stats.total_docs} docs, "
            f"{stats.total_chunks} chunks in {stats.duration_seconds:.1f}s"
        )

        return stats
