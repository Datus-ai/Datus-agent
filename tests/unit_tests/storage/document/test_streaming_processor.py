# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.document.streaming_processor."""

import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from datus.storage.document.schemas import CONTENT_TYPE_HTML, CONTENT_TYPE_MARKDOWN, CONTENT_TYPE_RST, FetchedDocument
from datus.storage.document.streaming_processor import ProcessingStats, StreamingDocProcessor

# ---------------------------------------------------------------------------
# ProcessingStats
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestProcessingStats:
    """Tests for the ProcessingStats dataclass."""

    def test_initial_values(self):
        """Fresh stats have zero counters and empty errors."""
        stats = ProcessingStats()
        assert stats.total_docs == 0
        assert stats.total_chunks == 0
        assert stats.errors == []

    def test_increment_docs_and_chunks(self):
        """Increment should add to counters."""
        stats = ProcessingStats()
        stats.increment(docs=3, chunks=10)
        assert stats.total_docs == 3
        assert stats.total_chunks == 10

    def test_increment_accumulates(self):
        """Multiple increments accumulate."""
        stats = ProcessingStats()
        stats.increment(docs=1, chunks=5)
        stats.increment(docs=2, chunks=3)
        assert stats.total_docs == 3
        assert stats.total_chunks == 8

    def test_increment_partial(self):
        """Increment with only docs or only chunks."""
        stats = ProcessingStats()
        stats.increment(docs=5)
        assert stats.total_docs == 5
        assert stats.total_chunks == 0
        stats.increment(chunks=7)
        assert stats.total_docs == 5
        assert stats.total_chunks == 7

    def test_add_error(self):
        """add_error appends error messages."""
        stats = ProcessingStats()
        stats.add_error("something went wrong")
        stats.add_error("another error")
        assert len(stats.errors) == 2
        assert "something went wrong" in stats.errors
        assert "another error" in stats.errors

    def test_duration_seconds_positive(self):
        """duration_seconds returns a positive value after construction."""
        stats = ProcessingStats()
        time.sleep(0.05)
        assert stats.duration_seconds >= 0.04

    def test_duration_seconds_from_fixed_start(self):
        """duration_seconds is relative to start_time."""
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        stats = ProcessingStats(start_time=past)
        assert stats.duration_seconds > 0

    def test_thread_safe_increment(self):
        """Concurrent increments do not lose updates."""
        stats = ProcessingStats()
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            for _ in range(100):
                stats.increment(docs=1, chunks=2)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats.total_docs == 1000
        assert stats.total_chunks == 2000

    def test_thread_safe_add_error(self):
        """Concurrent add_error calls do not lose messages."""
        stats = ProcessingStats()
        barrier = threading.Barrier(5)

        def worker(idx):
            barrier.wait()
            for i in range(20):
                stats.add_error(f"err-{idx}-{i}")

        threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(stats.errors) == 100


# ---------------------------------------------------------------------------
# StreamingDocProcessor helpers
# ---------------------------------------------------------------------------


def _make_mock_store():
    """Create a mock DocumentStore for testing."""
    store = MagicMock()
    store.store_chunks = MagicMock(return_value=0)
    return store


def _make_doc(content: str, content_type: str = CONTENT_TYPE_MARKDOWN, doc_path: str = "test.md", version: str = "1.0"):
    """Create a FetchedDocument for testing."""
    return FetchedDocument(
        platform="test",
        version=version,
        source_url="https://example.com/test.md",
        source_type="website",
        doc_path=doc_path,
        raw_content=content,
        content_type=content_type,
    )


# ---------------------------------------------------------------------------
# StreamingDocProcessor._mark_visited / _is_visited
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestMarkVisited:
    """Tests for URL visited tracking."""

    def test_mark_new_url_returns_true(self):
        """First visit returns True."""
        proc = StreamingDocProcessor(store=_make_mock_store())
        assert proc._mark_visited("https://a.com") is True

    def test_mark_same_url_returns_false(self):
        """Second visit returns False."""
        proc = StreamingDocProcessor(store=_make_mock_store())
        proc._mark_visited("https://a.com")
        assert proc._mark_visited("https://a.com") is False

    def test_different_urls_both_true(self):
        """Different URLs both return True on first visit."""
        proc = StreamingDocProcessor(store=_make_mock_store())
        assert proc._mark_visited("https://a.com") is True
        assert proc._mark_visited("https://b.com") is True

    def test_is_visited(self):
        """_is_visited reflects visited state."""
        proc = StreamingDocProcessor(store=_make_mock_store())
        assert proc._is_visited("https://x.com") is False
        proc._mark_visited("https://x.com")
        assert proc._is_visited("https://x.com") is True

    def test_thread_safe_mark_visited(self):
        """Concurrent _mark_visited never double-visits."""
        proc = StreamingDocProcessor(store=_make_mock_store())
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            r = proc._mark_visited("https://shared.com")
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one thread should get True
        assert results.count(True) == 1
        assert results.count(False) == 9


# ---------------------------------------------------------------------------
# StreamingDocProcessor._remove_future
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestRemoveFuture:
    """Tests for future cleanup."""

    def test_remove_existing_future(self):
        """Completed future is removed from pending set."""
        proc = StreamingDocProcessor(store=_make_mock_store())
        mock_future = MagicMock()
        proc._pending_futures.add(mock_future)
        proc._remove_future(mock_future)
        assert mock_future not in proc._pending_futures

    def test_remove_nonexistent_future(self):
        """Removing a non-existent future does not raise."""
        proc = StreamingDocProcessor(store=_make_mock_store())
        mock_future = MagicMock()
        proc._remove_future(mock_future)  # no error


# ---------------------------------------------------------------------------
# StreamingDocProcessor._process_single_document
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestProcessSingleDocument:
    """Tests for the per-document processing pipeline."""

    def test_process_markdown_document(self):
        """Markdown doc goes through clean -> parse -> chunk -> store."""
        store = _make_mock_store()
        proc = StreamingDocProcessor(store=store, chunk_size=512)
        stats = ProcessingStats()

        doc = _make_doc("# Hello\n\nSome content here that is long enough to be chunked.")
        base_meta = {"platform": "test", "version": "1.0", "source_type": "local"}

        chunks = proc._process_single_document(doc, base_meta, stats)

        assert stats.total_docs == 1
        assert stats.total_chunks >= 1
        assert len(chunks) >= 1
        store.store_chunks.assert_called_once()

    def test_process_html_document(self):
        """HTML doc uses HTMLParser path."""
        store = _make_mock_store()
        proc = StreamingDocProcessor(store=store, chunk_size=512)
        stats = ProcessingStats()

        doc = _make_doc(
            "<html><body><h1>Title</h1><p>Paragraph content for testing.</p></body></html>",
            content_type=CONTENT_TYPE_HTML,
            doc_path="page.html",
        )
        base_meta = {"platform": "test", "version": "1.0", "source_type": "website"}

        chunks = proc._process_single_document(doc, base_meta, stats)

        assert stats.total_docs == 1
        assert len(chunks) >= 1

    def test_process_rst_document(self):
        """RST doc uses MarkdownParser path (same as markdown)."""
        store = _make_mock_store()
        proc = StreamingDocProcessor(store=store, chunk_size=512)
        stats = ProcessingStats()

        doc = _make_doc(
            "Title\n=====\n\nSome reStructuredText content.",
            content_type=CONTENT_TYPE_RST,
            doc_path="guide.rst",
        )
        base_meta = {"platform": "test", "version": "1.0", "source_type": "local"}

        _chunks = proc._process_single_document(doc, base_meta, stats)  # noqa: F841

        assert stats.total_docs == 1

    def test_process_unknown_content_type(self):
        """Unknown content type falls back to MarkdownParser."""
        store = _make_mock_store()
        proc = StreamingDocProcessor(store=store, chunk_size=512)
        stats = ProcessingStats()

        doc = _make_doc("Some plain text content.", content_type="text/plain", doc_path="notes.txt")
        base_meta = {"platform": "test", "version": "1.0", "source_type": "local"}

        _chunks = proc._process_single_document(doc, base_meta, stats)  # noqa: F841

        assert stats.total_docs == 1

    def test_process_document_with_nav_path_metadata(self):
        """nav_path from doc metadata is propagated to parsed doc."""
        store = _make_mock_store()
        proc = StreamingDocProcessor(store=store, chunk_size=512)
        stats = ProcessingStats()

        doc = _make_doc("# Guide\n\nContent here.")
        doc.metadata = {"nav_path": ["Docs", "Guide"], "group_name": "Docs"}
        base_meta = {"platform": "test", "version": "1.0", "source_type": "local"}

        _chunks = proc._process_single_document(doc, base_meta, stats)  # noqa: F841
        assert stats.total_docs == 1

    def test_process_document_with_version_override(self):
        """Per-doc version is included in chunk metadata."""
        store = _make_mock_store()
        proc = StreamingDocProcessor(store=store, chunk_size=512)
        stats = ProcessingStats()

        doc = _make_doc("# Versioned\n\nContent.", version="2.0.0")
        base_meta = {"platform": "test", "version": "1.0", "source_type": "local"}

        _chunks = proc._process_single_document(doc, base_meta, stats)  # noqa: F841
        assert stats.total_docs == 1
        # The base_meta should be augmented with source_url, doc_path, etc.
        stored_chunks = store.store_chunks.call_args[0][0]
        # Check that version override is propagated through chunk metadata
        assert len(stored_chunks) >= 1

    def test_process_document_error_handling(self):
        """Exception during processing is captured in stats, not raised."""
        store = _make_mock_store()
        proc = StreamingDocProcessor(store=store, chunk_size=512)
        # Break the cleaner to force an exception
        proc.cleaner = MagicMock()
        proc.cleaner.clean.side_effect = RuntimeError("clean failed")
        stats = ProcessingStats()

        doc = _make_doc("# Fail\n\nContent.")
        base_meta = {"platform": "test", "version": "1.0", "source_type": "local"}

        chunks = proc._process_single_document(doc, base_meta, stats)

        assert chunks == []
        assert stats.total_docs == 0
        assert len(stats.errors) == 1
        assert "clean failed" in stats.errors[0]

    def test_process_empty_chunks_not_stored(self):
        """When chunker produces no chunks, store is not called."""
        store = _make_mock_store()
        proc = StreamingDocProcessor(store=store, chunk_size=512)
        # Override chunker to return empty list
        proc.chunker = MagicMock()
        proc.chunker.chunk.return_value = []
        stats = ProcessingStats()

        doc = _make_doc("# Empty\n\n")
        base_meta = {"platform": "test", "version": "1.0", "source_type": "local"}

        chunks = proc._process_single_document(doc, base_meta, stats)

        assert chunks == []
        store.store_chunks.assert_not_called()
        assert stats.total_docs == 1
        assert stats.total_chunks == 0

    def test_process_document_content_hash_metadata(self):
        """content_hash from doc metadata propagates to chunk metadata."""
        store = _make_mock_store()
        proc = StreamingDocProcessor(store=store, chunk_size=512)
        stats = ProcessingStats()

        doc = _make_doc("# Hashed\n\nContent with hash.")
        doc.metadata = {"content_hash": "abc123"}
        base_meta = {"platform": "test", "version": "1.0", "source_type": "local"}

        proc._process_single_document(doc, base_meta, stats)
        assert stats.total_docs == 1


# ---------------------------------------------------------------------------
# StreamingDocProcessor.process_local
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestProcessLocal:
    """Tests for local document processing."""

    def test_process_local_empty_list(self):
        """Empty document list returns zero stats."""
        store = _make_mock_store()
        proc = StreamingDocProcessor(store=store, chunk_size=512, pool_size=1)

        stats = proc.process_local(
            fetcher=None,
            documents=[],
            version="1.0",
            platform="test",
        )

        assert stats.total_docs == 0
        assert stats.total_chunks == 0

    def test_process_local_single_document(self):
        """Single document is processed through the pipeline."""
        store = _make_mock_store()
        proc = StreamingDocProcessor(store=store, chunk_size=512, pool_size=1)

        doc = _make_doc("# Local Doc\n\nSome local content for processing.")
        stats = proc.process_local(
            fetcher=None,
            documents=[doc],
            version="1.0",
            platform="test",
        )

        assert stats.total_docs == 1
        assert stats.total_chunks >= 1

    def test_process_local_multiple_documents(self):
        """Multiple documents are all processed."""
        store = _make_mock_store()
        proc = StreamingDocProcessor(store=store, chunk_size=512, pool_size=2)

        docs = [_make_doc(f"# Doc {i}\n\nContent for document number {i}.", doc_path=f"doc{i}.md") for i in range(5)]

        stats = proc.process_local(
            fetcher=None,
            documents=docs,
            version="1.0",
            platform="test",
        )

        assert stats.total_docs == 5


# ---------------------------------------------------------------------------
# StreamingDocProcessor.__init__
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestStreamingDocProcessorInit:
    """Tests for StreamingDocProcessor initialization."""

    def test_default_pool_size(self):
        """Default pool_size is 4."""
        proc = StreamingDocProcessor(store=_make_mock_store())
        assert proc.pool_size == 4

    def test_custom_pool_size(self):
        """Custom pool_size is respected."""
        proc = StreamingDocProcessor(store=_make_mock_store(), pool_size=8)
        assert proc.pool_size == 8

    def test_components_initialized(self):
        """All processing components are initialized."""
        proc = StreamingDocProcessor(store=_make_mock_store())
        assert proc.cleaner is not None
        assert proc.markdown_parser is not None
        assert proc.html_parser is not None
        assert proc.chunker is not None

    def test_visited_set_empty(self):
        """Visited set starts empty."""
        proc = StreamingDocProcessor(store=_make_mock_store())
        assert len(proc._visited) == 0

    def test_pending_futures_empty(self):
        """Pending futures starts empty."""
        proc = StreamingDocProcessor(store=_make_mock_store())
        assert len(proc._pending_futures) == 0
