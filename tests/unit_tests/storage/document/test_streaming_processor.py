"""Unit tests for StreamingDocProcessor — on_doc_complete callback."""

import threading
from unittest.mock import MagicMock

from datus.storage.document.schemas import CONTENT_TYPE_MARKDOWN, FetchedDocument
from datus.storage.document.streaming_processor import ProcessingStats, StreamingDocProcessor


class TestOnDocCompleteCallback:
    """Tests for the on_doc_complete callback in StreamingDocProcessor."""

    def _make_processor(self, store=None, on_doc_complete=None):
        """Create a StreamingDocProcessor with a mock store."""
        mock_store = store or MagicMock()
        mock_store.store_chunks = MagicMock(return_value=3)
        return StreamingDocProcessor(
            store=mock_store,
            chunk_size=1024,
            pool_size=1,
            on_doc_complete=on_doc_complete,
        )

    def _make_doc(self, path="test.md", content="# Hello\n\nSome content here."):
        return FetchedDocument(
            platform="test",
            version="1.0",
            source_url=f"https://example.com/{path}",
            source_type="local",
            doc_path=path,
            raw_content=content,
            content_type=CONTENT_TYPE_MARKDOWN,
        )

    def test_callback_invoked_on_success(self):
        """on_doc_complete is called with (doc_path, chunk_count) on success."""
        callback = MagicMock()
        processor = self._make_processor(on_doc_complete=callback)
        doc = self._make_doc()
        stats = ProcessingStats()

        processor._process_single_document(doc, {"platform": "test", "version": "1.0"}, stats)

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == "test.md"
        assert isinstance(call_args[1], int)
        assert call_args[1] >= 0

    def test_callback_invoked_on_failure(self):
        """on_doc_complete is called with chunk_count=0 on processing failure."""
        callback = MagicMock()
        mock_store = MagicMock()
        # Chunker succeeds, but store_chunks raises to simulate storage failure
        mock_store.store_chunks = MagicMock(side_effect=RuntimeError("store error"))
        processor = StreamingDocProcessor(
            store=mock_store,
            chunk_size=1024,
            pool_size=1,
            on_doc_complete=callback,
        )
        doc = self._make_doc()
        stats = ProcessingStats()

        # Should not raise
        result = processor._process_single_document(doc, {"platform": "test", "version": "1.0"}, stats)
        assert result == []

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == "test.md"
        assert call_args[1] == 0

    def test_no_callback_does_not_crash(self):
        """Default (None) callback processes normally."""
        processor = self._make_processor(on_doc_complete=None)
        doc = self._make_doc()
        stats = ProcessingStats()

        processor._process_single_document(doc, {"platform": "test", "version": "1.0"}, stats)
        assert stats.total_docs == 1

    def test_broken_callback_does_not_halt_processing(self):
        """A callback that raises does not prevent document processing."""
        callback = MagicMock(side_effect=RuntimeError("callback boom"))
        processor = self._make_processor(on_doc_complete=callback)
        doc = self._make_doc()
        stats = ProcessingStats()

        processor._process_single_document(doc, {"platform": "test", "version": "1.0"}, stats)

        # Processing still succeeded despite callback failure
        assert stats.total_docs == 1
        assert stats.total_chunks > 0
        callback.assert_called_once()

    def test_callback_receives_correct_chunk_count(self):
        """Callback receives the actual number of chunks created."""
        chunks_created = []

        def track_callback(doc_path, chunk_count):
            chunks_created.append((doc_path, chunk_count))

        processor = self._make_processor(on_doc_complete=track_callback)
        doc = self._make_doc(content="# Title\n\n" + "Content paragraph. " * 50)
        stats = ProcessingStats()

        processor._process_single_document(doc, {"platform": "test", "version": "1.0"}, stats)

        assert len(chunks_created) == 1
        assert chunks_created[0][0] == "test.md"
        assert chunks_created[0][1] == stats.total_chunks


class TestProcessingStats:
    """Tests for ProcessingStats thread safety."""

    def test_thread_safe_increment(self):
        """Multiple threads can safely increment stats."""
        stats = ProcessingStats()

        def increment_many():
            for _ in range(100):
                stats.increment(docs=1, chunks=5)

        threads = [threading.Thread(target=increment_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats.total_docs == 400
        assert stats.total_chunks == 2000

    def test_thread_safe_add_error(self):
        """Multiple threads can safely add errors."""
        stats = ProcessingStats()

        def add_errors():
            for i in range(50):
                stats.add_error(f"error-{i}")

        threads = [threading.Thread(target=add_errors) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(stats.errors) == 200
