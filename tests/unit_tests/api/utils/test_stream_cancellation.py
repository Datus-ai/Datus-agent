"""Tests for datus.api.utils.stream_cancellation — SSE cancel token management."""

import pytest

from datus.api.utils import stream_cancellation
from datus.api.utils.stream_cancellation import (
    cancel_stream,
    cleanup_cancel_token,
    create_cancel_token,
)


@pytest.fixture(autouse=True)
def _clear_tokens():
    """Ensure token dict is clean before/after each test."""
    stream_cancellation._tokens.clear()
    yield
    stream_cancellation._tokens.clear()


class TestCreateCancelToken:
    """Tests for create_cancel_token lifecycle."""

    def test_creates_asyncio_event(self):
        """Token is an asyncio.Event registered in the global dict."""
        import asyncio

        event = create_cancel_token("stream-1")
        assert isinstance(event, asyncio.Event)
        assert not event.is_set()

    def test_token_stored_by_stream_id(self):
        """Created token is retrievable by stream_id."""
        event = create_cancel_token("stream-abc")
        assert stream_cancellation._tokens["stream-abc"] is event

    def test_overwrite_existing_token(self):
        """Creating token with same ID replaces the old one."""
        old = create_cancel_token("dup")
        new = create_cancel_token("dup")
        assert old is not new
        assert stream_cancellation._tokens["dup"] is new


class TestCancelStream:
    """Tests for cancel_stream signaling."""

    def test_cancel_existing_stream_returns_true(self):
        """Cancelling an existing stream sets the event and returns True."""
        event = create_cancel_token("s1")
        result = cancel_stream("s1")
        assert result is True
        assert event.is_set()

    def test_cancel_nonexistent_stream_returns_false(self):
        """Cancelling a stream that doesn't exist returns False."""
        result = cancel_stream("nonexistent")
        assert result is False

    def test_cancel_idempotent(self):
        """Cancelling the same stream twice still returns True."""
        create_cancel_token("s2")
        assert cancel_stream("s2") is True
        assert cancel_stream("s2") is True


class TestCleanupCancelToken:
    """Tests for cleanup_cancel_token removal."""

    def test_cleanup_removes_token(self):
        """Cleanup removes the token from the global dict."""
        create_cancel_token("to-clean")
        cleanup_cancel_token("to-clean")
        assert "to-clean" not in stream_cancellation._tokens

    def test_cleanup_nonexistent_is_noop(self):
        """Cleanup of non-existent token doesn't raise."""
        cleanup_cancel_token("ghost")
        assert "ghost" not in stream_cancellation._tokens

    def test_full_lifecycle(self):
        """Create → cancel → cleanup: each step works as expected."""
        event = create_cancel_token("lifecycle")
        assert not event.is_set()
        assert cancel_stream("lifecycle") is True
        assert event.is_set()
        cleanup_cancel_token("lifecycle")
        assert "lifecycle" not in stream_cancellation._tokens
