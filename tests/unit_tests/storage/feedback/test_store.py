# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for FeedbackStore across all available storage backends."""

import pytest

from datus.storage.feedback.store import FeedbackStore


@pytest.fixture
def feedback_store(storage_test_project):
    """Create a FeedbackStore backed by the current test backend."""
    return FeedbackStore(project=storage_test_project)


class TestFeedbackStoreInit:
    """Tests for FeedbackStore initialization."""

    def test_table_created(self, feedback_store):
        """The feedback table is created and queryable on initialization."""
        results = feedback_store.get_all_feedback()
        assert isinstance(results, list)


class TestFeedbackStoreCrud:
    """Tests for CRUD operations."""

    def test_record_feedback(self, feedback_store):
        """record_feedback stores and returns feedback data."""
        result = feedback_store.record_feedback("task1", "positive")
        assert result["task_id"] == "task1"
        assert result["status"] == "positive"
        assert result["recorded_at"] != ""

    def test_get_feedback(self, feedback_store):
        """get_feedback retrieves stored feedback."""
        feedback_store.record_feedback("task2", "negative")
        fb = feedback_store.get_feedback("task2")
        assert fb is not None
        assert fb["task_id"] == "task2"
        assert fb["status"] == "negative"

    def test_get_feedback_nonexistent(self, feedback_store):
        """get_feedback returns None for nonexistent task."""
        assert feedback_store.get_feedback("missing") is None

    def test_get_all_feedback(self, feedback_store):
        """get_all_feedback returns all feedback entries."""
        feedback_store.record_feedback("t1", "good")
        feedback_store.record_feedback("t2", "bad")
        all_fb = feedback_store.get_all_feedback()
        assert len(all_fb) == 2
        ids = {f["task_id"] for f in all_fb}
        assert ids == {"t1", "t2"}

    def test_delete_feedback(self, feedback_store):
        """delete_feedback removes a feedback entry."""
        feedback_store.record_feedback("d1", "ok")
        assert feedback_store.delete_feedback("d1") is True
        assert feedback_store.get_feedback("d1") is None

    def test_delete_feedback_nonexistent(self, feedback_store):
        """delete_feedback returns False for nonexistent entry."""
        assert feedback_store.delete_feedback("missing") is False

    def test_upsert_feedback(self, feedback_store):
        """Recording feedback twice for same task_id upserts."""
        feedback_store.record_feedback("u1", "first")
        feedback_store.record_feedback("u1", "updated")
        fb = feedback_store.get_feedback("u1")
        assert fb is not None
        assert fb["status"] == "updated"

    def test_get_all_feedback_empty(self, feedback_store):
        """get_all_feedback returns empty list when no feedback exists."""
        assert feedback_store.get_all_feedback() == []
