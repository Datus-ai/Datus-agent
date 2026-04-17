# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for TaskStore across all available storage backends."""

import pytest

from datus.storage.task.store import TaskStore


@pytest.fixture
def task_store(storage_test_project):
    """Create a TaskStore backed by the current test backend."""
    return TaskStore(project=storage_test_project)


class TestTaskStoreInit:
    """Tests for TaskStore initialization."""

    def test_table_created(self, task_store):
        """The tasks table is created and queryable on initialization."""
        task = task_store.get_task("nonexistent")
        assert task is None


class TestTaskStoreCrud:
    """Tests for task CRUD operations."""

    def test_create_task(self, task_store):
        """create_task creates a new task and returns its data."""
        result = task_store.create_task("t1", "SELECT 1")
        assert result["task_id"] == "t1"
        assert result["task_query"] == "SELECT 1"
        assert result["status"] == "running"
        assert result["created_at"] != ""

    def test_get_task(self, task_store):
        """get_task retrieves a created task."""
        task_store.create_task("t2", "SELECT 2")
        task = task_store.get_task("t2")
        assert task is not None
        assert task["task_id"] == "t2"
        assert task["task_query"] == "SELECT 2"

    def test_get_task_nonexistent(self, task_store):
        """get_task returns None for nonexistent task."""
        assert task_store.get_task("nonexistent") is None

    def test_update_task(self, task_store):
        """update_task modifies task fields."""
        task_store.create_task("t3", "SELECT 3")
        result = task_store.update_task("t3", sql_query="SELECT * FROM t", status="completed")
        assert result is True

        task = task_store.get_task("t3")
        assert task["sql_query"] == "SELECT * FROM t"
        assert task["status"] == "completed"

    def test_update_task_nonexistent(self, task_store):
        """update_task returns False for nonexistent task."""
        assert task_store.update_task("missing", status="done") is False

    def test_delete_task(self, task_store):
        """delete_task removes a task."""
        task_store.create_task("t4", "SELECT 4")
        assert task_store.delete_task("t4") is True
        assert task_store.get_task("t4") is None

    def test_delete_task_nonexistent(self, task_store):
        """delete_task returns False for nonexistent task."""
        assert task_store.delete_task("missing") is False


class TestTaskStoreFeedback:
    """Tests for feedback operations."""

    def test_record_and_get_feedback(self, task_store):
        """record_feedback stores feedback and get_feedback retrieves it."""
        task_store.create_task("f1", "SELECT f1")
        result = task_store.record_feedback("f1", "positive")
        assert result["user_feedback"] == "positive"
        assert result["task_id"] == "f1"

        feedback = task_store.get_feedback("f1")
        assert feedback is not None
        assert feedback["user_feedback"] == "positive"

    def test_get_feedback_none(self, task_store):
        """get_feedback returns None when no feedback exists."""
        task_store.create_task("f2", "SELECT f2")
        assert task_store.get_feedback("f2") is None

    def test_get_all_feedback(self, task_store):
        """get_all_feedback returns all tasks with feedback."""
        task_store.create_task("af1", "q1")
        task_store.create_task("af2", "q2")
        task_store.record_feedback("af1", "good")
        task_store.record_feedback("af2", "bad")

        all_fb = task_store.get_all_feedback()
        assert len(all_fb) == 2
        ids = {f["task_id"] for f in all_fb}
        assert ids == {"af1", "af2"}

    def test_delete_feedback(self, task_store):
        """delete_feedback clears feedback for a task."""
        task_store.create_task("df1", "q")
        task_store.record_feedback("df1", "ok")
        assert task_store.delete_feedback("df1") is True
        assert task_store.get_feedback("df1") is None

    def test_delete_feedback_no_feedback(self, task_store):
        """delete_feedback returns False when no feedback exists."""
        task_store.create_task("df2", "q")
        assert task_store.delete_feedback("df2") is False


class TestTaskStoreEdgeCases:
    """Tests for edge cases."""

    def test_record_feedback_nonexistent_task(self, task_store):
        """record_feedback for nonexistent task raises DatusException."""
        from datus.utils.exceptions import DatusException

        with pytest.raises(DatusException):
            task_store.record_feedback("no_such_task", "feedback")

    def test_cleanup_old_tasks(self, task_store):
        """cleanup_old_tasks removes tasks older than cutoff."""
        task_store.create_task("old1", "q1")
        # Tasks just created won't be old enough; cleanup returns 0
        count = task_store.cleanup_old_tasks(hours=24)
        assert count == 0

    def test_create_task_idempotent(self, task_store):
        """Creating a task with same ID preserves original state (no clobber)."""
        task_store.create_task("dup", "query1")
        result = task_store.create_task("dup", "query2")
        assert result["task_query"] == "query1"

        task = task_store.get_task("dup")
        assert task is not None
        assert task["task_query"] == "query1"
