# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pytest

from datus.storage.feedback.store import FeedbackStore
from datus.storage.task.store import TaskStore


@pytest.mark.acceptance
def test_feedback_storage_write_read_update_and_delete(real_agent_config):
    """Feedback storage supports deterministic write/read-back for both feedback tables."""
    project = real_agent_config.project_name

    feedback_store = FeedbackStore(project=project)
    first = feedback_store.record_feedback("task-feedback-1", "success")
    assert first["task_id"] == "task-feedback-1"
    assert first["status"] == "success"

    feedback_store.record_feedback("task-feedback-1", "failure")
    assert feedback_store.get_feedback("task-feedback-1")["status"] == "failure"
    assert [item["task_id"] for item in feedback_store.get_all_feedback()] == ["task-feedback-1"]
    assert feedback_store.delete_feedback("task-feedback-1") is True
    assert feedback_store.get_feedback("task-feedback-1") is None

    task_store = TaskStore(project=project)
    task_store.create_task("task-chat-1", "How many schools are in Alameda?")
    recorded = task_store.record_feedback("task-chat-1", "success")
    assert recorded["task_id"] == "task-chat-1"
    assert recorded["user_feedback"] == "success"
    assert task_store.get_feedback("task-chat-1")["user_feedback"] == "success"
    assert task_store.delete_feedback("task-chat-1") is True
    assert task_store.get_feedback("task-chat-1") is None
