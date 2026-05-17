# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from uuid import uuid4

import pytest

from datus.storage.feedback.store import FeedbackStore
from datus.storage.task.store import TaskStore


@pytest.mark.acceptance
def test_feedback_storage_write_read_update_and_delete(real_agent_config):
    """Feedback storage supports deterministic write/read-back for both feedback tables."""
    project = real_agent_config.project_name
    feedback_task_id = f"task-feedback-{uuid4().hex}"
    chat_task_id = f"task-chat-{uuid4().hex}"

    feedback_store = FeedbackStore(project=project)
    first = feedback_store.record_feedback(feedback_task_id, "success")
    assert first["task_id"] == feedback_task_id
    assert first["status"] == "success"

    feedback_store.record_feedback(feedback_task_id, "failure")
    assert feedback_store.get_feedback(feedback_task_id)["status"] == "failure"
    all_feedback = feedback_store.get_all_feedback()
    assert any(item["task_id"] == feedback_task_id and item["status"] == "failure" for item in all_feedback)
    assert feedback_store.delete_feedback(feedback_task_id) is True
    assert feedback_store.get_feedback(feedback_task_id) is None

    task_store = TaskStore(project=project)
    task_store.create_task(chat_task_id, "How many schools are in Alameda?")
    recorded = task_store.record_feedback(chat_task_id, "success")
    assert recorded["task_id"] == chat_task_id
    assert recorded["user_feedback"] == "success"
    assert task_store.get_feedback(chat_task_id)["user_feedback"] == "success"
    assert task_store.delete_feedback(chat_task_id) is True
    assert task_store.get_feedback(chat_task_id) is None
