# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Business-level storage assertions shared by local and service acceptance."""

from uuid import uuid4

from datus.storage.feedback.store import FeedbackStore
from datus.storage.task.store import TaskStore


def assert_feedback_storage_crud(project: str) -> dict[str, object]:
    """Verify public CRUD behavior and return both final persisted values."""
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
    return {"feedback": feedback_store.get_feedback(feedback_task_id), "task": task_store.get_feedback(chat_task_id)}
