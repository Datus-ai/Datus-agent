"""Turn snapshots compare original bytes with final bytes, independent of tools."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datus.api.services.semantic_model_diff import SemanticModelSnapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor,expected", [(None, False), (0, True)])
async def test_completed_turn_resume_only_replays_diff_for_cursor_reconnect(cursor, expected):
    """Replay transient diffs for explicit cursor reconnects, not ordinary history entry."""
    from datus.api.models.cli_models import (
        IMessageContent,
        SSEDataType,
        SSEEvent,
        SSEMessageData,
        SSEMessagePayload,
    )
    from datus.api.routes.chat_routes import ResumeChatInput, resume_chat
    from datus.api.services.chat_task_manager import ChatTask, ChatTaskManager

    manager = ChatTaskManager()
    task = ChatTask("session-diff", MagicMock(), user_id="user")
    task.status = "completed"
    manager._completed_tasks[task.session_id] = task
    await manager._push_event(
        task,
        SSEEvent(
            id=0,
            event="message",
            data=SSEMessageData(
                type=SSEDataType.CREATE_MESSAGE,
                payload=SSEMessagePayload(
                    message_id="diff",
                    role="assistant",
                    content=[IMessageContent(type="semantic-model-diff", payload={"turnId": "turn", "files": []})],
                ),
            ),
        ),
    )
    response = await resume_chat(
        ResumeChatInput(session_id=task.session_id, source="web", from_event_id=cursor),
        SimpleNamespace(task_manager=manager),
        SimpleNamespace(user_id="user"),
    )
    stream = "".join([chunk async for chunk in response.body_iterator])
    assert ("semantic-model-diff" in stream) is expected


def test_aggregates_writes_creates_and_deletions(tmp_path):
    """Compare all changed files against the starting bytes, including nested YAML paths."""
    root = tmp_path / "subject" / "semantic_models" / "ds" / "nested" / "models"
    root.mkdir(parents=True)
    model = root / "model.yml"
    model.write_text("metrics: [old, retained]\n")
    removed = root / "removed.yaml"
    removed.write_text("removed\n")
    snapshot = SemanticModelSnapshot.capture(tmp_path)
    model.write_text("metrics: [old, retained, new]\n")
    model.write_text("metrics: [retained, new]\n")
    removed.unlink()
    created = root / "created.yaml"
    created.write_text("new model\n")
    (root / "ignored.json").write_text("{}")

    changes = {change["filePath"]: change for change in snapshot.changes()}
    path = "subject/semantic_models/ds/nested/models/"
    assert changes[path + "model.yml"] == {
        "filePath": path + "model.yml",
        "changeType": "modified",
        "oldText": "metrics: [old, retained]\n",
        "newText": "metrics: [retained, new]\n",
    }
    assert changes[path + "created.yaml"]["oldText"] is None
    assert changes[path + "removed.yaml"]["newText"] is None
    assert len(changes) == 3


def test_no_change_after_rollback_or_create_then_delete(tmp_path):
    """Omit restored bytes and temporary files removed before the turn ends."""
    root = tmp_path / "subject" / "semantic_models" / "ds"
    root.mkdir(parents=True)
    model = root / "model.yml"
    model.write_bytes(b"original\r\n")
    snapshot = SemanticModelSnapshot.capture(tmp_path)
    model.write_text("intermediate")
    model.write_bytes(b"original\r\n")
    created = root / "temporary.yml"
    created.write_text("temporary")
    created.unlink()
    assert snapshot.changes() == []


def test_captures_new_directory_and_distinguishes_empty_file(tmp_path):
    """Treat an empty new YAML file as an addition rather than a missing snapshot."""
    snapshot = SemanticModelSnapshot.capture(tmp_path)
    root = tmp_path / "subject" / "semantic_models" / "ds"
    root.mkdir(parents=True)
    (root / "empty.yml").touch()
    assert snapshot.changes() == [
        {
            "filePath": "subject/semantic_models/ds/empty.yml",
            "changeType": "added",
            "oldText": None,
            "newText": "",
        }
    ]


def test_unreadable_scan_is_not_a_deletion_and_symlinks_are_ignored(tmp_path):
    """Ignore symlinks and suppress incomplete scans instead of inventing deletions."""
    root = tmp_path / "subject" / "semantic_models"
    root.mkdir(parents=True)
    external = tmp_path / "external.yml"
    external.write_text("outside")
    (root / "link.yml").symlink_to(external)
    snapshot = SemanticModelSnapshot.capture(tmp_path)
    assert snapshot.files == {}
    model = root / "invalid.yml"
    model.write_bytes(b"\xff")
    assert snapshot.changes() == []
    assert SemanticModelSnapshot.capture(tmp_path) is None


def test_scan_failure_never_emits_partial_changes(tmp_path, monkeypatch):
    """Suppress a failed directory scan rather than reporting partial file changes."""
    snapshot = SemanticModelSnapshot.capture(tmp_path)

    def fail(_path, *_args, **_kwargs):
        """Simulate an unreadable semantic-model directory."""
        raise PermissionError("cannot enumerate")

    monkeypatch.setattr(Path, "rglob", fail)
    root = tmp_path / "subject" / "semantic_models"
    root.mkdir(parents=True)
    assert snapshot.changes() == []
