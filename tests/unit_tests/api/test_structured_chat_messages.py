# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Structured chat inputs keep user content separate from host-resolved paths."""

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request
from pydantic import ValidationError

from datus.api.hooks.message_hooks import prepare_chat_messages, set_chat_message_resolver
from datus.api.models.chat_models import InsertMessageInput
from datus.api.models.cli_models import StreamChatInput
from datus.api.routes.chat_routes import insert_message
from datus.api.services.chat_task_manager import ChatTask, ChatTaskManager
from datus.models.session_manager import SessionManager
from datus.utils.image_model_support import image_model_support


@pytest.fixture
def image_part():
    return {
        "type": "image",
        "payload": {
            "attachmentId": "a" * 32,
            "mimeType": "image/png",
            "width": 80,
            "height": 80,
            "createdAt": "2026-10-07T00:00:00Z",
        },
    }


@pytest.mark.parametrize("input_type", [StreamChatInput, InsertMessageInput])
def test_messages_take_precedence_and_support_image_only(input_type, image_part):
    extra = {"session_id": "chat_session_test"}
    request = input_type(message="legacy", messages=[image_part], **extra)
    assert request.message == ""
    assert request.messages[0].payload.attachmentId == "a" * 32
    assert input_type(message="legacy", **extra).message == "legacy"
    with pytest.raises(ValidationError):
        input_type(messages=[{"type": "markdown", "payload": {"content": "  "}}], **extra)
    with pytest.raises(ValidationError):
        input_type(messages=[], **extra)


def test_client_paths_and_base64_are_rejected(image_part):
    for field in ("path", "base64"):
        image_part["payload"][field] = "untrusted"
        with pytest.raises(ValidationError):
            StreamChatInput(messages=[image_part])
        del image_part["payload"][field]


@pytest.mark.asyncio
async def test_host_resolver_and_standalone_fallback(image_part):
    http = Request({"type": "http"})
    request = StreamChatInput(messages=[image_part])
    set_chat_message_resolver(None)
    with pytest.raises(HTTPException) as error:
        await prepare_chat_messages(http, request)
    assert error.value.status_code == 400

    async def resolve(http_request, contents):
        return "[Image #1] read_image /server/image.png", contents

    try:
        set_chat_message_resolver(resolve)
        await prepare_chat_messages(http, request)
        assert request.message.endswith("/server/image.png")
        assert request.messages[0].payload.attachmentId == "a" * 32
    finally:
        set_chat_message_resolver(None)
    text = StreamChatInput(messages=[{"type": "markdown", "payload": {"content": "hello"}}])
    await prepare_chat_messages(http, text)
    assert text.message == "hello"


@pytest.mark.asyncio
async def test_insert_echoes_structured_content(image_part):
    task = ChatTask(session_id="chat_session_test", asyncio_task=None)
    svc = SimpleNamespace(task_manager=SimpleNamespace(get_task=lambda _: task))

    async def resolve(http, contents):
        return "[Image #1] read_image /server/image.png", contents

    try:
        set_chat_message_resolver(resolve)
        result = await insert_message(
            InsertMessageInput(session_id=task.session_id, messages=[image_part]), svc, Request({"type": "http"})
        )
        assert result.success is True
        text = task.pending_input_queue.snapshot()[0]
        manager = ChatTaskManager()
        await manager._emit_user_insert_sse(task, text, 1)
        assert task.events[0].data.payload.content[0].model_dump() == image_part
    finally:
        set_chat_message_resolver(None)


def test_history_and_copy_preserve_original_contents(tmp_path, image_part):
    manager = SessionManager(session_dir=str(tmp_path))
    try:
        session_id = "chat_session_images"
        session = manager.get_session(session_id)
        text = "[Image #1] read_image /server/image.png"
        asyncio.run(
            session.add_items([{"role": "user", "content": f"<system_reminder>context</system_reminder>\n{text}"}])
        )
        manager.save_message_contents(session_id, {"submission": {"text": text, "parts": [image_part]}})
        assert manager.get_session_messages(session_id)[0]["message_contents"] == [image_part]
        copied = manager.copy_session(session_id, "chat")
        assert manager.get_session_messages(copied)[0]["message_contents"] == [image_part]
        # Model history remains intact; archival display must not rewrite it.
        with sqlite3.connect(tmp_path / f"{session_id}.db") as conn:
            raw = conn.execute("SELECT message_data FROM agent_messages").fetchone()[0]
            assert "/server/image.png" in raw
    finally:
        manager.close_all_sessions()


def test_capability_metadata_does_not_guess_unknown_models():
    assert image_model_support("openai/gpt-4o") is True
    assert image_model_support("gpt-5.4") is True
    assert image_model_support("deepseek-v4-pro") is False
    assert image_model_support("new-private-model") is None
    assert image_model_support("new-private-model", ["text", "image"]) is True
    assert image_model_support("new-private-model", ["text"]) is False


def test_insert_limits_the_total_joined_markdown_length():
    parts = [{"type": "markdown", "payload": {"content": "x" * 2000}}] * 2
    with pytest.raises(ValidationError, match="4000"):
        InsertMessageInput(session_id="chat_session_limits", messages=parts)
    parts[1] = {"type": "markdown", "payload": {"content": "x" * 1998}}
    assert len(InsertMessageInput(session_id="chat_session_limits", messages=parts).message) == 4000


@pytest.mark.asyncio
async def test_oversized_resolved_insert_never_enters_queue(image_part):
    task = ChatTask(session_id="chat_session_limits", asyncio_task=None)
    svc = SimpleNamespace(task_manager=SimpleNamespace(get_task=lambda _: task))

    async def resolve(http, contents):
        return "x" * 4001, contents

    try:
        set_chat_message_resolver(resolve)
        with pytest.raises(HTTPException) as error:
            await insert_message(
                InsertMessageInput(session_id=task.session_id, messages=[image_part]), svc, Request({"type": "http"})
            )
        assert error.value.status_code == 422
        assert len(task.pending_input_queue) == 0
        assert not task.message_contents
    finally:
        set_chat_message_resolver(None)


def test_duplicate_text_retains_distinct_parts_across_copy_rewind_and_clear(tmp_path, image_part):
    import copy

    first = [image_part]
    second_image = copy.deepcopy(image_part)
    second_image["payload"]["attachmentId"] = "b" * 32
    second = [second_image]
    manager = SessionManager(session_dir=str(tmp_path))
    task = ChatTask(session_id="chat_session_duplicates", asyncio_task=None)
    try:
        session = manager.get_session(task.session_id)
        task.register_message_contents("same", first)
        task.register_message_contents("same", second)
        asyncio.run(
            session.add_items(
                [
                    {"role": "user", "content": "same"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "same"},
                    {"role": "assistant", "content": "second answer"},
                ]
            )
        )
        manager.save_message_contents(task.session_id, task.message_contents)
        # A second best-effort persist must remain idempotent.
        manager.save_message_contents(task.session_id, task.message_contents)
        for session_id in (task.session_id, manager.copy_session(task.session_id, "chat")):
            users = [item for item in manager.get_session_messages(session_id) if item["role"] == "user"]
            assert [item["message_contents"] for item in users] == [first, second]
        rewound = manager.rewind_session(task.session_id, 1)
        with sqlite3.connect(tmp_path / f"{rewound}.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM chat_message_contents").fetchone()[0] == 1
        assert manager.get_session_messages(rewound)[0]["message_contents"] == first
        manager.clear_session(task.session_id)
        with sqlite3.connect(tmp_path / f"{task.session_id}.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM chat_message_contents").fetchone()[0] == 0
    finally:
        manager.close_all_sessions()


@pytest.mark.asyncio
async def test_duplicate_inserts_restore_by_submission_identity(image_part):
    from datus.cli.execution_state import InteractionBroker

    task = ChatTask(session_id="chat_session_duplicates", asyncio_task=None)
    first = task.register_message_contents("same", [image_part])
    second_parts = [{"type": "markdown", "payload": {"content": "same"}}]
    second = task.register_message_contents("same", second_parts)
    assert first.content_id != second.content_id
    broker = InteractionBroker()
    broker.emit_user_insert(second)
    action = broker._output_queue.get_nowait()
    assert (
        task.restore_message_contents(action.messages, action.action_id, action.input["message_content_id"])
        == second_parts
    )
    manager = ChatTaskManager()
    await manager._emit_user_insert_sse(task, first, 1)
    await manager._emit_user_insert_sse(task, second, 2)
    assert task.events[0].data.payload.content[0].model_dump() == image_part
    assert task.events[1].data.payload.content[0].model_dump() == second_parts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_run_persists_against_the_nodes_actual_session(real_agent_config, failed):
    from unittest.mock import AsyncMock

    class Node:
        session_id = "chat_session_copied"

        def get_node_name(self):
            return "chat"

        async def execute_stream_with_interactions(self, action_history_manager):
            if failed:
                raise RuntimeError("inference failed")
            if False:
                yield None

        async def get_last_turn_usage(self):
            return None

    manager = ChatTaskManager()
    manager._create_node = lambda *args, **kwargs: Node()
    manager._persist_message_contents = AsyncMock()
    task = ChatTask(session_id="chat_session_original", asyncio_task=None)
    await manager._run_loop(task, real_agent_config, StreamChatInput(message="hello"))
    assert manager._persist_message_contents.await_count >= 1
    assert all(call.args[2] == Node.session_id for call in manager._persist_message_contents.await_args_list)
