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
        manager.save_message_contents(session_id, {text: [image_part]})
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
    assert image_model_support("deepseek-v4-pro") is False
    assert image_model_support("new-private-model") is None
    assert image_model_support("new-private-model", ["text", "image"]) is True
    assert image_model_support("new-private-model", ["text"]) is False
