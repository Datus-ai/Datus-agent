# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Protocol-level tests use a local WebSocket peer, never the WeCom service."""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiohttp import web

from datus.gateway.adapters.wecom import WeComAdapter, _chunks
from datus.gateway.bridge import ChannelBridge
from datus.gateway.models import OutboundMessage


def callback(msgid="m1", sender="user1", group=False, text="hello", req_id=None):
    return {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": req_id or f"req-{msgid}"},
        "body": {
            "msgid": msgid,
            "chattype": "group" if group else "single",
            "chatid": "group1",
            "from": {"userid": sender},
            "msgtype": "text",
            "text": {"content": text},
        },
    }


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest_asyncio.fixture
async def peer():
    frames = []
    sockets = []
    ack = {"auth_error": False, "ping": True, "stream_finish": "ok"}
    closed_streams = set()
    stream_close_seen = asyncio.Event()

    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sockets.append(ws)
        async for message in ws:
            frame = message.json()
            frames.append(frame)
            if frame["cmd"] == "ping" and not ack["ping"]:
                continue
            errcode = 40014 if ack["auth_error"] and frame["cmd"] == "aibot_subscribe" else 0
            if frame["cmd"] == "aibot_respond_msg":
                req_id = frame["headers"]["req_id"]
                if req_id in closed_streams:
                    errcode = 40001
                if frame["body"]["stream"]["finish"]:
                    closed_streams.add(req_id)
                    stream_close_seen.set()
                    if ack["stream_finish"] == "drop":
                        continue
                    if ack["stream_finish"] == "reject":
                        errcode = 40001
            await ws.send_json(
                {
                    "headers": frame["headers"],
                    "errcode": errcode,
                }
            )
        return ws

    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield SimpleNamespace(
        url=f"http://127.0.0.1:{port}", sockets=sockets, frames=frames, ack=ack, stream_close_seen=stream_close_seen
    )
    for ws in sockets:
        await ws.close()
    await runner.cleanup()


@contextlib.asynccontextmanager
async def running(peer, bridge=None, **options):
    bridge = bridge or SimpleNamespace(handle_message=AsyncMock())
    adapter = WeComAdapter(
        "wecom",
        {"bot_id": "bot", "secret": "test-secret", "ws_url": peer.url, "request_timeout": 0.2, **options},
        bridge,
    )
    await adapter.start()
    await until(lambda: peer.frames and not adapter._pending)
    try:
        yield adapter
    finally:
        await adapter.stop()
    assert not adapter._handlers
    assert not adapter._pending
    assert adapter._session is None


@pytest.mark.asyncio
async def test_streaming_ack_dedup_and_group_session_mapping(peer):
    received = []

    async def handle(msg, adapter, config):
        received.append(msg)
        await adapter.send_message(
            OutboundMessage(channel_id="wecom", conversation_id=msg.conversation_id, text="first")
        )
        await adapter.send_message(
            OutboundMessage(channel_id="wecom", conversation_id=msg.conversation_id, text=" second")
        )

    async with running(peer, SimpleNamespace(handle_message=handle)):
        frame = callback(group=True)
        await peer.sockets[-1].send_json(frame)
        await until(lambda: any(f.get("body", {}).get("stream", {}).get("finish") for f in peer.frames))
        await peer.sockets[-1].send_json(frame)
        await asyncio.sleep(0.03)
        assert len(received) == 1
        msg = received[0]
        assert (msg.conversation_id, msg.chat_type, msg.session_scope, msg.thread_id) == (
            "group1",
            "group",
            "user1",
            None,
        )
        assert msg.mentions_bot
        stream = [f for f in peer.frames if f["cmd"] == "aibot_respond_msg"]
        assert stream[0]["body"]["stream"]["content"] == "Processing…"
        assert stream[-1]["body"]["stream"]["content"] == "first second"
        assert {f["headers"]["req_id"] for f in stream} == {"req-m1"}
        assert len({f["body"]["stream"]["id"] for f in stream}) == 1
        bridge = ChannelBridge(SimpleNamespace(), None)
        assert bridge.build_session_id(msg) == bridge.build_session_id(msg.model_copy(update={"message_id": "m2"}))
        assert bridge.build_session_id(msg) != bridge.build_session_id(
            msg.model_copy(update={"session_scope": "user2"})
        )


@pytest.mark.asyncio
async def test_concurrent_callbacks_keep_reply_context_and_do_not_block_heartbeats(peer):
    release = asyncio.Event()
    received = []

    async def handle(msg, adapter, config):
        received.append(msg)
        await release.wait()
        await adapter.send_message(
            OutboundMessage(channel_id="wecom", conversation_id=msg.conversation_id, text=msg.sender_id)
        )

    async with running(peer, SimpleNamespace(handle_message=handle), heartbeat_interval=0.03):
        for user in ["u1", "u2"]:
            await peer.sockets[-1].send_json(callback(user, user))
        await until(lambda: len(received) == 2 and any(f["cmd"] == "ping" for f in peer.frames))
        release.set()
        await until(lambda: sum(bool(f.get("body", {}).get("stream", {}).get("finish")) for f in peer.frames) == 2)
        finals = {
            f["headers"]["req_id"]: f["body"]["stream"]["content"]
            for f in peer.frames
            if f.get("body", {}).get("stream", {}).get("finish")
        }
        assert finals == {"req-u1": "u1", "req-u2": "u2"}


@pytest.mark.asyncio
async def test_expired_turn_finishes_placeholder_then_pushes_result(peer):
    async def handle(msg, adapter, config):
        await asyncio.sleep(0.06)
        await adapter.send_message(
            OutboundMessage(channel_id="wecom", conversation_id=msg.conversation_id, text="final result")
        )

    async with running(peer, SimpleNamespace(handle_message=handle), stream_timeout=0.03):
        await peer.sockets[-1].send_json(callback())
        await until(lambda: any(f["cmd"] == "aibot_send_msg" for f in peer.frames))
        pushed = next(f["body"] for f in peer.frames if f["cmd"] == "aibot_send_msg")
        assert pushed == {"chatid": "user1", "msgtype": "markdown", "markdown": {"content": "final result"}}
        assert sum(bool(f.get("body", {}).get("stream", {}).get("finish")) for f in peer.frames) == 1


@pytest.mark.asyncio
async def test_disconnect_reauthenticates_and_shutdown_cancels_active_callback(peer):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def handle(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async with running(peer, SimpleNamespace(handle_message=handle)) as adapter:
        await peer.sockets[-1].close()
        await until(lambda: len(peer.sockets) == 2 and sum(f["cmd"] == "aibot_subscribe" for f in peer.frames) == 2)
        await peer.sockets[-1].send_json(callback())
        await asyncio.wait_for(entered.wait(), 2)
        await adapter.stop()
        assert cancelled.is_set()


@pytest.mark.asyncio
async def test_unsupported_attachment_is_explained_and_never_runs_agent(peer):
    bridge = SimpleNamespace(handle_message=AsyncMock())
    async with running(peer, bridge):
        frame = callback()
        frame["body"]["msgtype"] = "image"
        await peer.sockets[-1].send_json(frame)
        await until(lambda: any(f.get("body", {}).get("stream", {}).get("finish") for f in peer.frames))
        bridge.handle_message.assert_not_called()
        assert "text message" in peer.frames[-1]["body"]["stream"]["content"]


@pytest.mark.asyncio
async def test_missing_heartbeat_ack_reconnects(peer):
    async with running(peer, heartbeat_interval=0.02):
        peer.ack["ping"] = False
        await until(lambda: len(peer.sockets) >= 2)
        assert sum(f["cmd"] == "aibot_subscribe" for f in peer.frames) >= 2


def test_utf8_chunks_preserve_all_characters():
    text = "查询企业微信" * 5000
    chunks = list(_chunks(text))
    assert "".join(chunks) == text
    assert all(len(chunk.encode("utf-8")) <= 4000 for chunk in chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize("close_ack", ["drop", "reject"])
@pytest.mark.parametrize("completion", ["after_close_failure", "during_close"])
async def test_stream_close_failure_cannot_discard_the_agent_result(peer, close_ack, completion):
    peer.ack["stream_finish"] = close_ack
    expected = "partial result"
    if completion == "after_close_failure":
        expected += " and final result"

    async def handle(msg, adapter, config):
        await adapter.send_message(
            OutboundMessage(channel_id="wecom", conversation_id=msg.conversation_id, text="partial result")
        )
        await peer.stream_close_seen.wait()
        if completion == "after_close_failure":
            # Let the dropped ACK time out before the agent produces its final output.
            await asyncio.sleep(0.25)
            await adapter.send_message(
                OutboundMessage(channel_id="wecom", conversation_id=msg.conversation_id, text=" and final result")
            )
        # Returning while close ACK is pending cancels the stream-expiry task.

    async with running(peer, SimpleNamespace(handle_message=handle), stream_timeout=0.03) as adapter:
        await peer.sockets[-1].send_json(callback())
        await until(lambda: any(f["cmd"] == "aibot_send_msg" for f in peer.frames))
        await until(lambda: not adapter._handlers)
        pushed = [f["body"] for f in peer.frames if f["cmd"] == "aibot_send_msg"]
        assert pushed == [{"chatid": "user1", "msgtype": "markdown", "markdown": {"content": expected}}]
        close_index = next(i for i, f in enumerate(peer.frames) if f.get("body", {}).get("stream", {}).get("finish"))
        assert all(f["cmd"] != "aibot_respond_msg" for f in peer.frames[close_index + 1 :])


@pytest.mark.asyncio
@pytest.mark.parametrize("server_closes_socket", [False, True])
async def test_connection_takeover_stops_reconnecting_and_cancels_callbacks(peer, server_closes_socket):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def handle(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async with running(peer, SimpleNamespace(handle_message=handle), heartbeat_interval=0.03) as adapter:
        ws = peer.sockets[-1]
        await ws.send_json(callback())
        await asyncio.wait_for(entered.wait(), 2)
        await ws.send_json(
            {
                "cmd": "aibot_event_callback",
                "headers": {"req_id": "takeover-event"},
                "body": {"msgtype": "event", "event": {"eventtype": "disconnected_event"}},
            }
        )
        if server_closes_socket:
            await ws.close()
        await until(lambda: adapter._runner.done())
        await asyncio.wait_for(cancelled.wait(), 1)
        await until(lambda: ws.closed)
        # A normal network disconnect reconnects after one second; takeover must not.
        await asyncio.sleep(1.1)
        assert len(peer.sockets) == 1
        assert sum(f["cmd"] == "aibot_subscribe" for f in peer.frames) == 1
        assert not adapter._pending
        assert not adapter._handlers
