# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""WeCom AI Bot API-mode WebSocket adapter (no public callback required)."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from datus.gateway.channel.base import ChannelAdapter
from datus.gateway.models import InboundMessage, OutboundMessage
from datus.utils.loggings import get_logger

logger = get_logger(__name__)
WS_URL = "wss://openws.work.weixin.qq.com"
_STREAM_LIMIT = 20000  # Conservative UTF-8 byte limit; overflow is delivered separately.


def _chunks(text: str, limit: int = 4000):
    """Split by UTF-8 bytes without breaking characters."""
    data = text.encode("utf-8")
    while data:
        piece = data[:limit].decode("utf-8", errors="ignore")
        yield piece
        data = data[len(piece.encode("utf-8")) :]


@dataclass
class _Reply:
    req_id: str
    chat_id: str
    stream_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    content: str = ""
    expired: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class WeComAdapter(ChannelAdapter):
    """Run one authenticated connection on the host's asyncio loop.

    Callbacks are handled in separate tasks so slow agent turns never block
    receiving ACKs or heartbeat responses. Reply context is task-local: two
    simultaneous conversations cannot accidentally reply to each other's req_id.
    """

    def __init__(self, channel_id, config, bridge, channel_config=None):
        super().__init__(channel_id, config, bridge, channel_config)
        self._bot_id = config.get("bot_id", "")
        self._secret = config.get("secret", "")
        self._url = config.get("ws_url", WS_URL)
        self._heartbeat_interval = float(config.get("heartbeat_interval", 30))
        self._handshake_timeout = float(config.get("handshake_timeout", 10))
        self._request_timeout = float(config.get("request_timeout", 5))
        self._stream_timeout = float(config.get("stream_timeout", 300))
        if min(self._heartbeat_interval, self._handshake_timeout, self._request_timeout, self._stream_timeout) <= 0:
            raise ValueError("WeCom timeouts must be positive")
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._runner: Optional[asyncio.Task] = None
        self._taken_over = False
        self._pending: dict[str, asyncio.Future] = {}
        self._handlers: set[asyncio.Task] = set()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._reply: contextvars.ContextVar[Optional[_Reply]] = contextvars.ContextVar(
            f"wecom-reply-{channel_id}", default=None
        )

    @property
    def supports_threads(self) -> bool:
        return False

    @property
    def supports_streaming(self) -> bool:
        return True

    def is_configured(self) -> bool:
        return bool(self._bot_id and self._secret)

    async def start(self) -> None:
        if self._runner and not self._runner.done():
            return
        if not self.is_configured():
            raise ValueError("WeCom requires bot_id and secret")
        self._taken_over = False
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=self._handshake_timeout)
        )
        self._runner = asyncio.create_task(self._run(), name=f"wecom-{self.channel_id}")

    async def stop(self) -> None:
        if self._runner:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
            self._runner = None
        if self._session:
            await self._session.close()
            self._session = None

    async def _request(self, cmd: str, body=None, req_id: Optional[str] = None) -> dict:
        if self._taken_over or self._ws is None or self._ws.closed:
            raise ConnectionError("WeCom connection is unavailable")
        req_id = req_id or f"{cmd}_{uuid.uuid4().hex}"
        future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        frame = {"cmd": cmd, "headers": {"req_id": req_id}}
        if body is not None:
            frame["body"] = body
        try:
            await self._ws.send_json(frame)
            reply = await asyncio.wait_for(future, self._request_timeout)
            if reply.get("errcode") != 0:
                # Do not log platform payloads: they can contain user content or credentials.
                raise RuntimeError(f"WeCom {cmd} failed (code={reply.get('errcode')})")
            return reply
        finally:
            self._pending.pop(req_id, None)

    async def _receive(self) -> None:
        async for message in self._ws:
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                frame = json.loads(message.data)
                if not isinstance(frame, dict):
                    continue
                if frame.get("cmd") == "aibot_msg_callback":
                    task = asyncio.create_task(self._handle_message(frame))
                    self._handlers.add(task)
                    task.add_done_callback(self._handlers.discard)
                elif frame.get("cmd") == "aibot_event_callback":
                    if frame.get("body", {}).get("event", {}).get("eventtype") == "disconnected_event":
                        self._taken_over = True
                        logger.warning(
                            "WeCom channel %s received disconnected_event: another connection took over; "
                            "automatic reconnect disabled. Resolve duplicate bot connections before restarting.",
                            self.channel_id,
                        )
                        # Wake auth/heartbeat requests too: the runner may be waiting for an ACK.
                        for future in self._pending.values():
                            if not future.done():
                                future.set_exception(ConnectionError("WeCom connection taken over"))
                        return
                else:
                    future = self._pending.get(frame.get("headers", {}).get("req_id"))
                    if future is not None and not future.done():
                        future.set_result(frame)
            except (ValueError, TypeError, AttributeError):
                logger.warning("Ignoring malformed WeCom frame on %s", self.channel_id)
        raise ConnectionError("WeCom connection closed")

    async def _run(self) -> None:
        delay = 1
        while True:
            receiver = None
            try:
                # Bound connection establishment and upgrade headers, not the live socket.
                async with asyncio.timeout(self._handshake_timeout):
                    ws = await self._session.ws_connect(self._url, max_msg_size=1024 * 1024)
                async with ws:
                    self._ws = ws
                    receiver = asyncio.create_task(self._receive())
                    await self._request("aibot_subscribe", {"bot_id": self._bot_id, "secret": self._secret})
                    logger.info("WeCom channel %s authenticated", self.channel_id)
                    delay = 1
                    while True:
                        # Detect an early disconnect without waiting for the next heartbeat.
                        done, _ = await asyncio.wait({receiver}, timeout=self._heartbeat_interval)
                        if done:
                            await receiver
                            break
                        await self._request("ping")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._taken_over:
                    logger.warning(
                        "WeCom channel %s disconnected (%s); reconnecting", self.channel_id, type(exc).__name__
                    )
            finally:
                if receiver:
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(ConnectionError("WeCom disconnected"))
                handlers = list(self._handlers)
                for task in handlers:
                    task.cancel()
                await asyncio.gather(*handlers, return_exceptions=True)
                self._ws = None
            if self._taken_over:
                await self._session.close()
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)

    async def _stream(self, state: _Reply, finish: bool, content: Optional[str] = None) -> str:
        text = state.content if content is None else content
        text = text.encode("utf-8")[:_STREAM_LIMIT].decode("utf-8", errors="ignore")
        await self._request(
            "aibot_respond_msg",
            {"msgtype": "stream", "stream": {"id": state.stream_id, "finish": finish, "content": text}},
            state.req_id,
        )
        return text

    async def _expire_stream(self, state: _Reply) -> None:
        await asyncio.sleep(self._stream_timeout)
        # Commit the one-way transition before waiting for either the reply lock
        # or a close ACK. Callback completion may cancel this task at either await.
        state.expired = True
        async with state.lock:
            try:
                await self._stream(state, True, "Still working. The result will be sent as a separate message.")
            except Exception as exc:
                logger.warning(
                    "WeCom stream close failed on %s (%s); final result will use proactive delivery",
                    self.channel_id,
                    type(exc).__name__,
                )

    async def _handle_message(self, frame: dict) -> None:
        state = None
        token = None
        expiry = None
        try:
            body = frame.get("body", {})
            req_id = frame.get("headers", {}).get("req_id")
            msg_id = body.get("msgid")
            sender = body.get("from", {}).get("userid")
            group = body.get("chattype") == "group"
            chat_id = body.get("chatid") if group else sender
            if not req_id or not msg_id or not sender or not chat_id or msg_id in self._seen:
                return
            self._seen[msg_id] = None
            while len(self._seen) > 1000:
                self._seen.popitem(last=False)
            state = _Reply(req_id, chat_id)
            token = self._reply.set(state)
            # Reply before config/storage/model initialization can take longer than 5s.
            await self._stream(state, False, "Processing…")
            expiry = asyncio.create_task(self._expire_stream(state))
            if body.get("msgtype") != "text":
                state.content = "Please send a text message. Attachments are not supported on this channel yet."
                return
            text = body.get("text", {}).get("content", "").strip()
            if not text:
                state.content = "Please send a non-empty text message."
                return
            await self.dispatch_message(
                InboundMessage(
                    channel_id=self.channel_id,
                    sender_id=sender,
                    conversation_id=chat_id,
                    message_id=msg_id,
                    text=text,
                    chat_type="group" if group else "p2p",
                    # WeCom delivers group messages addressed to this bot; it has no native threads.
                    mentions_bot=group,
                    session_scope=sender if group else None,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("WeCom message processing failed on %s (%s)", self.channel_id, type(exc).__name__)
            if state:
                state.content = "The request could not be completed. Please try again."
        finally:
            if expiry:
                expiry.cancel()
                await asyncio.gather(expiry, return_exceptions=True)
            if state:
                with contextlib.suppress(Exception):
                    await self._finish(state)
            if token is not None:
                self._reply.reset(token)

    async def send_message(self, message: OutboundMessage) -> Optional[str]:
        state = self._reply.get()
        if state is None:
            raise RuntimeError("WeCom reply requires an active message callback")
        text = message.text
        if message.sql:
            text += f"\n\n```sql\n{message.sql}\n```"
        async with state.lock:
            state.content += text
            if not state.expired:
                await self._stream(state, False)
        # WeCom stream IDs are not reaction-addressable message IDs.
        return None

    async def _finish(self, state: _Reply) -> None:
        async with state.lock:
            content = state.content or "(No response generated)"
            if not state.expired:
                streamed = await self._stream(state, True, content)
                content = content[len(streamed) :]
            for chunk in _chunks(content):
                await self._request(
                    "aibot_send_msg",
                    {"chatid": state.chat_id, "msgtype": "markdown", "markdown": {"content": chunk}},
                )

    async def add_reaction(self, conversation_id, message_id, emoji, thread_id=None) -> None:
        """WeCom AI Bot has no equivalent reaction API."""

    async def remove_reaction(self, conversation_id, message_id, emoji, thread_id=None) -> None:
        """WeCom AI Bot has no equivalent reaction API."""
