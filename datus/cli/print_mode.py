# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Print mode runner for CLI --print flag.

Streams MessagePayload JSON lines to stdout and reads interaction responses from stdin.
"""

import asyncio
import json
import os
import select
import sys
import threading

from pydantic import ValidationError

from datus.agent.node.node_factory import create_interactive_node, create_node_input
from datus.cli.autocomplete import AtReferenceCompleter
from datus.configuration.agent_config_loader import load_agent_config
from datus.schemas.action_content_builder import action_to_content, build_interaction_content, build_response_content
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.message_content import MessageContent, MessagePayload
from datus.utils.async_utils import run_async
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class PrintModeRunner:
    """Run a single prompt in print mode, streaming JSON to stdout."""

    def __init__(self, args):
        self.agent_config = load_agent_config(**vars(args))
        self.at_completer = AtReferenceCompleter(self.agent_config)
        self.actions = ActionHistoryManager()
        self.message = args.print_mode
        self.session_id = getattr(args, "resume", None)
        self.subagent_name = getattr(args, "subagent", None) or None
        self.proxy_tool_patterns = getattr(args, "proxy_tools", None)
        self.scope = getattr(args, "session_scope", None)
        self.stream_thinking = getattr(args, "stream_thinking", False)

        # Datasource context from args
        self.catalog = getattr(args, "catalog", None)
        self.database = getattr(args, "datasource", None) or None
        self.db_schema = getattr(args, "schema", None)

    def run(self):
        if self.session_id:
            self._validate_and_resolve_session()

        node = create_interactive_node(self.subagent_name, self.agent_config, node_id_suffix="_print", scope=self.scope)
        if self.session_id:
            node.session_id = self.session_id

        if self.proxy_tool_patterns:
            from datus.tools.proxy.proxy_tool import apply_proxy_tools

            patterns = [p.strip() for p in self.proxy_tool_patterns.split(",")]
            apply_proxy_tools(node, patterns)

        at_tables, at_metrics, at_sqls = self.at_completer.parse_at_context(self.message)
        node_input = create_node_input(
            user_message=self.message,
            node=node,
            catalog=self.catalog,
            database=self.database,
            db_schema=self.db_schema,
            at_tables=at_tables,
            at_metrics=at_metrics,
            at_sqls=at_sqls,
        )
        node.input = node_input
        run_async(self._stream_chat(node))

    async def _stream_chat(self, node):
        dispatch_task = None
        self._stdin_stop_event = threading.Event()
        if self.proxy_tool_patterns:
            dispatch_task = asyncio.create_task(self._stdin_dispatch_loop(node))

        try:
            async for action in node.execute_stream_with_interactions(self.actions):
                if action.role == ActionRole.INTERACTION and action.status == ActionStatus.PROCESSING:
                    contents = build_interaction_content(action)
                    self._write_payload(
                        MessagePayload(
                            message_id=action.action_id,
                            role="assistant",
                            content=contents,
                            depth=action.depth,
                            parent_action_id=action.parent_action_id,
                        )
                    )
                    if not self.proxy_tool_patterns:
                        user_input = await asyncio.to_thread(self._read_interaction_input)
                        await node.interaction_broker.submit(action.action_id, user_input)
                    continue

                # Streaming thinking deltas: emit only when --stream is enabled
                if action.action_type == "thinking_delta":
                    if self.stream_thinking:
                        output = action.output if isinstance(action.output, dict) else {}
                        delta_text = output.get("delta", "")
                        contents = [MessageContent(type="thinking-delta", payload={"delta": delta_text})]
                        self._write_payload(
                            MessagePayload(
                                message_id=action.action_id,
                                role="assistant",
                                content=contents,
                                depth=action.depth,
                                parent_action_id=action.parent_action_id,
                            )
                        )
                    continue

                if (
                    action.role == ActionRole.ASSISTANT
                    and action.status == ActionStatus.SUCCESS
                    and action.action_type
                    and action.action_type.endswith("_response")
                ):
                    contents = build_response_content(action)
                    self._write_payload(
                        MessagePayload(
                            message_id=action.action_id,
                            role="assistant",
                            content=contents,
                            depth=action.depth,
                            parent_action_id=action.parent_action_id,
                        )
                    )
                    continue

                contents = action_to_content(action)
                if contents:
                    self._write_payload(
                        MessagePayload(
                            message_id=action.action_id,
                            role="assistant",
                            content=contents,
                            depth=action.depth,
                            parent_action_id=action.parent_action_id,
                        )
                    )
        finally:
            self._stdin_stop_event.set()
            if self.proxy_tool_patterns:
                node.tool_channel.cancel_all("stream ended")
            if dispatch_task and not dispatch_task.done():
                dispatch_task.cancel()
                try:
                    await dispatch_task
                except asyncio.CancelledError:
                    pass

    async def _stdin_dispatch_loop(self, node):
        """Read stdin lines and dispatch call-tool-result / user-interaction to the node."""
        stop_event = self._stdin_stop_event
        loop = asyncio.get_running_loop()

        while not stop_event.is_set():
            line = await loop.run_in_executor(None, self._read_stdin_line, stop_event)
            if line is None:
                node.tool_channel.cancel_all("stdin EOF")
                break
            if not line.strip():
                continue
            try:
                data = MessagePayload.model_validate_json(line.strip())
                for item in data.content:
                    if item.type == "call-tool-result":
                        call_id = item.payload.get("callToolId", "")
                        result = item.payload.get("result")
                        if call_id:
                            await node.tool_channel.publish(call_id, result)
                    elif item.type == "user-interaction":
                        content = item.payload.get("content", "")
                        await node.interaction_broker.submit(data.message_id, content)
            except (json.JSONDecodeError, ValidationError):
                logger.warning("Failed to parse stdin in proxy mode")

    @staticmethod
    def _read_stdin_line(stop_event: threading.Event) -> str | None:
        """Read one line from stdin, checking stop_event periodically.

        Returns None on EOF or when stop_event is set.
        Uses ``select`` on Unix to avoid blocking indefinitely.
        On Windows falls back to a short polling loop.
        """
        fd = sys.stdin.fileno()
        if sys.platform == "win32":
            str_buf: list[str] = []
            while not stop_event.is_set():
                if sys.stdin.readable():
                    ch = sys.stdin.read(1)
                    if not ch:
                        return None
                    if ch == "\n":
                        return "".join(str_buf)
                    str_buf.append(ch)
                else:
                    stop_event.wait(0.05)
        else:
            buf: list[bytes] = []
            while not stop_event.is_set():
                ready, _, _ = select.select([fd], [], [], 0.1)
                if ready:
                    chunk = os.read(fd, 4096)
                    if not chunk:
                        return None
                    buf.append(chunk)
                    if b"\n" in chunk:
                        data = b"".join(buf)
                        line, _, remainder = data.partition(b"\n")
                        # Put back any bytes after the newline (shouldn't happen
                        # normally since each JSON message is one line, but be safe).
                        if remainder:
                            buf[:] = [remainder]
                        return line.decode("utf-8", errors="replace")
        return None

    def _validate_and_resolve_session(self):
        """Validate session exists or create a new one with the given session_id."""
        from datus.models.session_manager import SessionManager

        session_manager = SessionManager(session_dir=self.agent_config.session_dir, scope=self.scope)
        if not session_manager.session_exists(self.session_id):
            logger.info("Session '%s' not found, will create a new session with this id.", self.session_id)
            return

    def _write_payload(self, payload: MessagePayload):
        sys.stdout.write(payload.model_dump_json() + "\n")
        sys.stdout.flush()

    def _read_interaction_input(self) -> str:
        line = sys.stdin.readline()
        if not line.strip():
            return ""
        try:
            data = MessagePayload.model_validate_json(line.strip())
            for item in data.content:
                if item.type == "user-interaction":
                    return item.payload.get("content", "")
            return ""
        except (json.JSONDecodeError, ValidationError):
            logger.warning("Failed to parse interaction input, returning raw line")
            return line.strip()
