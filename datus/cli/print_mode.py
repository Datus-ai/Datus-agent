# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Print mode runner for CLI --print flag.

Streams MessagePayload JSON lines to stdout and reads interaction responses from stdin.
"""

import asyncio
import json
import sys
import uuid

from pydantic import ValidationError

from datus.agent.node.node_factory import create_interactive_node, create_node_input
from datus.cli.autocomplete import AtReferenceCompleter
from datus.configuration.agent_config_loader import load_agent_config
from datus.schemas.action_content_builder import action_to_content, build_interaction_content, build_response_content
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.message_content import MessagePayload
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
        self.message_id = str(uuid.uuid4())

        # Database context from args
        self.catalog = getattr(args, "catalog", None)
        self.database = getattr(args, "database", None) or None
        self.db_schema = getattr(args, "schema", None)

    def run(self):
        node = create_interactive_node(self.subagent_name, self.agent_config, node_id_suffix="_print")
        if self.session_id:
            node.session_id = self.session_id

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
        async for action in node.execute_stream_with_interactions(self.actions):
            if action.role == ActionRole.INTERACTION and action.status == ActionStatus.PROCESSING:
                contents = build_interaction_content(action)
                self._write_payload(MessagePayload(message_id=self.message_id, role="assistant", content=contents))
                user_input = await asyncio.to_thread(self._read_interaction_input)
                await node.interaction_broker.submit(action.action_id, user_input)
                continue

            if (
                action.role == ActionRole.ASSISTANT
                and action.status == ActionStatus.SUCCESS
                and action.action_type
                and action.action_type.endswith("_response")
            ):
                contents = build_response_content(action)
                self._write_payload(MessagePayload(message_id=self.message_id, role="assistant", content=contents))
                continue

            contents = action_to_content(action)
            if contents:
                self._write_payload(MessagePayload(message_id=self.message_id, role="assistant", content=contents))

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
                if item.type == "interaction":
                    return item.payload.get("content", "")
            return ""
        except (json.JSONDecodeError, ValidationError):
            logger.warning("Failed to parse interaction input, returning raw line")
            return line.strip()
