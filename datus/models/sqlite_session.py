# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Repository-owned atomic history replacement for the agents SQLite session."""

import asyncio
import json
from typing import Any

from agents.extensions.memory import AdvancedSQLiteSession

from datus.utils.message_utils import extract_user_input, is_compact_resume_text

CONTEXT_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS context_occupancy (
    session_id TEXT PRIMARY KEY,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    context_length INTEGER NOT NULL DEFAULT 0,
    valid INTEGER NOT NULL DEFAULT 0
)
"""


class DatusSQLiteSession(AdvancedSQLiteSession):
    def _is_user_message(self, item: dict[str, Any]) -> bool:
        if not super()._is_user_message(item):
            return False
        content = item.get("content")
        if is_compact_resume_text(extract_user_input(content)):
            return False
        if (
            isinstance(content, list)
            and content
            and all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
        ):
            return False
        return True

    def skip_persisted_input_once(self, items: list[dict[str, Any]]) -> None:
        """Acknowledge input already stored by a first-request rewrite.

        The SDK appends its original input after the request filter. This guard
        applies only to that next append and is reset when a new run prepares input.
        """
        self._persisted_input = list(items)

    async def add_items(self, items: list[dict[str, Any]]) -> None:
        persisted = getattr(self, "_persisted_input", [])
        self._persisted_input = []
        if persisted and items[: len(persisted)] == persisted:
            items = items[len(persisted) :]
        await super().add_items(items)

    async def replace_items(self, items: list[dict[str, Any]], *, pending_user_turns: int = 0) -> None:
        """Replace history and invalidate its measurement in one transaction.

        Usage records and monotonic turn numbers survive compaction. A failed
        message or metadata insert rolls back the entire replacement.
        """
        # Serialize before deleting anything. Reuse the SDK's message dialect
        # classifiers, but do not call add_items: it commits messages separately
        # from their structure metadata.
        prepared = [
            (
                json.dumps(item),
                self._classify_message_type(item),
                self._extract_tool_name(item),
                self._is_user_message(item),
            )
            for item in items
        ]

        def replace() -> None:
            conn = self._get_connection()
            with self._lock:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    conn.execute(CONTEXT_STATE_TABLE)
                    seq, turn, branch_turn = conn.execute(
                        "SELECT COALESCE(MAX(sequence_number), 0), COALESCE(MAX(user_turn_number), 0), "
                        "COALESCE(MAX(branch_turn_number), 0) FROM message_structure WHERE session_id = ?",
                        (self.session_id,),
                    ).fetchone()
                    usage_turn = conn.execute(
                        "SELECT COALESCE(MAX(user_turn_number), 0) FROM turn_usage WHERE session_id = ?",
                        (self.session_id,),
                    ).fetchone()[0]
                    users = sum(row[3] for row in prepared)
                    turn = max(0, max(turn, usage_turn) + pending_user_turns - users)
                    branch_turn = max(0, branch_turn + pending_user_turns - users)
                    conn.execute("DELETE FROM message_structure WHERE session_id = ?", (self.session_id,))
                    conn.execute(f"DELETE FROM {self.messages_table} WHERE session_id = ?", (self.session_id,))
                    conn.execute(
                        f"INSERT OR IGNORE INTO {self.sessions_table} (session_id) VALUES (?)", (self.session_id,)
                    )
                    for data, kind, tool, is_user in prepared:
                        seq += 1
                        turn += int(is_user)
                        branch_turn += int(is_user)
                        cursor = conn.execute(
                            f"INSERT INTO {self.messages_table} (session_id, message_data) VALUES (?, ?)",
                            (self.session_id, data),
                        )
                        conn.execute(
                            "INSERT INTO message_structure (session_id, message_id, branch_id, message_type, "
                            "sequence_number, user_turn_number, branch_turn_number, tool_name) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                self.session_id,
                                cursor.lastrowid,
                                self._current_branch_id,
                                kind,
                                seq,
                                turn,
                                branch_turn,
                                tool,
                            ),
                        )
                    conn.execute(
                        "INSERT INTO context_occupancy (session_id) VALUES (?) "
                        "ON CONFLICT(session_id) DO UPDATE SET input_tokens = 0, valid = 0",
                        (self.session_id,),
                    )
                    if conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'running_turn_usage'"
                    ).fetchone():
                        row = conn.execute(
                            "SELECT cumulative_json FROM running_turn_usage WHERE session_id = ?",
                            (self.session_id,),
                        ).fetchone()
                        if row:
                            cumulative = json.loads(row[0])
                            cumulative.update(
                                last_call_input_tokens=0, context_usage_ratio=0.0, context_usage_valid=False
                            )
                            conn.execute(
                                "UPDATE running_turn_usage SET cumulative_json = ? WHERE session_id = ?",
                                (json.dumps(cumulative), self.session_id),
                            )
                    conn.execute(
                        f"UPDATE {self.sessions_table} SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                        (self.session_id,),
                    )
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise

        # Do not leave a background commit racing with ESC rollback on cancellation.
        task = asyncio.create_task(asyncio.to_thread(replace))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
