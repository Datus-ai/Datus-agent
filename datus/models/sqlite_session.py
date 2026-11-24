# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Repository-owned atomic history replacement for the agents SQLite session."""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from agents.extensions.memory import AdvancedSQLiteSession

from datus.models.history_archive import HISTORY_VERSION_KEY, HistoryArchive
from datus.utils.loggings import get_logger
from datus.utils.message_utils import extract_user_input, is_compact_resume_text

logger = get_logger(__name__)

# One row per (session, key), holding everything about a session that is not
# a message or a usage record. Values are text; each key documents its own
# shape below.
SESSION_META_TABLE = """
CREATE TABLE IF NOT EXISTS session_meta (
    session_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    PRIMARY KEY (session_id, key)
)
"""

#: Label a session is listed under. Holds the first user message.
SESSION_TITLE_KEY = "title"

#: Plan-mode flags, stored as one JSON object so the three fields that are
#: always written together cannot land out of step with each other.
SESSION_PLAN_MODE_KEY = "plan_mode"

#: Context-window occupancy of the last model call, as a decimal integer.
#: Unlike its neighbours this one is rewritten on every model response and
#: reset to ``"0"`` by any history rewrite — the reading describes the history
#: that was just deleted, so it cannot outlive it.
SESSION_CONTEXT_USED_KEY = "context_used"

#: Cap on the stored title. It labels a list row, it is not a document.
MAX_SESSION_TITLE_CHARS = 500

#: Metadata a derived session (copy / rewind) inherits. Only the title
#: survives a history rewrite: ``context_used`` describes the exact history the
#: derivation just changed, and ``plan_mode`` is a live user toggle that the
#: derived session's own user has not asked for.
DERIVED_SESSION_META_KEYS = (SESSION_TITLE_KEY,)

#: Invalidate the occupancy reading of a session whose history was rewritten.
#: Shared by every rewrite path so they cannot drift apart — the atomicity
#: argument for ``replace_items`` and ``rollback_turn`` rests on them writing
#: the same row the same way.
CONTEXT_USED_RESET_SQL = (
    "INSERT INTO session_meta (session_id, key, value) VALUES (?, ?, '0') "
    "ON CONFLICT(session_id, key) DO UPDATE SET value = '0'"
)


class DatusSQLiteSession(AdvancedSQLiteSession):
    def __init__(self, *, archive_history: bool = True, **kwargs: Any):
        self._enable_history_archive = archive_history
        super().__init__(**kwargs)

    def _init_db_for_connection(self, conn: sqlite3.Connection) -> None:
        if not self._enable_history_archive or self._is_memory_db:
            super()._init_db_for_connection(conn)
            return
        # Serialize creation across replicas. Existing files with the SDK's
        # schema keep their original history mode, even when empty.
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name IN (?,?)",
                (self.sessions_table, self.messages_table),
            ).fetchone()
            if not existing:
                conn.execute(SESSION_META_TABLE)
                conn.execute("INSERT INTO session_meta VALUES (?,?,?)", (self.session_id, HISTORY_VERSION_KEY, "1"))
            super()._init_db_for_connection(conn)

    def track_model_input(self, items: list[dict[str, Any]]) -> None:
        self._pending_model_input = list(items)

    def pending_model_input(self) -> list[dict[str, Any]]:
        return getattr(self, "_pending_model_input", [])

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
        appended = items
        if persisted and items[: len(persisted)] == persisted:
            appended = items[len(persisted) :]
        await super().add_items(appended)
        pending = self.pending_model_input()
        if pending and items[: len(pending)] == pending:
            self.track_model_input([])
        # After the write, so the scan below sees this batch too: when
        # compaction ran on the very first request, the opening message is the
        # part skipped here and only the stored history still holds it.
        await self._record_title_once()

    async def _record_title_once(self) -> None:
        """Persist the session's opening user message as its durable title.

        Recorded as messages arrive rather than salvaged before a compact: the
        row lives in ``session_meta``, which a history rewrite never touches,
        so the label survives any number of compacts without a caller having to
        race the clear. An existing row always wins, so a later message can
        never retitle the chat.

        Best-effort — a title is a list label, and losing one must never fail
        the message write that just succeeded.
        """
        if getattr(self, "_title_recorded", False):
            return

        def record() -> bool:
            """Return whether the session now has a title, so the guard latches.

            A batch of purely assistant messages leaves an untitled session
            untitled; latching on that would stop the recorder before the
            opening message ever arrives.
            """
            conn = self._get_connection()
            with self._lock:
                conn.execute(SESSION_META_TABLE)
                titled = bool(
                    conn.execute(
                        "SELECT 1 FROM session_meta WHERE session_id = ? AND key = ?",
                        (self.session_id, SESSION_TITLE_KEY),
                    ).fetchone()
                )
                if not titled:
                    title = self._opening_user_message(conn)
                    if title:
                        conn.execute(
                            "INSERT OR IGNORE INTO session_meta (session_id, key, value) VALUES (?, ?, ?)",
                            (self.session_id, SESSION_TITLE_KEY, title[:MAX_SESSION_TITLE_CHARS]),
                        )
                        titled = True
                conn.commit()
                return titled

        try:
            self._title_recorded = await asyncio.to_thread(record)
        except Exception as exc:  # noqa: BLE001 — see docstring
            logger.warning("Could not record the title of session %s: %s", self.session_id, exc)

    def _opening_user_message(self, conn: Any) -> str:
        """Return the session's own opening message, not this batch's first one.

        A session that predates ``session_meta`` has no title row but does have
        history, so naming it from the incoming batch would rename the chat to
        whatever the user says next — the exact bug the stored title exists to
        prevent. Reading the earliest stored user message names old and new
        sessions the same way, and yields what ``get_session_info``'s scan
        would have shown.

        Runs at most once per session: the caller only reaches here while no
        title row exists, and the row check short-circuits every later call.
        """
        for (data,) in conn.execute(
            f"SELECT message_data FROM {self.messages_table} WHERE session_id = ? ORDER BY id",
            (self.session_id,),
        ):
            try:
                item = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(item, dict) and self._is_user_message(item):
                title = extract_user_input(item.get("content", "")).strip()
                if title:
                    return title
        return ""

    def forget_recorded_title(self) -> None:
        """Let the next user message name the session again.

        Called when the stored title is dropped (``/clear``). Without it the
        write-once guard above would keep short-circuiting and the restarted
        conversation would stay unnamed.
        """
        self._title_recorded = False

    async def clear_session(self) -> None:
        """Clear model and archived messages under the same transaction."""

        def clear() -> None:
            with self._locked_connection() as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                HistoryArchive(conn, Path(self.db_path), self.session_id).truncate(0)
                conn.execute(f"DELETE FROM {self.messages_table} WHERE session_id=?", (self.session_id,))
                conn.execute(f"DELETE FROM {self.sessions_table} WHERE session_id=?", (self.session_id,))

        await asyncio.to_thread(clear)

    async def replace_items(
        self,
        items: list[dict[str, Any]],
        *,
        pending_user_turns: int = 0,
        history_pending: list[dict[str, Any]] | None = None,
    ) -> None:
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
                    conn.execute(SESSION_META_TABLE)
                    seq, turn, branch_turn = conn.execute(
                        "SELECT COALESCE(MAX(sequence_number), 0), COALESCE(MAX(user_turn_number), 0), "
                        "COALESCE(MAX(branch_turn_number), 0) FROM message_structure WHERE session_id = ?",
                        (self.session_id,),
                    ).fetchone()
                    usage_turn = conn.execute(
                        "SELECT COALESCE(MAX(user_turn_number), 0) FROM turn_usage WHERE session_id = ?",
                        (self.session_id,),
                    ).fetchone()[0]
                    conn.execute(
                        f"INSERT OR IGNORE INTO {self.sessions_table} (session_id) VALUES (?)", (self.session_id,)
                    )
                    archive = HistoryArchive(conn, Path(self.db_path), self.session_id)
                    if archive.hidden_through is not None:
                        archive_rows = list(
                            conn.execute(
                                f"SELECT m.id,m.message_data,m.created_at,s.user_turn_number FROM {self.messages_table} m "
                                "LEFT JOIN message_structure s ON s.session_id=m.session_id AND s.message_id=m.id "
                                "WHERE m.session_id=? AND m.id>? ORDER BY m.id",
                                (self.session_id, archive.hidden_through),
                            )
                        )
                        # Allocate real SQLite IDs for uncommitted model input before
                        # installing the replacement. The IDs never get reused.
                        pending_turn = max(turn, usage_turn)
                        pending = self.pending_model_input() if history_pending is None else history_pending
                        for item in pending:
                            pending_turn += int(self._is_user_message(item))
                            row_id, raw, timestamp = conn.execute(
                                f"INSERT INTO {self.messages_table}(session_id,message_data) VALUES (?,?) "
                                "RETURNING id,message_data,created_at",
                                (self.session_id, json.dumps(item)),
                            ).fetchone()
                            archive_rows.append((row_id, raw, timestamp, pending_turn))
                        pending_user_count = max(0, pending_turn - max(turn, usage_turn))
                    else:
                        pending_user_count = pending_user_turns
                    users = sum(row[3] for row in prepared)
                    turn = max(0, max(turn, usage_turn) + pending_user_count - users)
                    branch_turn = max(0, branch_turn + pending_user_count - users)
                    conn.execute("DELETE FROM message_structure WHERE session_id = ?", (self.session_id,))
                    conn.execute(f"DELETE FROM {self.messages_table} WHERE session_id = ?", (self.session_id,))
                    last_id = archive_rows[-1][0] if archive.hidden_through is not None and archive_rows else 0
                    for data, kind, tool, is_user in prepared:
                        seq += 1
                        turn += int(is_user)
                        branch_turn += int(is_user)
                        cursor = conn.execute(
                            f"INSERT INTO {self.messages_table} (session_id, message_data) VALUES (?, ?)",
                            (self.session_id, data),
                        )
                        last_id = cursor.lastrowid
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
                    if archive.hidden_through is not None:
                        archive.write_segment(archive_rows, max(last_id, archive.hidden_through))
                    # The occupancy reading describes the history just deleted.
                    # Zeroed in this same transaction so the compaction gate
                    # and the status bar can never see it outlive its subject.
                    conn.execute(CONTEXT_USED_RESET_SQL, (self.session_id, SESSION_CONTEXT_USED_KEY))
                    if conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'running_turn_usage'"
                    ).fetchone():
                        row = conn.execute(
                            "SELECT cumulative_json FROM running_turn_usage WHERE session_id = ?",
                            (self.session_id,),
                        ).fetchone()
                        if row:
                            # An unreadable snapshot must not abort the rewrite:
                            # the ``except BaseException`` below would roll the
                            # message replacement back too and leave the
                            # oversized history in place. Treat it as absent,
                            # matching ``SessionManager._read_running_turn_usage``.
                            try:
                                cumulative = json.loads(row[0]) if row[0] else {}
                            except (json.JSONDecodeError, TypeError):
                                cumulative = {}
                            if not isinstance(cumulative, dict):
                                cumulative = {}
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
            self.track_model_input([])
            raise
        self.track_model_input([])
