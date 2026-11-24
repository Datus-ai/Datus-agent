# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Immutable JSONL segments for the user-visible history of a compacted session."""

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datus.utils.message_utils import extract_user_input, is_compact_resume_text

HISTORY_VERSION_KEY = "chat_history_version"
HISTORY_REWRITE_KEY = "chat_history_hidden_through"
_SEGMENT_NAME = re.compile(r"chat-[0-9a-f]{32}\.jsonl")
_SEGMENTS = """
CREATE TABLE IF NOT EXISTS chat_history_segments (
    session_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    first_id INTEGER NOT NULL,
    last_id INTEGER NOT NULL,
    anchors TEXT NOT NULL,
    PRIMARY KEY (session_id, first_id)
)
"""
HistoryRow = tuple[int, str, str, int | None]
# Message ID, cursor digest, display preview, and byte offset in the segment.
Anchor = tuple[int, str, str, int]


def user_prompt(item: dict[str, Any]) -> str | None:
    if item.get("role") != "user":
        return None
    content = item.get("content")
    text = extract_user_input(content)
    if is_compact_resume_text(text):
        return None
    if isinstance(content, list):
        types = {block.get("type") for block in content if isinstance(block, dict)}
        if "tool_result" in types and not types.intersection(("text", "input_text", "output_text")):
            return None
    return text


def prompt_anchor(row_id: int, raw: str, offset: int = 0) -> Anchor | None:
    prompt = user_prompt(json.loads(raw))
    if prompt is None:
        return None
    return row_id, hashlib.sha256(raw.encode()).hexdigest(), prompt[:160], offset


class HistoryArchive:
    """Read a session's archive and live rows under the caller's SQLite snapshot."""

    def __init__(self, conn: sqlite3.Connection, db_path: Path, session_id: str):
        self.conn = conn
        self.session_id = session_id
        self.directory = db_path.parent.resolve() / session_id / "chat-history"
        self.hidden_through = self._hidden_through()

    def _hidden_through(self) -> int | None:
        if not self.conn.execute("SELECT 1 FROM sqlite_master WHERE name='session_meta'").fetchone():
            return None
        meta = dict(
            self.conn.execute(
                "SELECT key,value FROM session_meta WHERE session_id=? AND key IN (?,?)",
                (self.session_id, HISTORY_VERSION_KEY, HISTORY_REWRITE_KEY),
            )
        )
        return int(meta.get(HISTORY_REWRITE_KEY, 0)) if meta.get(HISTORY_VERSION_KEY) == "1" else None

    def _segments(self, start: int, end: int) -> Iterator[tuple[str, int, int, str]]:
        if not self.hidden_through:
            return
        yield from self.conn.execute(
            "SELECT filename,first_id,last_id,anchors FROM chat_history_segments "
            "WHERE session_id=? AND last_id>=? AND first_id<=? ORDER BY first_id",
            (self.session_id, start, end),
        )

    def _segment_path(self, filename: str) -> Path:
        path = self.directory / filename
        if self.session_id in (".", "..") or not _SEGMENT_NAME.fullmatch(filename) or path.resolve() != path:
            raise ValueError("Invalid chat history segment")
        return path

    def write_segment(self, rows: list[HistoryRow], hidden_through: int) -> None:
        """Publish a complete file before registering it in the rewrite transaction."""
        self.conn.execute(_SEGMENTS)
        if rows:
            filename = f"chat-{uuid.uuid4().hex}.jsonl"
            path = self._segment_path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            from datus.storage.semantic_model.artifact_file import atomic_write_bytes

            data = bytearray()
            anchors = []
            for row_id, raw, created_at, turn in rows:
                anchor = prompt_anchor(row_id, raw, len(data))
                if anchor is not None:
                    anchors.append(anchor)
                record = {"id": row_id, "turn": turn, "created_at": created_at, "item": json.loads(raw)}
                data.extend(json.dumps(record, ensure_ascii=False).encode() + b"\n")
            atomic_write_bytes(path, bytes(data), mode=0o600)
            self.conn.execute(
                "INSERT INTO chat_history_segments VALUES (?,?,?,?,?)",
                (self.session_id, filename, rows[0][0], rows[-1][0], json.dumps(anchors)),
            )
        self.conn.execute(
            "INSERT OR REPLACE INTO session_meta VALUES (?,?,?)",
            (self.session_id, HISTORY_REWRITE_KEY, str(hidden_through)),
        )

    def snapshot(self) -> int:
        head = self.conn.execute(
            "SELECT COALESCE(MAX(id),0) FROM agent_messages WHERE session_id=?", (self.session_id,)
        ).fetchone()[0]
        if self.hidden_through:
            archived = self.conn.execute(
                "SELECT COALESCE(MAX(last_id),0) FROM chat_history_segments WHERE session_id=?", (self.session_id,)
            ).fetchone()[0]
            return max(head, archived)
        return head

    def archived_anchors(self, snapshot: int) -> Iterator[Anchor]:
        for _, _, last, anchors in self._segments(1, snapshot):
            for row_id, digest, text, offset in json.loads(anchors):
                if row_id <= min(last, snapshot):
                    yield row_id, digest, text, offset

    def archived_rows(self, start: int, end: int) -> Iterator[HistoryRow]:
        for filename, _, last, anchors in self._segments(start, end):
            path = self._segment_path(filename)
            offset = max((a[3] for a in json.loads(anchors) if a[0] <= start), default=0)
            with path.open("rb") as stream:
                stream.seek(offset)
                for line in stream:
                    row = json.loads(line)
                    if row["id"] > min(last, end):
                        break
                    if row["id"] >= start:
                        yield row["id"], json.dumps(row["item"]), row["created_at"], row["turn"]

    def rows(self, start: int, end: int) -> Iterator[HistoryRow]:
        yield from self.archived_rows(start, end)
        yield from self.conn.execute(
            "SELECT m.id,m.message_data,m.created_at,s.user_turn_number FROM agent_messages m "
            "LEFT JOIN message_structure s ON s.session_id=m.session_id AND s.message_id=m.id "
            "WHERE m.session_id=? AND m.id BETWEEN ? AND ? ORDER BY m.id",
            (self.session_id, max(start, (self.hidden_through or 0) + 1), end),
        )

    def truncate(self, last_id: int) -> None:
        """Drop archived items beyond a rollback boundary from the visible history."""
        if self.hidden_through:
            self.conn.execute(
                "DELETE FROM chat_history_segments WHERE session_id=? AND first_id>?", (self.session_id, last_id)
            )
            self.conn.execute(
                "UPDATE chat_history_segments SET last_id=? WHERE session_id=? AND last_id>?",
                (last_id, self.session_id, last_id),
            )
