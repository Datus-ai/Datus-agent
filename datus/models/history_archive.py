# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Immutable JSONL segments for the user-visible history of a compacted session."""

import ast
import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

from datus.utils.message_utils import extract_user_input, is_compact_resume_text
from datus.utils.tool_archive import is_archived_output, parse_archived_marker

HISTORY_VERSION_KEY = "chat_history_version"
HISTORY_REWRITE_KEY = "chat_history_hidden_through"
# A result whose tool archive could not be read. This cannot be a session ID.
TASK_CHILD_UNAVAILABLE = "!"
# An ordinary tool call replaced an unfinished task with the same provider ID.
TASK_CALL_SHADOWED = "?"
_SEGMENT_NAME = re.compile(r"chat-[0-9a-f]{32}\.jsonl")
_SEGMENTS = """
CREATE TABLE IF NOT EXISTS chat_history_segments (
    session_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    first_id INTEGER NOT NULL,
    last_id INTEGER NOT NULL,
    anchors TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    PRIMARY KEY (session_id, first_id)
)
"""
_RECORDS = """
CREATE TABLE IF NOT EXISTS chat_history_records (
    session_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    byte_offset INTEGER NOT NULL,
    byte_length INTEGER NOT NULL,
    digest BLOB NOT NULL,
    starts_run INTEGER NOT NULL,
    PRIMARY KEY (session_id, message_id)
)
"""
_IMPLICIT_RUN_INDEX = """
CREATE INDEX IF NOT EXISTS idx_chat_history_implicit_run
ON chat_history_records(session_id, message_id) WHERE starts_run = 1
"""
_TASKS = """
CREATE TABLE IF NOT EXISTS chat_history_tasks (
    session_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    call_id TEXT NOT NULL,
    child_id TEXT,
    PRIMARY KEY (session_id, message_id, call_id)
)
"""
_TASK_EVENTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_chat_history_task_events
ON chat_history_tasks(session_id, call_id, message_id)
"""
_SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9_.-]+")
_TOOL_OUTPUT_NAME = re.compile(r"[0-9]{6,}_output_([0-9a-f]{8})\.txt")


class HistoryArchiveReadError(RuntimeError):
    """A registered archive cannot be read."""


def _parse_output(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw


def _archived_task_output(raw: str, directory: Path) -> str | None:
    marker = parse_archived_marker(raw)
    if marker is None:
        return None
    path = Path(marker["path"])
    filename = _TOOL_OUTPUT_NAME.fullmatch(path.name)
    if not filename:
        return None

    # The marker is tool output. Only open a result written inside this session's data directory.
    try:
        resolved = path.resolve()
        if resolved != directory / path.name or not resolved.is_file():
            return None
        payload = resolved.read_bytes()
        if not hashlib.sha256(payload).hexdigest().startswith(filename[1]):
            return None
        return payload.decode("utf-8")
    except (OSError, RuntimeError, ValueError):
        return None


def _task_result(raw: Any) -> Any:
    output = _parse_output(raw)
    if isinstance(output, list):
        output = _parse_output("\n".join(str(block.get("text", "")) for block in output if isinstance(block, dict)))
    return _parse_output(output.get("result", output)) if isinstance(output, dict) else output


def task_child_session_id(raw: Any, *, archive_dir: Path) -> str | None:
    """Resolve a task result, distinguishing an unreadable marker from a result with no child."""
    result = _task_result(raw)
    if is_archived_output(result):
        archived = _archived_task_output(result, archive_dir)
        if archived is None:
            return TASK_CHILD_UNAVAILABLE
        result = _task_result(archived)
    child = result.get("session_id") if isinstance(result, dict) else None
    if isinstance(child, str) and _SAFE_SESSION_ID.fullmatch(child) and ".." not in child:
        return child
    return None


def _tool_events(item: dict[str, Any]) -> Iterator[tuple[Any, str | None, Any]]:
    """Yield call ID, tool name for calls, and payload for results."""
    kind = item.get("type")
    if kind == "function_call":
        yield item.get("call_id"), item.get("name") or "", None
    elif kind == "function_call_output":
        yield item.get("call_id"), None, item.get("output")
    content = item.get("content")
    if not isinstance(content, list):
        return
    role = item.get("role")
    for block in content:
        if not isinstance(block, dict):
            continue
        if role == "assistant" and block.get("type") == "tool_use":
            yield block.get("id"), block.get("name") or "", None
        elif role == "user" and block.get("type") == "tool_result":
            yield block.get("tool_use_id"), None, block.get("content")


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


def _starts_implicit_run(item: dict[str, Any]) -> bool:
    if item.get("type") == "function_call":
        return True
    content = item.get("content")
    return (
        item.get("role") == "assistant"
        and isinstance(content, list)
        and any(
            isinstance(block, dict)
            and (
                block.get("type") in ("tool_use", "server_tool_use")
                or (block.get("type") in ("text", "output_text") and block.get("text"))
            )
            for block in content
        )
    )


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

    def _segments(
        self, start: int, end: int, *, legacy_task_index: bool = False
    ) -> Iterator[tuple[str, int, int, str, int]]:
        if not self.hidden_through:
            return
        query = (
            "SELECT filename,first_id,last_id,anchors,byte_size FROM chat_history_segments s "
            "WHERE session_id=? AND last_id>=? AND first_id<=?"
        )
        if legacy_task_index:
            query += (
                " AND NOT EXISTS (SELECT 1 FROM chat_history_tasks t "
                "WHERE t.session_id=s.session_id AND t.message_id=s.first_id AND t.call_id='')"
            )
        yield from self.conn.execute(query + " ORDER BY first_id", (self.session_id, start, end))

    def _segment_path(self, filename: str) -> Path:
        path = self.directory / filename
        if self.session_id in (".", "..") or not _SEGMENT_NAME.fullmatch(filename) or path.resolve() != path:
            raise ValueError("Invalid chat history segment")
        return path

    def write_segment(self, rows: list[HistoryRow], hidden_through: int) -> None:
        """Publish a complete file before registering it in the rewrite transaction."""
        self.conn.execute(_SEGMENTS)
        self.conn.execute(_RECORDS)
        self.conn.execute(_IMPLICIT_RUN_INDEX)
        self.conn.execute(_TASKS)
        self.conn.execute(_TASK_EVENTS_INDEX)
        if rows:
            filename = f"chat-{uuid.uuid4().hex}.jsonl"
            path = self._segment_path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            from datus.storage.semantic_model.artifact_file import atomic_write_bytes

            data = bytearray()
            anchors = []
            records = []
            task_index_complete = True
            for row_id, raw, created_at, turn in rows:
                anchor = prompt_anchor(row_id, raw, len(data))
                if anchor is not None:
                    anchors.append(anchor)
                item = json.loads(raw)
                if not self._index_tasks(row_id, item):
                    task_index_complete = False
                record = {"id": row_id, "turn": turn, "created_at": created_at, "item": item}
                line = json.dumps(record, ensure_ascii=False).encode() + b"\n"
                records.append(
                    (
                        self.session_id,
                        row_id,
                        len(data),
                        len(line),
                        hashlib.sha256(line).digest(),
                        _starts_implicit_run(item),
                    )
                )
                data.extend(line)
            atomic_write_bytes(path, bytes(data), mode=0o600)
            self.conn.executemany("INSERT INTO chat_history_records VALUES (?,?,?,?,?,?)", records)
            self.conn.execute(
                "INSERT INTO chat_history_segments VALUES (?,?,?,?,?,?)",
                (self.session_id, filename, rows[0][0], rows[-1][0], json.dumps(anchors), len(data)),
            )
            # Without this marker, a deferred result can be recovered once its legacy segment is readable.
            if task_index_complete:
                self.conn.execute("INSERT INTO chat_history_tasks VALUES (?,?,'','')", (self.session_id, rows[0][0]))
        self.conn.execute(
            "INSERT OR REPLACE INTO session_meta VALUES (?,?,?)",
            (self.session_id, HISTORY_REWRITE_KEY, str(hidden_through)),
        )

    def _index_tasks(self, row_id: int, item: dict[str, Any]) -> bool:
        complete = True
        for call_id, tool_name, output in _tool_events(item):
            if not isinstance(call_id, str) or not call_id:
                continue
            child = None
            if tool_name != "task":
                prior = self.conn.execute(
                    "SELECT message_id,child_id FROM chat_history_tasks "
                    "WHERE session_id=? AND call_id=? AND message_id<? ORDER BY message_id DESC LIMIT 1",
                    (self.session_id, call_id, row_id),
                ).fetchone()
                if prior is None or prior[1] is not None:
                    continue
                if tool_name is None:
                    # Legacy indexes omit ordinary calls and some task results.
                    # A later event in those segments means this task is no longer waiting for a result.
                    try:
                        later = self._archived_tool_event(
                            call_id, prior[0] + 1, min(row_id - 1, self.hidden_through or 0), legacy_task_index=True
                        )
                    except HistoryArchiveReadError:
                        complete = False
                        continue
                    if later is not None:
                        continue
                child = (
                    TASK_CALL_SHADOWED
                    if tool_name is not None
                    else task_child_session_id(output, archive_dir=self.directory.parent / "data") or ""
                )
            self.conn.execute(
                "INSERT OR IGNORE INTO chat_history_tasks VALUES (?,?,?,?)",
                (self.session_id, row_id, call_id, child),
            )
        return complete

    def indexed_task_children(self, start: int, end: int) -> dict[tuple[int, str], str]:
        return {
            (row_id, call_id): child
            for row_id, call_id, child in self.conn.execute(
                "SELECT message_id,call_id,child_id FROM chat_history_tasks "
                "WHERE session_id=? AND message_id BETWEEN ? AND ? AND child_id NOT IN ('',?,?)",
                (self.session_id, start, end, TASK_CHILD_UNAVAILABLE, TASK_CALL_SHADOWED),
            )
        }

    def _archived_tool_event(
        self, call_id: str, start: int, end: int, *, legacy_task_index: bool
    ) -> tuple[str | None, Any] | None:
        if start > end:
            return None
        with closing(self.archived_rows(start, end, legacy_task_index=legacy_task_index)) as rows:
            for _, raw, _, _ in rows:
                for event_id, tool_name, output in _tool_events(json.loads(raw)):
                    if event_id == call_id:
                        return tool_name, output
        return None

    def unindexed_task_events(self, before: int) -> Iterator[tuple[int, str, str | None, Any]]:
        """Yield the indexed row and ID alongside its result or superseding call."""
        if not self.hidden_through:
            return
        candidates = self.conn.execute(
            "WITH task_events AS ("
            "SELECT message_id,call_id,child_id,"
            "LEAD(message_id,1,:before) OVER call_events AS next_id,"
            "LEAD(child_id) OVER call_events AS next_child "
            "FROM chat_history_tasks WHERE session_id=:session AND message_id<:before AND call_id!='' "
            "WINDOW call_events AS (PARTITION BY call_id ORDER BY message_id)) "
            "SELECT message_id,call_id,child_id,next_id FROM task_events "
            "WHERE child_id=:unavailable OR "
            "(child_id IS NULL AND (next_child IS NULL OR next_child IN ('',:shadowed)))",
            {
                "session": self.session_id,
                "before": before,
                "unavailable": TASK_CHILD_UNAVAILABLE,
                "shadowed": TASK_CALL_SHADOWED,
            },
        )
        for row_id, call_id, child, next_id in candidates:
            # Old writers left only the call indexed. Search until its result or
            # the next use of the call ID, skipping segments with complete indexes.
            start, end = (row_id, row_id) if child else (row_id + 1, min(next_id - 1, self.hidden_through))
            event = self._archived_tool_event(call_id, start, end, legacy_task_index=child is None)
            if event is not None:
                tool_name, output = event
                yield row_id, call_id, tool_name, output

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
        for _, _, last, anchors, _ in self._segments(1, snapshot):
            for row_id, digest, text, offset in json.loads(anchors):
                if row_id <= min(last, snapshot):
                    yield row_id, digest, text, offset

    def archived_rows(self, start: int, end: int, *, legacy_task_index: bool = False) -> Iterator[HistoryRow]:
        for filename, first, last, _, byte_size in self._segments(start, end, legacy_task_index=legacy_task_index):
            try:
                path = self._segment_path(filename)
                with path.open("rb") as stream:
                    if os.fstat(stream.fileno()).st_size != byte_size:
                        raise ValueError("Unexpected archive size")
                    indexed = self.conn.execute(
                        "SELECT message_id,byte_offset,byte_length,digest FROM chat_history_records "
                        "WHERE session_id=? AND message_id BETWEEN ? AND ? ORDER BY message_id",
                        (self.session_id, max(first, start), min(last, end)),
                    )
                    for row_id, offset, length, digest in indexed:
                        stream.seek(offset)
                        line = stream.read(length)
                        if hashlib.sha256(line).digest() != digest:
                            raise ValueError("Archive record checksum mismatch")
                        row = json.loads(line)
                        if row["id"] != row_id or not isinstance(row["item"], dict):
                            raise ValueError("Invalid archive message")
                        yield row_id, json.dumps(row["item"]), row["created_at"], row["turn"]
            except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
                raise HistoryArchiveReadError("Chat history archive is temporarily unavailable") from exc

    def implicit_run_start(self, before: int) -> int | None:
        """Find assistant activity before the first prompt without opening an archive."""
        if self.hidden_through:
            row = self.conn.execute(
                "SELECT message_id FROM chat_history_records "
                "WHERE session_id=? AND message_id<? AND starts_run=1 ORDER BY message_id LIMIT 1",
                (self.session_id, before),
            ).fetchone()
            if row:
                return row[0]
        row = self.conn.execute(
            "SELECT id FROM agent_messages WHERE session_id=? AND id>? AND id<? AND ("
            "json_extract(message_data,'$.type')='function_call' OR ("
            "json_extract(message_data,'$.role')='assistant' AND EXISTS("
            "SELECT 1 FROM json_each(message_data,'$.content') b WHERE b.type='object' AND ("
            "json_extract(b.value,'$.type') IN ('tool_use','server_tool_use') OR ("
            "json_extract(b.value,'$.type') IN ('text','output_text') "
            "AND json_extract(b.value,'$.text')!=''))))) ORDER BY id LIMIT 1",
            (self.session_id, self.hidden_through or 0, before),
        ).fetchone()
        return row[0] if row else None

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
                "DELETE FROM chat_history_records WHERE session_id=? AND message_id>?", (self.session_id, last_id)
            )
            self.conn.execute(
                "DELETE FROM chat_history_tasks WHERE session_id=? AND message_id>?", (self.session_id, last_id)
            )
            self.conn.execute(
                "DELETE FROM chat_history_segments WHERE session_id=? AND first_id>?", (self.session_id, last_id)
            )
            self.conn.execute(
                "UPDATE chat_history_segments SET last_id=? WHERE session_id=? AND last_id>?",
                (last_id, self.session_id, last_id),
            )
