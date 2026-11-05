# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Durable host-dispatched first turns in the normal chat history database.

The host supplies a stable session/request identity and validates its opaque
metadata on retries. The Agent owns schema, atomic acceptance and the execution
fence. Authorization and deciding whether an interrupted turn may retry belong
to the host; acquiring the fence alone never starts or replays a model call.
"""

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from datus.models.sqlite_session import SESSION_META_TABLE, SESSION_TITLE_KEY, DatusSQLiteSession
from datus.utils.exceptions import DatusException, ErrorCode


def _invalid(message: str) -> DatusException:
    return DatusException(ErrorCode.STORAGE_INVALID_ARGUMENT, message=message)


class AcceptedTurnStore:
    def __init__(self, directory: str, session_id: str, *, namespace: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id) or not namespace or namespace == SESSION_TITLE_KEY:
            raise _invalid("Invalid accepted turn identity")
        self.session_id = session_id
        self.namespace = namespace
        self.path = Path(directory) / f"{session_id}.db"
        if self.path.is_symlink():
            raise _invalid("Invalid session path")

    def read(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=10) as conn:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='session_meta'").fetchone():
                return None
            row = conn.execute(
                "SELECT value FROM session_meta WHERE session_id=? AND key=?", (self.session_id, self.namespace)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def accept(
        self,
        request_id: str,
        message: str,
        *,
        metadata: dict[str, Any],
        validate_existing: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Commit the opening, receipt and host metadata together, or recover it.

        A retry must have the same request ID and satisfy validate_existing under
        the transaction lock. The validator must be synchronous and perform no I/O.
        It lets the host distinguish immutable evidence from refreshable metadata.
        A session with messages but no matching receipt is never adopted.
        """
        if not request_id or not message:
            raise _invalid("Request ID and opening message are required")
        if set(metadata) & {"sessionId", "messageId", "executionId", "status", "safeErrorCode", "openingDigest"}:
            raise _invalid("Host metadata cannot overwrite receipt fields")
        # Fail before touching history if metadata cannot be persisted.
        json.dumps(metadata, allow_nan=False)
        opening_digest = hashlib.sha256(message.encode()).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        session = DatusSQLiteSession(session_id=self.session_id, db_path=str(self.path), create_tables=True)
        conn = session._get_connection()
        with session._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(SESSION_META_TABLE)
                row = conn.execute(
                    "SELECT value FROM session_meta WHERE session_id=? AND key=?", (self.session_id, self.namespace)
                ).fetchone()
                if row:
                    existing = json.loads(row[0])
                    if (
                        existing["executionId"] != request_id
                        or existing.get("openingDigest", opening_digest) != opening_digest
                    ):
                        raise _invalid("Request conflict")
                    validate_existing(existing)
                    conn.rollback()
                    return existing
                # A conflicting pre-existing chat may never be adopted as an intent.
                if conn.execute("SELECT 1 FROM agent_messages LIMIT 1").fetchone():
                    raise _invalid("Request conflict")
                prompt = message
                conn.execute("INSERT OR IGNORE INTO agent_sessions (session_id) VALUES (?)", (self.session_id,))
                message_id = conn.execute(
                    "INSERT INTO agent_messages (session_id,message_data) VALUES (?,?)",
                    (self.session_id, json.dumps({"role": "user", "content": prompt})),
                ).lastrowid
                conn.execute(
                    "INSERT INTO message_structure (session_id,message_id,branch_id,message_type,sequence_number,"
                    "user_turn_number,branch_turn_number) VALUES (?,?,?,'user',1,1,1)",
                    (self.session_id, message_id, session._current_branch_id),
                )
                intent = {
                    **metadata,
                    "sessionId": self.session_id,
                    "messageId": str(message_id),
                    "executionId": request_id,
                    "openingDigest": opening_digest,
                    "status": "pending",
                    "safeErrorCode": None,
                }
                conn.executemany(
                    "INSERT INTO session_meta(session_id,key,value) VALUES (?,?,?)",
                    [
                        (self.session_id, self.namespace, json.dumps(intent)),
                        (self.session_id, SESSION_TITLE_KEY, prompt[:500]),
                    ],
                )
                conn.commit()
                return intent
            except BaseException:
                conn.rollback()
                raise
            finally:
                session.close()

    def update(self, status: str, safe_error: str | None = None) -> dict[str, Any]:
        with sqlite3.connect(self.path, timeout=10) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT value FROM session_meta WHERE session_id=? AND key=?", (self.session_id, self.namespace)
            ).fetchone()
            if row is None:
                raise _invalid("Accepted turn is unavailable")
            intent = json.loads(row[0])
            intent.update(status=status, safeErrorCode=safe_error)
            conn.execute(
                "UPDATE session_meta SET value=? WHERE session_id=? AND key=?",
                (json.dumps(intent), self.session_id, self.namespace),
            )
            return intent

    @contextmanager
    def execution_lock(self) -> Iterator[bool]:
        # POSIX hosts use flock on a separate inode: locking the database itself
        # would conflict with SQLite locks on macOS. Unsupported hosts fail closed.
        import fcntl

        # SQLite uses OS locks on the database inode (including flock on macOS).
        # Keep the execution fence on a separate inode so history and metadata
        # remain readable while the turn runs. Keep this stable file for the
        # session's lifetime; project/user directory cleanup removes it.
        with self.path.with_suffix(".execution.lock").open("a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class AcceptedOpeningSession(DatusSQLiteSession):
    """The durable opening is visible to history, but enters the first model once.

    This adapter is only installed on the executing node, not on history readers.
    Its independent guard survives the SDK's skip_persisted_input_once reset.
    """

    def __init__(self, *args: Any, opening: str, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.opening = opening
        self.opening_pending = True

    def is_opening(self, item: dict[str, Any]) -> bool:
        from datus.utils.message_utils import extract_user_input

        return item.get("role") == "user" and extract_user_input(item.get("content")) == self.opening

    async def get_items(self, limit: int | None = None) -> list[dict[str, Any]]:
        items = await super().get_items(limit)
        if self.opening_pending:
            return [item for item in items if not self.is_opening(item)]
        return items

    async def add_items(self, items: list[dict[str, Any]]) -> None:
        if self.opening_pending:
            for index, item in enumerate(items):
                if self.is_opening(item):
                    items = items[:index] + items[index + 1 :]
                    break
        await super().add_items(items)

    async def replace_items(self, items: list[dict[str, Any]], **kwargs: Any) -> None:
        opening = {"role": "user", "content": self.opening}
        await super().replace_items([opening, *[item for item in items if not self.is_opening(item)]], **kwargs)
