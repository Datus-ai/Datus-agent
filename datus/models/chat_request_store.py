"""Durable receipts for API chat sends, separate from the SSE replay buffer."""

import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

CHAT_REQUEST_WINDOW_MS = 60 * 60 * 1000
_MAX_CLOCK_SKEW_MS = 5 * 60 * 1000


class ChatRequestExpired(Exception):
    """The request ID is outside the safe admission window."""


@dataclass(frozen=True)
class ChatRequestReceipt:
    session_id: str
    run_id: str


class ChatRequestStore:
    """Keep only the session/run identity in the existing scoped session directory."""

    def __init__(
        self,
        directory: str,
        request_id: str,
        *,
        project_id: str | None,
        user_id: str | None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.request_id = request_id
        self._issued_at = int(request_id.partition("-")[0])
        self._expires_at = self._issued_at + CHAT_REQUEST_WINDOW_MS
        self._key = (project_id or "", user_id or "", request_id)
        self._path = Path(directory) / ".chat-request-receipts.sqlite3"
        self._clock = clock

    def _check_window(self) -> int:
        now = int(self._clock() * 1000)
        if now >= self._expires_at or self._issued_at > now + _MAX_CLOCK_SKEW_MS:
            raise ChatRequestExpired
        return now

    def _connect(self) -> sqlite3.Connection:
        if self._path.is_symlink():
            raise ValueError("Invalid chat request storage path")
        conn = sqlite3.connect(self._path, timeout=5)
        try:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS chat_requests (
                    project_id TEXT NOT NULL, user_id TEXT NOT NULL, request_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, run_id TEXT NOT NULL, expires_at INTEGER NOT NULL,
                    PRIMARY KEY (project_id, user_id, request_id)
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS chat_requests_expiry ON chat_requests(expires_at)")
        except BaseException:
            conn.close()
            raise
        return conn

    def _select(self, conn: sqlite3.Connection) -> ChatRequestReceipt | None:
        row = conn.execute(
            "SELECT session_id, run_id FROM chat_requests WHERE project_id=? AND user_id=? AND request_id=?",
            self._key,
        ).fetchone()
        return ChatRequestReceipt(*row) if row else None

    def find(self) -> ChatRequestReceipt | None:
        """Return the receipt without reserving the ID."""
        self._check_window()
        if not self._path.exists():
            return None
        with closing(self._connect()) as conn:
            return self._select(conn)

    def claim(self, session_id: str, run_id: str) -> ChatRequestReceipt | None:
        """Persist the winning turn before dispatch, or return the existing receipt."""
        self._check_window()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            # Check again after acquiring the lock. Expired IDs can never be
            # admitted again, even after their receipts have been pruned.
            now = self._check_window()
            conn.execute("DELETE FROM chat_requests WHERE expires_at <= ?", (now,))
            inserted = conn.execute(
                "INSERT INTO chat_requests VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(project_id, user_id, request_id) DO NOTHING",
                (*self._key, session_id, run_id, self._expires_at),
            )
            return None if inserted.rowcount else self._select(conn)
